import torch
import random

from typing import Tuple
from torch import nn, optim
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, DataCreator
from equations.PDEs import *


def bootstrap(x: torch.Tensor, Nboot: int=64, binsize: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bootstrapping mean or median to obtain standard deviation.
    Args:
        x (torch.Tensor): input tensor, which contains all the results on the different trajectories of the set at hand
        Nboot (int): number of bootstrapping steps, 64 is quite common default value
        binsize (int):
    Returns:
        torch.Tensor: bootstrapped mean/median of input
        torch.Tensor: bootstrapped variance of input
    """
    boots = []
    x = x.reshape(-1, binsize, *x.shape[1:])
    for i in range(Nboot):
        boots.append(torch.mean(x[torch.randint(len(x), (len(x),))], axis=(0, 1)))
    return torch.tensor(boots).mean(), torch.tensor(boots).std()

def training_loop(pde: PDE,
                  model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  data_creator: DataCreator,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory.
    Args:
        pde (PDE): PDE at hand
        model (torch.nn.Module): Pytorch model used for training
        unrolling (list): list of unrolling steps of size batch_size, only need for pushforward trick
        batch_size (int): batch size
        optimizer (torch.optim): chosen optimizer
        loader (DataLoader): train/valid/test dataloader
        data_creator (DataCreator): DataCreator to construct input data and labels
        criterion (torch.nn.modules.loss): loss criterion
        device: device (cpu/gpu)
    Returns:
        torch.Tensor: losses for whole dataset
    """
    losses = []
    for (u, dx, dt) in loader:
        optimizer.zero_grad()

        # Select the number of pushforward steps
        pf_steps = random.choice(unrolling)
        # Length of trajectory
        time_resolution = data_creator.t_res
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - data_creator.time_history
        # Number of future points to predict
        max_start_time = reduced_time_resolution - data_creator.time_future * (1 + pf_steps)

        # Choose initial random time point at the PDE solution manifold.
        # If time translation is chosen for data augmentation we use additional starting points,
        # and use all starting points of the trajectory.
        # If time translation is not chosen we use only those starting points for training
        # which are used for unrolling in the valid/test routing.
        if pde.time_shift:
            start_time = random.choices([t for t in range(max_start_time+1)], k=batch_size)
        else:
            start_time = random.choices([t for t in range(data_creator.time_history,
                                                          max_start_time + 1, data_creator.time_history)], k=batch_size)

        data, labels = data_creator.create_data(u, start_time, pf_steps)
        data, labels = data.to(device), labels.to(device)

        # Change [batch, time, space] -> [batch, space, time]
        data = data.permute(0, 2, 1)

        # The unrolling of the equation which serves as input at the current step
        with torch.no_grad():
            for _ in range(pf_steps):

                pred = model(data, dx, dt)

                data = torch.cat([data, pred], -1)
                data = data[..., -data_creator.time_history:]

        pred = model(data, dx, dt)

        loss = criterion(pred.permute(0, 2, 1), labels)
        loss = loss.sum()
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()

    losses = torch.stack(losses)
    return losses

def test_timestep_losses(model: torch.nn.Module,
                         batch_size: int,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> None:
    """
    Tests losses at specific time points of the trajectories. Helps to understand loss behavior for full
    trajectory unrolling.
    Args:
        model (torch.nn.Module): Pytorch model used for training
        batch_size (int): batch size
        loader (DataLoader): train/valid/test dataloader
        data_creator (DataCreator): DataCreator to construct input data and labels
        criterion (torch.nn.modules.loss): loss criterion
        device: device (cpu/gpu)
    Returns:
        None
    """
    # Length of trajectory
    time_resolution = data_creator.t_res
    # Max number of previous points solver can eat
    reduced_time_resolution = time_resolution - data_creator.time_history
    # Number of future points to predict
    max_start_time = reduced_time_resolution - data_creator.time_future
    # The first time steps are used for data augmentation according to time translate
    # We ignore these timesteps in the testing
    start_time = [t for t in range(data_creator.time_history, max_start_time+1, data_creator.time_future)]
    for start in start_time:

        losses = []
        for (u, dx, dt) in loader:
            with torch.no_grad():
                end_time = start + data_creator.time_history
                target_end_time = end_time + data_creator.time_future
                data = u[:, start:end_time]
                labels = u[:, end_time: target_end_time]
                data, labels = data.to(device), labels.to(device)

                data = data.permute(0, 2, 1)
                pred = model(data, dx, dt)
                loss = criterion(pred.permute(0, 2, 1), labels)
                loss = loss.sum()
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print(f'Input {start} - {start + data_creator.time_history}, mean loss {torch.mean(losses)}')

def test_unrolled_losses(model: torch.nn.Module,
                         batch_size: int,
                         nx: int,
                         loader: DataLoader,
                         data_creator: DataCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> (torch.Tensor, torch.Tensor):
    """
    Tests unrolled losses for full trajectory unrolling.
    Args:
        model (torch.nn.Module): Pytorch model used for training
        batch_size (int): batch_size
        nx (int): spatial resolution
        loader (DataLoader): train/valid/test dataloader
        data_creator (DataCreator): DataCreator to construct input data and labels
        criterion (torch.nn.modules.loss): loss criterion
        device: device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled losses for whole dataset
        torch.Tensor: unrolled normalized losses for whole dataset
    """
    time_resolution = data_creator.t_res
    # Max number of previous points solver can eat
    reduced_time_resolution = time_resolution - data_creator.time_history
    # Number of future points to predict
    max_start_time = reduced_time_resolution - data_creator.time_future

    losses, nlosses = [], []
    for (u, dx, dt) in loader:
        losses_tmp, nlosses_tmp = [], []
        with torch.no_grad():
            # the first time steps are used for data augmentation according to time translate
            # we ignore these timesteps in the testing
            for start in range(data_creator.time_history, max_start_time+1, data_creator.time_future):

                end_time = start + data_creator.time_history
                target_end_time = end_time + data_creator.time_future
                if start == data_creator.time_history:
                    data = u[:, start:end_time].to(device)
                    data = data.permute(0, 2, 1)
                else:
                    data = torch.cat([data, pred], -1)
                    data = data[..., -data_creator.time_history:]
                labels = u[:, end_time: target_end_time].to(device)

                if str(model) == "FNO2d":
                    data_ = data[..., None, :].repeat([1, 1, data_creator.time_future, 1])
                    pred = model(data_, dx, dt)[..., 0]
                else:
                    pred = model(data, dx, dt)

                loss = criterion(pred.permute(0, 2, 1), labels)
                nlabels = torch.mean(labels ** 2, dim=-1, keepdim=True)
                nloss = loss / nlabels
                loss, nloss = loss.sum(), nloss.sum()
                loss, nloss = loss / nx / batch_size, nloss / nx / batch_size
                losses_tmp.append(loss)
                nlosses_tmp.append(nloss)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        nlosses.append(torch.sum(torch.stack(nlosses_tmp)))

    losses = torch.stack(losses)
    nlosses = torch.stack(nlosses)

    return losses, nlosses




