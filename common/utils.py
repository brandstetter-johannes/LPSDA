import os
import h5py
import numpy as np
import torch
import random
import sys

from typing import Tuple
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from common.augmentation import KdV_augmentation, KS_augmentation, Heat_augmentation, to_coords
from common.augmentation import Subalgebra
from equations.PDEs import PDE, KdV


class HDF5Dataset(Dataset):
    """
    Load samples of an PDE Dataset, get items according to PDE.
    """
    def __init__(self, path: str,
                 mode: str,
                 nt: int,
                 nx: int,
                 shift: str,
                 pde: PDE = None,
                 dtype=torch.float64,
                 load_all: bool=False):
        """Initialize the dataset object.
        Args:
            path: path to dataset
            mode: [train, valid, test]
            nt: temporal resolution
            nx: spatial resolution
            shift: [fourier, linear]
            pde: PDE at hand
            dtype: floating precision of data
            load_all: load all the data into memory
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = dtype
        self.data = f[self.mode]
        self.dataset = f'pde_{nt}-{nx}'
        self.pde = PDE() if pde is None else pde
        self.augmentation = None
        self.shift = 'fourier' if shift is None else shift

        # Generators which are used for LSDAP
        # Time generator is treated a bit differently and is implemented in the training loop
        if str(self.pde) == 'KdV':
            self.augmentation = KdV_augmentation(self.pde.max_x_shift,
                                                 self.pde.max_velocity,
                                                 self.pde.max_scale)
        elif str(self.pde) == 'KS':
            self.augmentation = KS_augmentation(self.pde.max_x_shift,
                                                self.pde.max_velocity)

        # For the Heat equation, infinite subalgebra is evoked in __getitem__
        elif str(self.pde) == 'Heat':
            self.augmentation = Heat_augmentation(self.pde.max_x_shift)

        if load_all:
            data = {self.dataset: self.data[self.dataset][:]}
            f.close()
            self.data = data

    def __len__(self):
        return self.data[self.dataset].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns data items for batched training/validation/testing.
        Args:
            idx: data index
        Returns:
            torch.Tensor: data trajectory used for training/validation/testing
            torch.Tensor: dx
            torch.Tensor: dt
        """
        u = self.data[self.dataset][idx]
        x = self.data['x'][idx]
        t = self.data['t'][idx]

        if str(self.pde) == 'Heat':
            X = to_coords(torch.tensor(x), torch.tensor(t))
            sol = (torch.tensor(u), X)
            if self.mode == "train" and self.augmentation is not None:
                # Obtain a random second trajectory which is used for data mixing
                idx2 = torch.randint(0, self.__len__(), (1,))
                u2 = self.data[self.dataset][idx2]
                sol2 = (torch.tensor(u2), X)
                # Cole-Hopf transformation
                # For alpha > 0 mixing occurs
                sol = self.pde.subalgebra(sol, sol2, alpha=self.pde.alpha)
                # Remaining generators for data augmentation
                sol = self.augmentation(sol)

            else:
                # Cole-Hopf transformation
                # For alpha == 0 no, mixing between different trajectories
                sol = self.pde.subalgebra(sol, alpha=0.)

            # Scaling of the whole trajectory, otherwise amplitudes are pretty high
            u = sol[0]
            u = u / 100
            X = sol[1]
            # Only needed when scaling generator is added
            dx = X[0, 1, 0] - X[0, 0, 0]
            dt = X[1, 0, 1] - X[0, 0, 1]

        else:
            X = to_coords(torch.tensor(x), torch.tensor(t))
            sol = (torch.tensor(u), X)

            # Data augmentation using the defined generators for the equation at hand
            if self.mode == "train" and self.augmentation is not None:
                sol = self.augmentation(sol, self.shift)

            u = sol[0]
            X = sol[1]
            dx = X[0, 1, 0] - X[0, 0, 0]
            dt = X[1, 0, 1] - X[0, 0, 1]

        return u.float(), dx.float(), dt.float()


class DataCreator(nn.Module):
    """
    Helper class to construct input data and labels.
    """
    def __init__(self,
                 time_history,
                 time_future,
                 t_resolution,
                 x_resolution
                 ):
        """
        Initialize the DataCreator object.
        Args:
            time_history (int): how many time steps are used for PDE prediction
            time_future (int): how many time does the solver predict into the future
            t_resolution: temporal resolution
            x_resolution: spatial resolution
        """
        super().__init__()
        self.time_history = time_history
        self.time_future = time_future
        self.t_res = t_resolution
        self.x_res = x_resolution

    def create_data(self, datapoints: torch.Tensor, start_time: list, pf_steps=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data of PDEs for training, validation and testing.
        Args:
            datapoints (torch.Tensor): trajectory input
            start_time (int list): list of different starting times for different trajectories in one batch
            pf_steps (int): push forward steps
        Returns:
            torch.Tensor: neural network input data
            torch.Tensor: neural network labels
        """
        data = []
        labels = []
        # Loop over batch and different starting points
        # For every starting point, we take the number of time_history points as training data
        # and the number of time future data as labels
        for (dp, start) in zip(datapoints, start_time):
            end_time = start+self.time_history
            d = dp[start:end_time]
            target_start_time = end_time + self.time_future * pf_steps
            target_end_time = target_start_time + self.time_future
            l = dp[target_start_time:target_end_time]

            data.append(d.unsqueeze(dim=0))
            labels.append(l.unsqueeze(dim=0))

        return torch.cat(data, dim=0), torch.cat(labels, dim=0)
