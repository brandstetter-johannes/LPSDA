import argparse
import os
# One BLAS thread per process: multi-threaded BLAS sums in a varying order, so the same seed could give
# slightly different trajectories run to run, and it is slower for these small systems anyway.
# Set before numpy is imported; an explicit value in the environment still wins.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import sys
import math
import numpy as np
import torch
import h5py
import random

from typing import Tuple
from functools import partial
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from copy import copy
from datetime import datetime
from equations.PDEs import PDE, KdV, KS, Heat

# Mixed into per-trajectory seeds so train, valid and test never share initial conditions
MODE_IDS = {"train": 0, "valid": 1, "test": 2}


def check_files(pde: PDE, modes: dict) -> None:
    """
    Check if data files exist and replace them if wanted.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        modes (dict): mode ([train, valid, test]), replace, num_samples, training suffix
    Returns:
            None
    """
    for mode, replace, num_samples, suffix in modes:
        save_name = "data/" + "_".join([str(pde), mode])
        if mode == "train":
            save_name = save_name + "_" + str(num_samples)
        if suffix:
            save_name = save_name + "_" + suffix
        if (replace == True):
            if os.path.exists(f'{save_name}.h5'):
                os.remove(f'{save_name}.h5')
                print(f'File {save_name}.h5 is deleted.')
            else:
                print(f'No file {save_name}.h5 exists yet.')
        else:
            print(f'File {save_name}.h5 is kept.')

def check_directory() -> None:
    """
    Check if data and log directories exist, and create otherwise.
    """
    if os.path.exists(f'data'):
        print(f'Data directory exists and will be written to.')
    else:
        os.mkdir(f'data')
        print(f'Data directory created.')

    if not os.path.exists(f'data/log'):
        os.mkdir(f'data/log')

def initial_conditions(A: np.ndarray, phi: np.ndarray, l: np.ndarray, L: float):
    """
    Return initial conditions based on initial parameters.
    Args:
        A (np.ndarray): amplitude of different sine waves
        phi (np.ndarray): phase shift of different sine waves
        l (np.ndarray): frequency of different sine waves
        L (float): length of the spatial domain
    Returns:
        None
    """
    def fnc(x):
        u = np.sum(A * np.sin(2 * np.pi * l * x / L + phi), -1)
        return u
    return fnc

def params(pde: PDE, batch_size: int, device: torch.cuda.device="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get initial parameters for KdV, KS, and Burgers' equation.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        np.ndarray: amplitude of different sin waves
        np.ndarray: phase shift
        np.ndarray: space dependent frequency
    """
    A = np.random.rand(1, pde.N) - 0.5
    phi = 2.0 * np.pi * np.random.rand(1, pde.N)
    l = np.random.randint(pde.lmin, pde.lmax, (1, pde.N))
    return A, phi, l

def inv_cole_hopf(psi0: np.ndarray, scale: float = 10.) -> np.ndarray:
    """
    Inverse Cole-Hopf transformation to obtain Heat equation out of initial conditions of Burgers' equation.
    Args:
        psi0 (np.ndarray): Burgers' equation (at arbitrary timestep) which gets transformed into Heat equation
        scale (float): scaling factor for transformation
    Returns:
        np.ndarray: transformed Heat equation
    """
    psi0 = psi0 - np.amin(psi0)
    psi0 = scale * 2 * ((psi0 / np.amax(psi0)) - 0.5)
    psi0 = np.exp(psi0)
    return psi0

def limit_threads() -> None:
    """
    Keep each worker process on a single BLAS thread, so that num_workers processes do not oversubscribe the cores.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass

def solve_trajectory(pde: PDE, mode: str, seed: int, idx: int) -> Tuple[bool, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Sample initial conditions and solve a single trajectory.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        mode (str): [train, valid, test]
        seed (int): base seed; if set, the random state is reseeded from (seed, mode, idx), so each trajectory
                    is reproducible and does not depend on how many workers generate the dataset
        idx (int): index of the trajectory
    Returns:
        bool: whether the solver succeeded
        np.ndarray: the last nt_effective time steps of the solution
        np.ndarray: spatial coordinates
        float: dx
        np.ndarray: the last nt_effective time coordinates
        float: dt
    """
    if seed is not None:
        np.random.seed([seed, MODE_IDS[mode], idx])

    pde_string = str(pde)
    # Tolerance of the solver
    tol = 1e-9
    nt = pde.grid_size[0]
    nx = pde.grid_size[1]

    # For the Heat (Burgers') equation, a fixed grid is used
    # For KdV and KS, the grid is flexible ->  this is due to scale symmetries which we want to exploit
    if pde_string == 'Heat':
        T = pde.tmax
        L = pde.L

    else:
        t1 = pde.tmax - pde.tmax / 10
        t2 = pde.tmax + pde.tmax / 10
        T = (t1 - t2) * np.random.rand() + t2
        l1 = pde.L - pde.L / 10
        l2 = pde.L + pde.L / 10
        L = (l1 - l2) * np.random.rand() + l2

    t = np.linspace(pde.tmin, T, nt)
    x = np.linspace(0, (1 - 1.0 / nx) * L, nx)

    # Parameters for initial conditions
    A, omega, l = params(pde, 1, device=pde.device)

    # Initial condition of the equation at end
    u0 = initial_conditions(A, omega, l, L)(x[:, None])

    # We use the initial condition of Burgers' equation and inverse Cole-Hopf transform it into the Heat equation
    if pde_string == 'Heat':
        u0 = inv_cole_hopf(u0)

    # We use pseudospectral reconstruction as spatial solver
    spatial_method = pde.pseudospectral_reconstruction

    # Solving for the full trajectories
    # For integration in time, we use an implicit Runge-Kutta method of Radau IIA family, order 5
    solved_trajectory = solve_ivp(fun=spatial_method,
                                  t_span=[t[0], t[-1]],
                                  y0=u0,
                                  method='Radau',
                                  t_eval=t,
                                  args=(L, ),
                                  atol=tol,
                                  rtol=tol)

    if not solved_trajectory.success:
        return False, None, None, None, None, None
    sol = solved_trajectory.y.T[-pde.nt_effective:]
    return True, sol, x, L/nx, t[-pde.nt_effective:], T/(nt-1)

def generate_trajectories(pde: PDE,
                          mode: str,
                          num_samples: int,
                          suffix: str,
                          batch_size: int,
                          num_workers: int=1,
                          seed: int=None,
                          device: torch.cuda.device="cpu") -> None:
    """
    Generate data trajectories for KdV, KS equation on periodic spatial domains.
    Args:
        pde (PDE): pde at hand [KS, KdV, Heat]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        suffix (str): naming suffix for special trajectories
        batch_size (int): batch size
        num_workers (int): number of processes solving trajectories in parallel
        seed (int): base seed for reproducible trajectories (None: unseeded)
        device: device (cpu/gpu)
    Returns:
        None
    """

    # batched solving is not implemented; trajectories are parallelized over processes via num_workers instead
    assert(batch_size == 1)
    num_batches = num_samples // batch_size

    pde_string = str(pde)
    print(f'Equation: {pde_string}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')
    print(f'Number of workers: {num_workers}')

    # Forked workers would inherit the same random state and produce duplicate trajectories,
    # so parallel runs always seed each trajectory; draw a base seed if none is given and report it.
    if seed is None and num_workers > 1:
        seed = int(np.random.SeedSequence().entropy % 2 ** 32)
        print(f'No seed given, using seed {seed}')

    sys.stdout.flush()

    save_name = "data/" + "_".join([pde_string, mode])
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    if suffix:
        save_name = save_name + '_' + suffix
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    tcoord = {}
    xcoord = {}
    dx = {}
    dt = {}
    h5f_u = {}

    nx = pde.grid_size[1]
    # The field u, the coordinations (xcoord, tcoord) and dx, dt are saved
    # Only nt_effective time steps of each trajectories are saved
    h5f_u = dataset.create_dataset(f'pde_{pde.nt_effective}-{nx}', (num_samples, pde.nt_effective, nx), dtype=float)
    xcoord = dataset.create_dataset(f'x', (num_samples, nx), dtype=float)
    dx = dataset.create_dataset(f'dx', (num_samples,), dtype=float)
    tcoord = dataset.create_dataset(f't', (num_samples, pde.nt_effective), dtype=float)
    dt = dataset.create_dataset(f'dt', (num_samples,), dtype=float)

    solve = partial(solve_trajectory, pde, mode, seed)
    pool = None
    if num_workers > 1:
        pool = Pool(num_workers, initializer=limit_threads)
        # imap keeps the order of trajectories, so idx matches the serial run
        results = pool.imap(solve, range(num_batches))
    else:
        results = map(solve, range(num_batches))

    for idx, (success, sol, x, dx_i, t, dt_i) in enumerate(results):

        # Saving the trajectories, if successfully solved
        if success:
            h5f_u[idx:idx+1, :, :] = sol
            xcoord[idx:idx+1, :] = x
            dx[idx:idx+1] = dx_i
            tcoord[idx:idx + 1, :] = t
            dt[idx:idx+1] = dt_i

        else:
            print("Solution was not successful.")

        print("Solved indices: {:d} : {:d}".format(idx * batch_size, (idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(idx + 1, num_batches))
        sys.stdout.flush()

    if pool is not None:
        pool.close()
        pool.join()

    print()
    print("Data saved")
    print()
    print()
    h5f.close()

def generate_data(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  num_workers: int=1,
                  seed: int=None,
                  device: torch.cuda.device="cpu") -> None:
    """
    Generate data for KdV, KS equation on periodic spatial domains.
    Args:
        experiment (str): pde at hand [KS, KdV, Heat]
        starting_time (float): starting time of PDE solving
        end_time (float): end time of PDE solving
        L (float): length of the spatial domain
        nx (int): spatial resolution
        nt (int): temporal resolution
        num_samples_train (int): number of trajectories created for training
        num_samples_valid (int): number of trajectories created for validation
        num_samples_test (int): number of trajectories created for testing
        batch_size (int): batch size
        num_workers (int): number of processes solving trajectories in parallel
        seed (int): base seed for reproducible trajectories (None: unseeded)
        device: device (cpu/gpu)
    Returns:
        None
    """
    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    if args.log:
        logfile = f'data/log/{experiment}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    # Create instances of PDE
    if experiment == 'KdV':
        pde = KdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'KS':
        pde = KS(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 L=L,
                 device=device)

    elif experiment == 'Burgers':
        # Heat equation is generated; afterwards trajectories are transformed via Cole-Hopf transformation.
        # L is not set for Burgers equation, since it is very sensitive. Default value is 2*math.pi.
        pde = Heat(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 device=device)

    else:
        raise Exception("Wrong experiment")

    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", True, num_samples_train, args.suffix),
             ("valid", num_samples_valid > 0, num_samples_valid, args.suffix),
             ("test", num_samples_test > 0, num_samples_test, args.suffix)}
    check_files(pde, files)

    # Obtain trajectories for different modes
    for mode, _, num_samples, suffix in files:
        if num_samples > 0:
            generate_trajectories(pde=pde,
                                  mode=mode,
                                  num_samples=num_samples,
                                  suffix=suffix,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  seed=seed,
                                  device=device)


def main(args: argparse) -> None:
    """
    Main method for data generation.
    """
    check_directory()

    if args.device != "cpu":
        raise NotImplementedError

    if args.batch_size != 1:
        raise NotImplementedError

    generate_data(experiment=args.experiment,
                  starting_time=0.0,
                  end_time=args.end_time,
                  L=args.L,
                  nt=args.nt,
                  nt_effective=args.nt_effective,
                  nx=args.nx,
                  num_samples_train=args.train_samples,
                  num_samples_valid=args.valid_samples,
                  num_samples_test=args.test_samples,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  seed=args.seed,
                  device=args.device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='KdV',
                        help='Experiment for which data should create for: [KdV, KS, Burgers]')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--end_time', type=float, default=100.,
                        help='How long do we want to simulate')
    parser.add_argument('--nt', type=int, default=250,
                        help='Time steps used for solving')
    parser.add_argument('--nt_effective', type=int, default=140,
                        help='Solved timesteps used for training')
    parser.add_argument('--nx', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--L', type=float, default=128.,
                        help='Length for which we want to solve the PDE')
    parser.add_argument('--train_samples', type=int, default=2 ** 5,
                        help='Samples in the training dataset')
    parser.add_argument('--valid_samples', type=int, default=2 ** 5,
                        help='Samples in the validation dataset')
    parser.add_argument('--test_samples', type=int, default=2 ** 5,
                        help='Samples in the test dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size used for creating training, val, and test dataset. So far the code only works for batch_size==1')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of processes used to solve trajectories in parallel')
    parser.add_argument('--seed', type=int, default=None,
                        help='Base seed; makes every trajectory reproducible, independent of num_workers')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for additional datasets')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')

    args = parser.parse_args()
    main(args)