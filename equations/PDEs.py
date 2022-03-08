import os
import sys
import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from common.augmentation import Subalgebra
from scipy.fftpack import diff as psdiff



class PDE(nn.Module):
    """
    Generic PDE template
    """
    def __init__(self):
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def pseudospectral_reconstruction(self):
        """
        A pseudospectral method template
        """
        pass

    def fvm_reconstruction(self):
        """
        A finite volumes method template
        """
        pass


class KdV(PDE):
    """
    The Korteweg-de Vries equation:
    ut + (0.5*u**2 + uxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        if self.device != "cpu":
            raise NotImplementedError

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.max_velocity = 0.0
        self.max_scale = 0.0

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KdV'

    def pseudospectral_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        Pseudospectral reconstruction of the spatial derivatives of the KdV equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed pseudospectral time derivative
        """
        # Compute the x derivatives using the pseudo-spectral method.
        ux = psdiff(u, period=L)
        uxxx = psdiff(u, order=3, period=L)
        # Compute du/dt.
        dudt = - u*ux - uxxx
        return dudt

    def fvm_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        FVM reconstruction for the spatial derivatives of the KdV equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed FVM time derivative
        """
        dx = L / len(u)
        # Integrate: exact at half nodes
        iu = np.cumsum(u) * dx
        # Derivatives
        u = psdiff(iu, order=0 + 1, period=L)
        uxx = psdiff(iu, order=2 + 1, period=L)
        # Compute du/dt.
        Jrhs = 0.5 * (u ** 2) + uxx
        Jlhs = np.roll(Jrhs, 1)
        dudt = -(Jrhs - Jlhs) / dx
        return dudt


class KS(PDE):
    """
    The Kuramoto-Sivashinsky equation:
    ut + (0.5*u**2 + ux + uxxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 40. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 64. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        if self.device != "cpu":
            raise NotImplementedError

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.max_velocity = 0.0

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KS'

    def pseudospectral_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        Pseudospectral reconstruction of the spatial derivatives of the KS equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed pseudospectral time derivative
        """
        # Compute the x derivatives using the pseudo-spectral method.
        ux = psdiff(u, period=L)
        uxx = psdiff(u, period=L, order=2)
        uxxxx = psdiff(u, period=L, order=4)
        # Compute du/dt.
        dudt = - u*ux - uxx - uxxxx
        return dudt

    def fvm_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        FVM reconstruction of the spatial derivatives of the KS equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: reconstructed FVM time derivative
        """
        dx = L / len(u)
        # Integrate: exact at half nodes
        iu = np.cumsum(u) * dx
        # Derivatives
        u = psdiff(iu, order=0 + 1, period=L)
        ux = psdiff(iu, order=1 +1 , period=L)
        uxxx = psdiff(iu, order=3 + 1, period=L)
        # Compute du/dt.
        Jrhs = 0.5 * (u ** 2) + ux + uxxx
        Jlhs = np.roll(Jrhs, 1)
        dudt = -(Jrhs - Jlhs) / dx
        return dudt


class Heat(PDE):
    """
    The heat equation ut - nu * uxx = 0
    which we use to get data for the Burgers' equation via the Cole-Hopf transformation
    """
    def __init__(self,
                 nu: float=None,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Diffusion coefficient
        self.nu = 0.01 if nu is None else nu
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 16. if tmax is None else tmax
        # Sin frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 7 if lmax is None else lmax
        # Number of different waves
        self.N = 20
        # Length of the spatial domain
        self.L = 2 * math.pi if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        if self.device != "cpu":
            raise NotImplementedError

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.alpha = 0.0

        # Subalgebra class for mixing different solutions
        self.subalgebra = Subalgebra(self.nu, self.alpha)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'Heat'

    def pseudospectral_reconstruction(self, t: float, u: np.ndarray, L: float) -> np.ndarray:
        """
        Pseudospectral reconstruction of the spatial derivatives of the Heat equation, discretized in x.
        Args:
            t (float): time point
            u (np.ndarray): 1D input field
            L (float): length of the spatial domain
        Returns:
            np.ndarray: Pseudospectral reconstructed time derivative
        """
        # Compute the x derivatives using the pseudo-spectral method.
        uxx = psdiff(u, period=L, order=2)
        # Compute du/dt.
        dudt = self.nu * uxx
        return dudt
