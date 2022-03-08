import numpy as np
import torch
import random
from typing import Optional, Tuple


def fourier_shift(u: torch.Tensor, eps: float=0., dim: int=-1, order: int=0) -> torch.Tensor:
    """
    Shift in Fourier space.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
        order (int): derivative order
    Returns:
        torch.Tensor: Fourier shifted input
    """
    assert dim < 0
    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm='ortho')
    # Fourier modes
    omega = torch.arange(n // 2 + 1)
    if n % 2 == 0:
        omega[-1] *= 0
    # Applying Fourier shift according to shift theorem
    fs = torch.exp(- 2 * np.pi * 1j * omega * eps)
    # For order>0 derivative is taken
    fs = (- 2 * np.pi * 1j * omega) ** order * fs
    for _ in range(-dim - 1):
        fs = fs[..., None]
    return torch.fft.irfft(fs * u_hat, n=n, dim=dim, norm='ortho')

def linear_shift(u: torch.Tensor, eps: float=0., dim:int=-1) -> torch.Tensor:
    """
    Linear shift.
    Args:
        u (torch.Tensor): input tensor, usually of shape [batch, t, x]
        eps (float): shift parameter
        dim (int): dimension which is used for shifting
    Returns:
        Linear shifted input
    """
    n = u.shape[dim]
    # Shift to the left and to the right and interpolate linearly
    q, r = torch.div(eps*n, 1, rounding_mode='floor'), (eps * n) % 1
    q_left, q_right = q/n, (q+1)/n
    u_left = fourier_shift(u, eps=q_left, dim=-1)
    u_right = fourier_shift(u, eps=q_right, dim=-1)
    return (1-r) * u_left + r * u_right

def to_coords(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Transforms the coordinates to a tensor X of shape [time, space, 2].
    Args:
        x: spatial coordinates
        t: temporal coordinates
    Returns:
        torch.Tensor: X[..., 0] is the space coordinate (in 2D)
                      X[..., 1] is the time coordinate (in 2D)
    """
    x_, t_ = torch.meshgrid(x, t)
    x_, t_ = x_.T, t_.T
    return torch.stack((x_, t_), -1)


class SpaceTranslate:
    def __init__(self, max_x_shift: float=1.):
        """
        Instantiate sub-pixel space translation.
        Translations are drawn from the distribution.
        Uniform(-max_shift/2, max_shift/2) where max_shift is in units of input side length.
        Args:
            max_shift (float): maximum shift length (rotations)
        """
        self.max_x_shift = max_x_shift

    def __call__(self, sample: torch.Tensor, eps: Optional[float]=None, shift: str='fourier') -> torch.Tensor:
        """
        Sub-pixel space translation shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift
        Returns:
            torch.Tensor: sub-pixel shifted tensor of the form [u, X]
        """
        u, X = sample

        if eps is None:
            eps = self.max_x_shift * (torch.rand(()) - 0.5)
        else:
            eps = eps * torch.ones(())

        if shift == 'fourier':
            output = (fourier_shift(u, eps=eps, dim=-1), X)
        elif shift == 'linear':
            output = (linear_shift(u, eps=eps, dim=-1), X)

        return output


class Scale:
    def __init__(self, max_scale: float=1.):
        """
        Instantiate scale generator.
        Scale transformations are drawn from the distribution
        Uniform(-max_scale, max_scale)
        Args:
            max_scale (float): maximum scale shift
        """
        self.max_scale = max_scale

    def __call__(self, sample: torch.Tensor, eps: Optional[float]=None, shift: str='fourier') -> torch.Tensor:
        """
        Scale shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: scale shifted tensor of the form [u, X]
        """
        u, X = sample
        X = X.clone()

        if eps is None:
            eps = self.max_scale * (torch.rand(()) - 0.5)
        else:
            eps = eps * torch.ones(())

        X[..., 0] *= torch.exp(-eps)
        X[..., 1] *= torch.exp(-3 * eps)

        return (torch.exp(2 * eps) * u, X)


class Galileo:
    def __init__(self, max_velocity: float=1) -> torch.Tensor:
        """
        Instantiate Galileo generator.
        Galilean transformations are drawn from the distribution
            Uniform(-max_velocity, max_velocity) where max_velocity is in units of m/s.
        Args:
            max_velocity: float for maximum velocity in m/s.
        """
        self.max_velocity = max_velocity

    def __call__(self, sample: torch.Tensor, eps: Optional[float]=None, shift: str='fourier') -> torch.Tensor:
        """
        Galilean shift.
        Args:
            sample (torch.Tensor): input tensor of the form [u, X]
            eps (float): shift parameter
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: Galilean shifted tensor of the form [u, X]
        """
        u, X = sample

        T = u.shape[-2]
        N = u.shape[-1]
        dx = X[0, 1, 0] - X[0, 0, 0]
        dt = X[1, 0, 1] - X[0, 0, 1]
        t = dt * torch.arange(T)
        L = dx * N

        if eps is None:
            eps = 2 * self.max_velocity * (torch.rand(()) - 0.5)
        else:
            eps = eps * torch.ones(())
        # shift in pixel
        d = -(eps * t) / L

        if shift == 'fourier':
            output = (fourier_shift(u, eps=d[:, None], dim=-1) - eps, X)
        elif shift == 'linear':
            output = (linear_shift(u, eps=d[:, None], dim=-1) - eps, X)

        return output


def heat_to_burgers(psi: torch.Tensor, nu: float, L: float) -> torch.Tensor:
    """
    Cole-Hopf transformation which transforms a trajectory of the Heat equation into the Burgers' equation.
    Args:
        psi (torch.Tensor): input trajectory of the Heat equation
        nu (float): diffusion coefficient
        L (float): length of spatial domain
    Returns:
        torch.Tensor: Cole-Hopf transformed trajectory of the Burgers' equation
    """
    psi_max = torch.amax(psi, dim=-1, keepdim=True)
    psi_min = torch.amin(psi, dim=-1, keepdim=True)
    psi = psi / (psi_max - psi_min)
    psix = fourier_shift(psi, order=1)
    return -(psix / psi) / (2 * nu)


class Subalgebra:
    def __init__(self, nu: float, alpha: float=0.5):
        """
        Instantiate Burgers A_alpha generator.
        CAVEAT: samples have to be solutions to the Heat equation!!!!
        Args:
            nu: dynamic viscocity coefficient
            alpha: convex interpolation coefficient [0, 1]
        """
        self.nu = nu
        self.alpha = alpha

    def __call__(self, heat_sample1: torch.Tensor, heat_sample2: torch.Tensor=None, alpha: float=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infinite subalgebra.
        Combines two trajectories of the Heat equation and returns a Cole-Hopf transformed
        trajectory of the Burgers' equation.
        Args:
            heat_sample1 (torch.Tensor): input tensor of the form [u, X] of the first trajectory of the Heat equation
            heat_sample2 (torch.Tensor): input tensor of the form [u, X] of the second trajectory of the Heat equation
            alpha (float): mixing parameter
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: Cole-Hopf transformed trajectory of the Burgers' equation of the form [u, X]
        """
        psi1, X = heat_sample1
        dx = X[1, 0, 1] - X[0, 0, 1]
        L = dx * X.shape[0]

        if heat_sample2 is not None:
            if alpha is None:
                alpha = self.max_alpha
            else:
                alpha = alpha
            psi2, _ = heat_sample2
            u = heat_to_burgers((1 - alpha) * psi1 + alpha * psi2, self.nu, L)
        else:
            u = heat_to_burgers(psi1, self.nu, L)

        return u, X


class KdV_augmentation:
    def __init__(self, max_x_shift: float, max_velocity: float, max_scale):
        """
        Instantiate KdV data augmentation.
        Args:
            max_x_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
            max_scale (float): parameter of scaling transformation
        """
        self.generators = [SpaceTranslate(max_x_shift),
                           Galileo(max_velocity),
                           Scale(max_scale)]

    def __call__(self, u: torch.Tensor, shift: str='fourier') -> torch.Tensor:
        """
        KdV data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: new space shifted, Galilean transformed and scaled trajectory
        """
        for g in self.generators:
            u = g(u, shift=shift)
        return u


class KS_augmentation:
    def __init__(self, max_x_shift: float, max_velocity: float):
        """
        Instantiate KS data augmentation.
        Args:
            max_x_shift (float): parameter of sub-pixel space translation
            max_velocity (float): parameter of Galilean transformation
        """
        self.generators = [SpaceTranslate(max_x_shift),
                           Galileo(max_velocity)]

    def __call__(self, u: torch.Tensor, shift: str='fourier') -> torch.Tensor:
        """
        KS data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
            shift (str): fourier or linear shift (not used, only for consistency w.r.t. other generators)
        Returns:
            torch.Tensor: new space shifted and Galilean transformed trajectory
        """
        for g in self.generators:
            u = g(u, shift=shift)
        return u

class Heat_augmentation:
    def __init__(self, max_x_shift: float):
        """
        Instantiate Heat data augmentation (other generators might still be added).
        Args:
            max_x_shift (float): parameter of sub-pixel space translation
        """
        self.generators = [SpaceTranslate(max_x_shift)]
        # TODO add other generators

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """
        Heat data augmentation, evoking one generator after each other.
        Args:
            u (torch.Tensor): input tensor of the form [u, X]
        Returns:
            torch.Tensor: new space shifted trajectory
        """
        for g in self.generators:
            u = g(u)
        return u
