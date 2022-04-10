"""
This file uses the Fourier Neural Operator implemented in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import sys
import numpy as np
import torch
from collections import OrderedDict
from typing import Tuple

from torch import nn
from torch.nn import functional as F
from equations.PDEs import PDE

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self,
                 pde: PDE,
                 time_history: int,
                 time_future: int,
                 modes: int = 32,
                 width: int = 256,
                 num_layers: int = 5
                 ):
        super(FNO1d, self).__init__()
        """
        Initialize the overall FNO network. It contains 5 layers of the Fourier layer.
        The input to the forward pass has the shape [batch, time_history, x].
        The output has the shape [batch, time_future, x].
        Args:
            pde (PDE): PDE at hand
            time_history (int): input timesteps of the trajectory
            time_future (int): output timesteps of the trajectory
            modes (int): low frequency Fourier modes considered for multiplication in the Fourier space
            width (int): hidden channel dimension
            num_layers (int): number of FNO layers
        """
        self.modes = modes
        self.width = width
        self.pde = pde
        self.time_history = time_history
        self.time_future = time_future
        self.fc0 = nn.Linear(self.time_history + 2, self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

        fourier_layers = []
        conv_layers = []
        for i in range(num_layers):
            fourier_layers.append(SpectralConv1d(self.width, self.width, self.modes))
            conv_layers.append(nn.Conv1d(self.width, self.width, 1))
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.conv_layers = nn.ModuleList(conv_layers)

    def __repr__(self):
        return f'FNO1d'

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FNO network.
        The input to the forward pass has the shape [batch, time_history, x].
        1. Add dx and dt as channel dimension to the time_history, repeat for every x
        2. Lift the input to the desired channel dimension by self.fc0
        3. 5 (default) FNO layers
        4. Project from the channel space to the output space by self.fc1 and self.fc2.
        The output has the shape [batch, time_future, x].
        Args:
            u (torch.Tensor): input tensor of shape [batch, time_history, x]
            dx (torch.Tensor): spatial distances
            dt (torch.Tensor): temporal distances
        Returns: torch.Tensor: output has the shape [batch, time_future, x]
        """
        # TODO: rewrite training method and forward pass without permutation
        # [b, x, c] = [b, x, t+2]
        x = torch.cat((u, dx[:, None, None].to(u.device).repeat(1, self.pde.nx, 1),
                       dt[:, None, None].repeat(1, self.pde.nx, 1).to(u.device)), -1)

        x = self.fc0(x)
        # [b, x, c] -> [b, c, x]
        x = x.permute(0, 2, 1)

        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x1 = fourier(x)
            x2 = conv(x)
            x = x1 + x2
            x = F.gelu(x)

        # [b, c, x] -> [b, x, c]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x
