from typing import List, Tuple

import torch
import torch.nn as nn

from models.scnet.utils import create_intervals


class Downsample(nn.Module):
    """
    Downsample class implements a module for downsampling input tensors using 2D convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - stride (int): Stride value for the convolution operation.

    Shapes:
    - Input: (B, C_in, F, T) where
        B is batch size,
        C_in is the number of input channels,
        F is the frequency dimension,
        T is the time dimension.
    - Output: (B, C_out, F // stride, T) where
        B is batch size,
        C_out is the number of output channels,
        F // stride is the downsampled frequency dimension.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
    ):
        """
        Initializes Downsample with input dimension, output dimension, and stride.
        """
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 1, (stride, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the Downsample module.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C_in, F, T).

        Returns:
        - torch.Tensor: Downsampled tensor of shape (B, C_out, F // stride, T).
        """
        return self.conv(x)


class ConvolutionModule(nn.Module):
    """
    ConvolutionModule class implements a module with a sequence of convolutional layers similar to Conformer.

    Args:
    - input_dim (int): Dimensionality of the input features.
    - hidden_dim (int): Dimensionality of the hidden features.
    - kernel_sizes (List[int]): List of kernel sizes for the convolutional layers.
    - bias (bool, optional): If True, adds a learnable bias to the output. Default is False.

    Shapes:
    - Input: (B, T, D) where
        B is batch size,
        T is sequence length,
        D is input dimensionality.
    - Output: (B, T, D) where
        B is batch size,
        T is sequence length,
        D is input dimensionality.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: List[int],
        bias: bool = False,
    ) -> None:
        """
        Initializes ConvolutionModule with input dimension, hidden dimension, kernel sizes, and bias.
        """
        super().__init__()
        self.sequential = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=input_dim),
            nn.Conv1d(
                input_dim,
                2 * hidden_dim,
                kernel_sizes[0],
                stride=1,
                padding=(kernel_sizes[0] - 1) // 2,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_sizes[1],
                stride=1,
                padding=(kernel_sizes[1] - 1) // 2,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv1d(
                hidden_dim,
                input_dim,
                kernel_sizes[2],
                stride=1,
                padding=(kernel_sizes[2] - 1) // 2,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the ConvolutionModule.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, T, D).

        Returns:
        - torch.Tensor: Output tensor of shape (B, T, D).
        """
        x = x.transpose(1, 2)
        x = x + self.sequential(x)
        x = x.transpose(1, 2)
        return x


class SDLayer(nn.Module):
    """
    SDLayer class implements a subband decomposition layer with downsampling and convolutional modules.

    Args:
    - subband_interval (Tuple[float, float]): Tuple representing the frequency interval for subband decomposition.
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels after downsampling.
    - downsample_stride (int): Stride value for the downsampling operation.
    - n_conv_modules (int): Number of convolutional modules.
    - kernel_sizes (List[int]): List of kernel sizes for the convolutional layers.
    - bias (bool, optional): If True, adds a learnable bias to the convolutional layers. Default is True.

    Shapes:
    - Input: (B, Fi, T, Ci) where
        B is batch size,
        Fi is the number of input subbands,
        T is sequence length, and
        Ci is the number of input channels.
    - Output: (B, Fi+1, T, Ci+1) where
        B is batch size,
        Fi+1 is the number of output subbands,
        T is sequence length,
        Ci+1 is the number of output channels.
    """

    def __init__(
        self,
        subband_interval: Tuple[float, float],
        input_dim: int,
        output_dim: int,
        downsample_stride: int,
        n_conv_modules: int,
        kernel_sizes: List[int],
        bias: bool = True,
    ):
        """
        Initializes SDLayer with subband interval, input dimension,
        output dimension, downsample stride, number of convolutional modules, kernel sizes, and bias.
        """
        super().__init__()
        self.subband_interval = subband_interval
        self.downsample = Downsample(input_dim, output_dim, downsample_stride)
        self.activation = nn.GELU()
        conv_modules = [
            ConvolutionModule(
                input_dim=output_dim,
                hidden_dim=output_dim // 4,
                kernel_sizes=kernel_sizes,
                bias=bias,
            )
            for _ in range(n_conv_modules)
        ]
        self.conv_modules = nn.Sequential(*conv_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SDLayer.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, Fi, T, Ci).

        Returns:
        - torch.Tensor: Output tensor of shape (B, Fi+1, T, Ci+1).
        """
        B, F, T, C = x.shape
        x = x[:, int(self.subband_interval[0] * F) : int(self.subband_interval[1] * F)]
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)

        B, F, T, C = x.shape
        x = x.reshape((B * F), T, C)
        x = self.conv_modules(x)
        x = x.reshape(B, F, T, C)

        return x


class SDBlock(nn.Module):
    """
    SDBlock class implements a block with subband decomposition layers and global convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - bandsplit_ratios (List[float]): List of ratios for splitting the frequency bands.
    - downsample_strides (List[int]): List of stride values for downsampling in each subband layer.
    - n_conv_modules (List[int]): List specifying the number of convolutional modules in each subband layer.
    - kernel_sizes (List[int], optional): List of kernel sizes for the convolutional layers. Default is None.

    Shapes:
    - Input: (B, Fi, T, Ci) where
        B is batch size,
        Fi is the number of input subbands,
        T is sequence length,
        Ci is the number of input channels.
    - Output: (B, Fi+1, T, Ci+1) where
        B is batch size,
        Fi+1 is the number of output subbands,
        T is sequence length,
        Ci+1 is the number of output channels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bandsplit_ratios: List[float],
        downsample_strides: List[int],
        n_conv_modules: List[int],
        kernel_sizes: List[int] = None,
    ):
        """
        Initializes SDBlock with input dimension, output dimension, band split ratios, downsample strides, number of convolutional modules, and kernel sizes.
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 1]
        assert sum(bandsplit_ratios) == 1, "The split ratios must sum up to 1."
        subband_intervals = create_intervals(bandsplit_ratios)
        self.sd_layers = nn.ModuleList(
            SDLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                subband_interval=sbi,
                downsample_stride=dss,
                n_conv_modules=ncm,
                kernel_sizes=kernel_sizes,
            )
            for sbi, dss, ncm in zip(
                subband_intervals, downsample_strides, n_conv_modules
            )
        )
        self.global_conv2d = nn.Conv2d(output_dim, output_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass through the SDBlock.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, Fi, T, Ci).

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Output tensor and skip connection tensor.
        """
        x_skip = torch.concat([layer(x) for layer in self.sd_layers], dim=1)
        x = self.global_conv2d(x_skip.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x, x_skip
