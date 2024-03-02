from typing import List, Tuple

import torch
import torch.nn as nn

from models.scnet.utils import get_convtranspose_output_padding


class FusionLayer(nn.Module):
    """
    FusionLayer class implements a module for fusing two input tensors using convolutional operations.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - kernel_size (int, optional): Kernel size for the convolutional layer. Default is 3.
    - stride (int, optional): Stride value for the convolutional layer. Default is 1.
    - padding (int, optional): Padding value for the convolutional layer. Default is 1.

    Shapes:
    - Input: (B, F, T, C) and (B, F, T, C) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality.
    - Output: (B, F, T, C) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality.
    """

    def __init__(
        self, input_dim: int, kernel_size: int = 3, stride: int = 1, padding: int = 1
    ):
        """
        Initializes FusionLayer with input dimension, kernel size, stride, and padding.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim * 2,
            input_dim * 2,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.activation = nn.GLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the FusionLayer.

        Args:
        - x1 (torch.Tensor): First input tensor of shape (B, F, T, C).
        - x2 (torch.Tensor): Second input tensor of shape (B, F, T, C).

        Returns:
        - torch.Tensor: Output tensor of shape (B, F, T, C).
        """
        x = x1 + x2
        x = x.repeat(1, 1, 1, 2)
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.activation(x)
        return x


class Upsample(nn.Module):
    """
    Upsample class implements a module for upsampling input tensors using transposed 2D convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - stride (int): Stride value for the transposed convolution operation.
    - output_padding (int): Output padding value for the transposed convolution operation.

    Shapes:
    - Input: (B, C_in, F, T) where
        B is batch size,
        C_in is the number of input channels,
        F is the frequency dimension,
        T is the time dimension.
    - Output: (B, C_out, F * stride + output_padding, T) where
        B is batch size,
        C_out is the number of output channels,
        F * stride + output_padding is the upsampled frequency dimension.
    """

    def __init__(
        self, input_dim: int, output_dim: int, stride: int, output_padding: int
    ):
        """
        Initializes Upsample with input dimension, output dimension, stride, and output padding.
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            input_dim, output_dim, 1, (stride, 1), output_padding=(output_padding, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the Upsample module.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, C_in, F, T).

        Returns:
        - torch.Tensor: Output tensor of shape (B, C_out, F * stride + output_padding, T).
        """
        return self.conv(x)


class SULayer(nn.Module):
    """
    SULayer class implements a subband upsampling layer using transposed convolution.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - upsample_stride (int): Stride value for the upsampling operation.
    - subband_shape (int): Shape of the subband.
    - sd_interval (Tuple[int, int]): Start and end indices of the subband interval.

    Shapes:
    - Input: (B, F, T, C) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality.
    - Output: (B, F, T, C) where
        B is batch size,
        F is the number of features,
        T is sequence length,
        C is input dimensionality.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        upsample_stride: int,
        subband_shape: int,
        sd_interval: Tuple[int, int],
    ):
        """
        Initializes SULayer with input dimension, output dimension, upsample stride, subband shape, and subband interval.
        """
        super().__init__()
        sd_shape = sd_interval[1] - sd_interval[0]
        upsample_output_padding = get_convtranspose_output_padding(
            input_shape=sd_shape, output_shape=subband_shape, stride=upsample_stride
        )
        self.upsample = Upsample(
            input_dim=input_dim,
            output_dim=output_dim,
            stride=upsample_stride,
            output_padding=upsample_output_padding,
        )
        self.sd_interval = sd_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SULayer.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, F, T, C).

        Returns:
        - torch.Tensor: Output tensor of shape (B, F, T, C).
        """
        x = x[:, self.sd_interval[0] : self.sd_interval[1]]
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SUBlock(nn.Module):
    """
    SUBlock class implements a block with fusion layer and subband upsampling layers.

    Args:
    - input_dim (int): Dimensionality of the input channels.
    - output_dim (int): Dimensionality of the output channels.
    - upsample_strides (List[int]): List of stride values for the upsampling operations.
    - subband_shapes (List[int]): List of shapes for the subbands.
    - sd_intervals (List[Tuple[int, int]]): List of intervals for subband decomposition.

    Shapes:
    - Input: (B, Fi-1, T, Ci-1) and (B, Fi-1, T, Ci-1) where
        B is batch size,
        Fi-1 is the number of input subbands,
        T is sequence length,
        Ci-1 is the number of input channels.
    - Output: (B, Fi, T, Ci) where
        B is batch size,
        Fi is the number of output subbands,
        T is sequence length,
        Ci is the number of output channels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        upsample_strides: List[int],
        subband_shapes: List[int],
        sd_intervals: List[Tuple[int, int]],
    ):
        """
        Initializes SUBlock with input dimension, output dimension,
        upsample strides, subband shapes, and subband intervals.
        """
        super().__init__()
        self.fusion_layer = FusionLayer(input_dim=input_dim)
        self.su_layers = nn.ModuleList(
            SULayer(
                input_dim=input_dim,
                output_dim=output_dim,
                upsample_stride=uss,
                subband_shape=sbs,
                sd_interval=sdi,
            )
            for i, (uss, sbs, sdi) in enumerate(
                zip(upsample_strides, subband_shapes, sd_intervals)
            )
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the SUBlock.

        Args:
        - x (torch.Tensor): Input tensor of shape (B, Fi-1, T, Ci-1).
        - x_skip (torch.Tensor): Input skip connection tensor of shape (B, Fi-1, T, Ci-1).

        Returns:
        - torch.Tensor: Output tensor of shape (B, Fi, T, Ci).
        """
        x = self.fusion_layer(x, x_skip)
        x = torch.concat([layer(x) for layer in self.su_layers], dim=1)
        return x
