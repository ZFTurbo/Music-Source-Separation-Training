'''
SCNet - great paper, great implementation
https://arxiv.org/pdf/2401.13276.pdf
https://github.com/amanteur/SCNet-PyTorch
'''

from typing import List, Tuple, Union

import torch


def create_intervals(
    splits: List[Union[float, int]]
) -> List[Union[Tuple[float, float], Tuple[int, int]]]:
    """
    Create intervals based on splits provided.

    Args:
    - splits (List[Union[float, int]]): List of floats or integers representing splits.

    Returns:
    - List[Union[Tuple[float, float], Tuple[int, int]]]: List of tuples representing intervals.
    """
    start = 0
    return [(start, start := start + split) for split in splits]


def get_conv_output_shape(
    input_shape: int,
    kernel_size: int = 1,
    padding: int = 0,
    dilation: int = 1,
    stride: int = 1,
) -> int:
    """
    Compute the output shape of a convolutional layer.

    Args:
    - input_shape (int): Input shape.
    - kernel_size (int, optional): Kernel size of the convolution. Default is 1.
    - padding (int, optional): Padding size. Default is 0.
    - dilation (int, optional): Dilation factor. Default is 1.
    - stride (int, optional): Stride value. Default is 1.

    Returns:
    - int: Output shape.
    """
    return int(
        (input_shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )


def get_convtranspose_output_padding(
    input_shape: int,
    output_shape: int,
    kernel_size: int = 1,
    padding: int = 0,
    dilation: int = 1,
    stride: int = 1,
) -> int:
    """
    Compute the output padding for a convolution transpose operation.

    Args:
    - input_shape (int): Input shape.
    - output_shape (int): Desired output shape.
    - kernel_size (int, optional): Kernel size of the convolution. Default is 1.
    - padding (int, optional): Padding size. Default is 0.
    - dilation (int, optional): Dilation factor. Default is 1.
    - stride (int, optional): Stride value. Default is 1.

    Returns:
    - int: Output padding.
    """
    return (
        output_shape
        - (input_shape - 1) * stride
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    )


def compute_sd_layer_shapes(
    input_shape: int,
    bandsplit_ratios: List[float],
    downsample_strides: List[int],
    n_layers: int,
) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Compute the shapes for the subband layers.

    Args:
    - input_shape (int): Input shape.
    - bandsplit_ratios (List[float]): Ratios for splitting the frequency bands.
    - downsample_strides (List[int]): Strides for downsampling in each layer.
    - n_layers (int): Number of layers.

    Returns:
    - Tuple[List[List[int]], List[List[Tuple[int, int]]]]: Tuple containing subband shapes and convolution shapes.
    """
    bandsplit_shapes_list = []
    conv2d_shapes_list = []
    for _ in range(n_layers):
        bandsplit_intervals = create_intervals(bandsplit_ratios)
        bandsplit_shapes = [
            int(right * input_shape) - int(left * input_shape)
            for left, right in bandsplit_intervals
        ]
        conv2d_shapes = [
            get_conv_output_shape(bs, stride=ds)
            for bs, ds in zip(bandsplit_shapes, downsample_strides)
        ]
        input_shape = sum(conv2d_shapes)
        bandsplit_shapes_list.append(bandsplit_shapes)
        conv2d_shapes_list.append(create_intervals(conv2d_shapes))

    return bandsplit_shapes_list, conv2d_shapes_list


def compute_gcr(subband_shapes: List[List[int]]) -> float:
    """
    Compute the global compression ratio.

    Args:
    - subband_shapes (List[List[int]]): List of subband shapes.

    Returns:
    - float: Global compression ratio.
    """
    t = torch.Tensor(subband_shapes)
    gcr = torch.stack(
        [(1 - t[i + 1] / t[i]).mean() for i in range(0, len(t) - 1)]
    ).mean()
    return float(gcr)