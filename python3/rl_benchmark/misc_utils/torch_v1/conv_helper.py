import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_output_size(size_input, padding, kernel_size,
                                stride, dilation = 1):
    """
    Size Of Convolutional Output
    (compute for one side)
    Args:
    size_input : int
    size_input = input size
    padding : int or list or tuple
    padding = padding size
    kernel_size : int
    kernel_size = size of the kernel
    stride : int
    stride = stride of the convolution
    dilation : int
    dilation = spacing between kernel elements
    Returns:
    size_output : int
    size_output = output size
    """
    if (isinstance(padding, int)):
        size_output = np.floor(((size_input + 2 * padding -
                        dilation * (kernel_size - 1) - 1) / stride) + 1)
    elif (isinstance(padding, tuple) or isinstance(padding, list)):
        size_output = np.floor(((size_input + np.sum(padding) -
                        dilation * (kernel_size - 1) - 1) / stride) + 1)
    else:
        raise NotImplementedError
    return int(size_output)

def deconv_output_size(size_input, padding, kernel_size, stride,
        output_padding = 0, dilation = 1):
    """
    Size Of Deconvolutional Output
    (compute for one side)
    Args:
    size_input : int
    size_input = input size
    padding : int or list or tuple
    padding = padding size
    kernel_size : int
    kernel_size = size of the kernel
    stride : int
    stride = stride of the convolution
    output_padding : int
    output_padding = Additional size added to one side of each dimension in the output shape
    dilation : int
    dilation = spacing between kernel elements
    Returns:
    size_output : int
    size_output = output size
    """
    size_output = (size_input - 1) * stride + dilation * (
            kernel_size - 1) + 1
    if (isinstance(padding, int)):
        size_output -= 2 * padding
    elif (isinstance(padding, list) or isinstance(padding, tuple)):
        size_output -= np.sum(padding)
    else:
        raise NotImplementedError
    if (isinstance(output_padding, int)):
        size_output += output_padding
    elif (isinstance(output_padding, list) or
            isinstance(output_padding, tuple)):
        size_output += np.sum(output_padding)
    else:
        raise NotImplementedError
    return size_output

