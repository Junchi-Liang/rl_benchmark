import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def convert_to_tuple(x, t = None):
    """
    Check Or Convert Parameter To Tuple
    (convert padding, kernel size, stride to tuple for 2d)
    Args:
    x : int or float or tuple or list
    x = input parameter
    t : type (int, float, ...)
    t = expect type for x
    Returns:
    tuple_output : tuple
    tuple_output = output tuple
    """
    if (isinstance(x, list) or isinstance(x, tuple)):
        return tuple(x)
    if (t is None):
        return (x, x)
    assert isinstance(x, t), "unexpected parameter type"
    return (x, x)

def get_activation(activation_param):
    """
    Get Activation Module
    Args:
    activation_param : string or tuple
    activation_param = parameters for activation
    Returns:
    activation_type : string
    activation_type = type of the activation
    activation_module : torch.nn.modules
    activation_module = activation module
    """
    if (isinstance(activation_param, str)):
        if (activation_param.lower() == 'relu'):
            return 'relu', nn.ReLU()
        elif (activation_param.lower() == 'tanh'):
            return 'tanh', nn.Tanh()
        elif (activation_param.lower() == 'sigmoid'):
            return 'sigmoid', nn.Sigmoid()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def calculate_gain_from_activation(activation_param):
    """
    Return The Recommended Gain Value For The Given Activation
    Args:
    activation_param : string or tuple
    activation_param = parameters for activation
    Returns:
    gain : float
    gain = recommended gain value
    """
    if (isinstance(activation_param, str)):
        if (activation_param.lower() == 'relu'):
            return nn.init.calculate_gain('relu')
        elif (activation_param.lower() == 'tanh'):
            return nn.init.calculate_gain('tanh')
        elif (activation_param.lower() == 'sigmoid'):
            return nn.init.calculate_gain('sigmoid')
        else:
            raise NotImplementedError
    elif (activation_param is None):
        return nn.init.calculate_gain('linear')
    else:
        raise NotImplementedError

class Flatten(nn.Module):
    """
    Flatten Data To Vector
    (convert [batch size, ....] to [batch size, size of features])
    """
    def forward(self, x):
        return x.view(x.size(0), -1)
