import tensorflow as tf
import numpy as np

def conv_filter_shape(height, width, num_in, num_out):
    """
    Shape Of Filter In Convolution Layer
    Args:
    height : int
    hegiht = height of filter
    width : int
    width = width of filter
    num_in : int
    num_in = number of input channels
    num_out : int
    num_out = number of output channels
    Returns:
    shape : list
    shape = shape of a convolution filter
    """
    return [height, width, num_in, num_out]

def conv_bias_shape(num_out):
    """
    Shape Of Bias In Convolution Layer
    Args:
    num_out : int
    num_out = number of output channels
    Returns:
    shape : list
    shape = shape of a convolution bias
    """
    return [num_out]

def conv_info_from_weight(weight):
    """
    Get Convolution Layer Filter Shape From Weight
    Args:
    weight : numpy.ndarray or tf.Variable
    weight = weight of convolutional layer, shape [height, width, number of input channels, number of output channels]
    Returns:
    height : int
    hegiht = height of filter
    width : int
    width = width of filter
    num_in : int
    num_in = number of input channels
    num_out : int
    num_out = number of output channels
    """
    if (isinstance(weight, np.ndarray)):
        height, width, num_in, num_out = weight.shape
        assert (isinstance(height, int) and isinstance(width, int)
                and isinstance(num_in, int) and
                isinstance(num_out, int)), 'shape of weight should be int'
    elif (isinstance(weight, tf.Variable)):
        height, width, num_in, num_out = weight.shape
        height, width, num_in, num_out = [int(height), int(width),
                                                int(num_in), int(num_out)]
    else:
        raise NotImplementedError
    return height, width, num_in, num_out

def deconv_filter_shape(height, width, num_in, num_out):
    """
    Shape Of Filter In Deconvolution Layer
    Args:
    height : int
    height = height of filter
    width : int
    width = width of filter
    num_in : int
    num_in = number of input channels
    num_out : int
    num_out = number of output channels
    Returns:
    shape : list
    shape = shape of a convolution filter
    """
    return [height, width, num_out, num_in]

def deconv_bias_shape(num_out):
    """
    Shape Of Bias In Deconvolution Layer
    Args:
    num_out : int
    num_out = number of output channels
    Returns:
    shape : list
    shape = shape of a convolution bias
    """
    return [num_out]

def deconv_info_from_weight(weight):
    """
    Get Deconvolution Layer Filter Shape From Weight
    Args:
    weight : numpy.ndarray or tf.Variable
    weight = weight of deconvolutional layer, shape [height, width, number of output channels, number of input channels]
    Returns:
    height : int
    height = height of filter
    width : int
    width = width of filter
    num_in : int
    num_in = number of input channels
    num_out : int
    num_out = number of output channels
    """
    if (isinstance(weight, np.ndarray)):
        height, width, num_out, num_in = weight.shape
        assert (isinstance(height, int) and isinstance(width, int)
                and isinstance(num_in, int) and
                isinstance(num_out, int)), 'shape of weight should be int'
    elif (isinstance(weight, tf.Variable)):
        height, width, num_out, num_in = weight.shape
        height, width, num_in, num_out = [int(height), int(width),
                                                int(num_in), int(num_out)]
    else:
        raise NotImplementedError
    return height, width, num_in, num_out

def fc_weight_shape(num_in, num_out):
    """
    Shape of Weight In Fully Connected Layer
    Args:
    num_in : int
    num_in = length of input vector
    num_out : int
    num_out = length of output vector
    Returns:
    shape : list
    shape = shape of weight in fully connected
    """
    return [num_in, num_out]

def fc_info_from_weight(weight):
    """
    Get Fully Connected Layer Weight Shape From Weight
    Args:
    weight : numpy.ndarray or tf.Variable
    weight = weight of fully connected layer, shape [number of input channels, number of output channels]
    Returns:
    num_in : int
    num_in = length of input vector
    num_out : int
    num_out = length of output vector
    """
    if (isinstance(weight, np.ndarray)):
        num_in, num_out = weight.shape
        assert (isinstance(num_in, int) and
                isinstance(num_out, int)), 'shape of weight should be int'
    elif (isinstance(weight, tf.Variable)):
        num_in, num_out = weight.shape
        num_in, num_out = [int(num_in), int(num_out)]
    else:
        raise NotImplementedError
    return num_in, num_out

def fc_bias_shape(num_out):
    """
    Shape Of Bias In Fully Connected Layer
    Args:
    num_out : int
    num_out = number of output channels
    Returns:
    shape : list
    shape = shape of a fully connected bias
    """
    return [num_out]

def batch_to_seq(tensor_input, n_batch, n_step,
                time_major_input = False, time_major_output = True):
    """
    Convert A Tensor For RNN Input
    Args:
    tensor_input : tf.Variable or tensorflow.python.framework.ops.Tensor
    tensor_input = input tensor, shape [n_batch * n_step, ...]
    n_batch : int
    n_batch = size of a batch
    n_step : int
    n_step = number of steps
    time_major_input : bool
    time_major_input = if input is time-major, tensor_input is [step 1 of sample 1, step 1 of sample 2, ..., step 2 of sample 1, step 2 of sample 2, ...]. otherwise, tensor_input is [step 1 of sample 1, step 2 of sample 1, ..., step 1 of sample 2, step 2 of sample 2, ...]
    time_major_output : bool
    time_major_output = if output is time-major, sequence_output is [batch of step 1, batch of step 2, ...]. otherwise, sequence_output is [all steps of sample 1, all steps of sample 2, ...]
    Returns:
    sequence_output : list
    sequence_output = list of tensor, see time_major_output for details, each tensor shape is [n_batch, length of vector] or [n_step, length of vector]
    """
    tensor_shape = [int(d) for d in tensor_input.shape]
    if (time_major_input):
        tensor_input = tf.reshape(tensor_input, [n_step, n_batch] + tensor_shape[1:])
        if (time_major_output):
            sequence_output = [tf.squeeze(tensor_output, axis = 0)
                                for tensor_output in
                                    tf.split(tensor_input, n_step, axis = 0)]
        else:
            sequence_output = [tf.squeeze(tensor_output, axis = 1)
                                for tensor_output in
                                    tf.split(tensor_input, n_batch, axis = 1)]
    else:
        tensor_input = tf.reshape(tensor_input, [n_batch, n_step] + tensor_shape[1:])
        if (time_major_output):
            sequence_output = [tf.squeeze(tensor_output, axis = 1)
                                for tensor_output in
                                    tf.split(tensor_input, n_step, axis = 1)]
        else:
            sequence_output = [tf.squeeze(tensor_output, axis = 0)
                                for tensor_output in
                                    tf.split(tensor_input, n_batch, axis = 0)]
    return sequence_output

def seq_to_batch(tensor_list, time_major_input = True, time_major_output = False):   
    """
    Convert RNN List Output To Tensor
    Args:
    tensor_list : list
    tensor_list = list of input tensor, [tensor_0, tensor_1, ...]
    time_major_input : bool
    time_major_input = when this is True, each tensor_i is [n_batch, ...], otherwise, it is [n_step, ....]
    time_major_output : bool
    time_major_output = when this is True, the first dimension of output tensor is [step 1 of sample 1, step 1 of sample 2, ..., step 2 of sample 1, step 2 of sample 2, ...], otherwise it is [step 1 of sample 1, step 2 of sample 1, ..., step 1 of sample 2, step 2 of sample 2, ...]
    Returns:
    tensor_output : tf.Variable or tensorflow.python.framework.ops.Tensor
    tensor_output = output tensor, shape [n_batch * n_step, ...], see time_major_output for details
    """
    if (time_major_input):
        tensor_stack = tf.stack(tensor_list, axis = 0)
        if (not time_major_output):
            perm = range(len(tensor_stack.shape))
            perm[0], perm[1] = 1, 0
            tensor_stack = tf.transpose(tensor_stack, perm = perm)
    else:
        tensor_stack = tf.stack(tensor_list, axis = 0)
        if (time_major_output):
            perm = range(len(tensor_stack.shape))
            perm[0], perm[1] = 1, 0
            tensor_stack = tf.transpose(tensor_stack, perm = perm)
    tensor_shape = [-1] + [int(d) for d in tensor_stack.shape[2:]]
    tensor_output = tf.reshape(tensor_stack, tensor_shape)
    return tensor_output

def activation_function(layer_input, activation_param):
    """
    Activation Function
    Args:
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input of activation
    activation_param : str or tuple
    activation_param = parameter for activation
    Returns:
    output_op : tensorflow.python.framework.ops.Tensor
    output_op = output operator
    activation_type : string
    activation_type = type of activation
    """
    if (isinstance(activation_param, str)):
        if (activation_param.lower() == 'relu'):
            output_op = tf.nn.relu(layer_input)
            activation_type = 'relu'
        else:
            raise NotImplementedError
    elif (isinstance(activation_param, list) or
            isinstance(activation_param, tuple)):
        if (len(activation_param) == 2 and
                isinstance(activation_param[0], str)):
            if (activation_param[0].lower() == 'leakyrelu'):
                leak = float(activation_param[1])
                f1 = 0.5 * (1.0 + leak)
                f2 = 0.5 * (1.0 - leak)
                output_op = f1 * layer_input + f2 * tf.abs(layer_input)
                activation_type = 'leakyRelu'
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return output_op, activation_type

