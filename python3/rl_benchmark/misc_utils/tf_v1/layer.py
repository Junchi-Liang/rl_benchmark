import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from rl_benchmark.misc_utils.tf_v1.helper import conv_filter_shape, conv_bias_shape, conv_info_from_weight
from rl_benchmark.misc_utils.tf_v1.helper import deconv_filter_shape, deconv_bias_shape, deconv_info_from_weight
from rl_benchmark.misc_utils.tf_v1.helper import fc_weight_shape, fc_bias_shape, fc_info_from_weight

def conv2d_layer(name, layer_input, stride, padding,
                kernel_size = None,
                input_channel = None, output_channel = None,
                weight = None, bias = None,
                add_bias = True,
                weight_initializer = None, bias_initializer = None):
    """
    2D Convolution Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer
    stride : list, tuple, or int
    stride = stride in convoltuon layer, when stride is list or tuple, it is [height, width] or [batch, height, width, channel]
    padding : string
    padding = padding style, 'SAME' or 'VALID'
    kernel_size : list, tuple or int
    kernel_size = size of convolution filter, when kernel_size is list or tuple, it is [height, width]
    input_channel : int
    input_channel = number of input channels, when this is None, weight should be provided
    output_channel : int
    output_channel = number of output channels, when this is None, weight should be provided
    weight : tf.Variable
    weight = weight of convolutional layer
    bias : tf.Variable
    bias = bias of convolutional layer
    add_bias : bool
    add_bias = when this is True, bias is added
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer_conv2d() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    Returns:
    conv_op : tensorflow.python.framework.ops.Tensor
    conv_op = output convolutional layer
    conv_param : dictionary
    conv_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    with tf.variable_scope(name) as conv_sc:
        # get Weight
        if (weight is None):
            if (isinstance(kernel_size, int)):
                kernel_size = [kernel_size, kernel_size]
            if ((not isinstance(kernel_size, list))
                and (not isinstance(kernel_size, tuple))):
                raise NotImplementedError
            weight_shape = conv_filter_shape(kernel_size[0], kernel_size[1],
                                            input_channel, output_channel)
            if (weight_initializer is None):
                weight_initializer =\
                        tf.contrib.layers.xavier_initializer_conv2d()
            weight = tf.get_variable('weight', weight_shape,
                                        initializer = weight_initializer)
        # stride
        if (isinstance(stride, int)):
            stride = [1, stride, stride, 1]
        elif (isinstance(stride, list) or isinstance(stride, tuple)):
            if (len(stride) == 2):
                stride = [1, stride[0], stride[1], 1]
            elif (len(stride) != 4):
                raise NotImplementedError
        else:
            raise NotImplementedError
        # operator and parameter
        conv_op = tf.nn.conv2d(layer_input, weight,
                                strides = stride, padding = padding)
        conv_param = {'weight': weight}
        # add bias if necessary
        if (add_bias):
            if (bias is None):
                bias_shape = conv_bias_shape(output_channel)
                if (bias_initializer is None):
                    bias_initializer = tf.constant_initializer(
                                                np.zeros(bias_shape))
                bias = tf.get_variable('bias', bias_shape,
                                        initializer = bias_initializer)
            conv_op = tf.nn.bias_add(conv_op, bias)
            conv_param['bias'] = bias
    return conv_op, conv_param

def deconv2d_layer(name, layer_input, stride, padding,
                kernel_size = None,
                input_channel = None, output_channel = None,
                weight = None, bias = None, add_bias = True,
                weight_initializer = None, bias_initializer = None,
                output_shape = None, matched_tensor = None):
    """
    2D Deconvolution Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer
    stride : list, tuple, or int
    stride = stride in deconvolution layer, when stride is list or tuple, it is [height, width] or [batch, height, width, channel]
    padding : string
    padding = padding style, 'SAME' or 'VALID'
    kernel_size : list, tuple or int
    kernel_size = size of deconvolution filter, when kernel_size is list or tuple, it is [height, width]
    input_channel : int
    input_channel = number of input channels, when this is None, weight should be provided
    output_channel : int
    output_channel = number of output channels, when this is None, weight or output_shape should be provided
    weight : tf.Variable
    weight = weight of convolutional layer
    bias : tf.Variable
    bias = bias of convolutional layer
    add_bias : bool
    add_bias = when this is True, bias is added
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer_conv2d() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    output_shape : list or tuple
    output_shape = output shape of this layer, when this is None, the shape will be infered from deconv_output_length
    matched_tensor : tensorflow.python.framework.ops.Tensor
    matched_tensor = when output_shape is None but this is not None, output shape will be matched to this tensor
    Returns:
    deconv_op : tensorflow.python.framework.ops.Tensor
    deconv_op = output deconvolutional layer
    deconv_param : dictionary
    deconv_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    with tf.variable_scope(name) as deconv_sc:
        # get weight
        if (weight is None):
            if (isinstance(kernel_size, int)):
                kernel_size = [kernel_size, kernel_size]
            if (not((isinstance(kernel_size, list)) and (kernel_size, tuple))):
                raise NotImplementedError
            if (output_channel is None):
                assert ((output_shape is not None)
                        and (isinstance(output_shape[3], int))
                        ), 'number of output channels is unknown'
                output_channel = output_shape[3]
            weight_shape = deconv_filter_shape(
                                kernel_size[0], kernel_size[1],
                                input_channel, output_channel)
            if (weight_initializer is None):
                weight_initializer =\
                        tf.contrib.layers.xavier_initializer_conv2d()
            weight = tf.get_variable('weight', weight_shape,
                                        initializer = weight_initializer)
        else:
            _, _, input_channel, output_channel =\
                                deconv_info_from_weight(weight)
        # stride
        if (isinstance(stride, int)):
            stride = [1, stride, stride, 1]
        elif (isinstance(stride, list) or isinstance(stride, tuple)):
            if (len(stride) == 2):
                stride = [1, stride[0], stride[1], 1]
            elif (len(stride) != 4):
                raise NotImplementedError
        else:
            raise NotImplementedError
        # output shape
        if (output_shape is None):
            if (matched_tensor is None):
                raise NotImplementedError
            matched_shape = array_ops.shape(matched_tensor)
            output_shape = [matched_shape[0], matched_shape[1],
                                matched_shape[2], output_channel]
        # operator and parameter
        deconv_op = tf.nn.conv2d_transpose(layer_input, weight,
                                                output_shape,
                                                strides = stride,
                                                padding = padding)
        deconv_param = {'weight': weight}
        # add bias if necessary
        if (add_bias):
            if (bias is None):
                bias_shape = deconv_bias_shape(output_channel)
                if (bias_initializer is None):
                    bias_initializer = tf.constant_initializer(
                                                np.zeros(bias_shape))
                bias = tf.get_variable('bias', bias_shape,
                                        initializer = bias_initializer)
            deconv_op = tf.nn.bias_add(deconv_op, bias)
            deconv_param['bias'] = bias
    return deconv_op, deconv_param

def fc_layer(name, layer_input, input_length = None, output_length = None,
                weight = None, bias = None, add_bias = True,
                weight_initializer = None, bias_initializer = None):
    """
    Fully Connected Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer
    input_length : int
    input_length = length of input vector
    output_length : int
    output_length = length of output vector
    weight : tf.Variable
    weight = weight of fully connected layer
    bias : tf.Variable
    bias = bias of fully connected layer
    add_bias : bool
    add_bias = when this is True, bias is added
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    Returns:
    fc_op : tensorflow.python.framework.ops.Tensor
    fc_op = output fully connected layer
    fc_param : dictionary
    fc_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    with tf.variable_scope(name) as fc_sc:
        # get weight
        if (weight is None):
            weight_shape = fc_weight_shape(input_length, output_length)
            if (weight_initializer is None):
                weight_initializer = tf.contrib.layers.xavier_initializer()
            weight = tf.get_variable('weight', weight_shape,
                                        initializer = weight_initializer)
        else:
            input_length, output_length = fc_info_from_weight(weight)
        # operator and parameter
        fc_op = tf.matmul(layer_input, weight)
        fc_param = {'weight': weight}
        # add bias if necessary
        if (add_bias):
            if (bias is None):
                bias_shape = fc_bias_shape(output_length)
                if (bias_initializer is None):
                    bias_initializer = tf.constant_initializer(
                                                np.zeros(bias_shape))
                bias = tf.get_variable('bias', bias_shape,
                                        initializer = bias_initializer)
            fc_op = tf.nn.bias_add(fc_op, bias)
            fc_param['bias'] = bias
    return fc_op, fc_param

def dynamic_lstm_layer(name, num_units, lstm_input, initial_state = None,
                weight = None, bias = None):
    """
    LSTM Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    num_units: int
    num_units = the number of units in the LSTM cell
    lstm_input : tensorflow.python.framework.ops.Tensor
    lstm_input = input layer, shape [batch_size, max_time, ...]
    initial_state : tuple or None
    initial_state = (c, h), their shape should be [batch_size, cell.state_size.c],  [batch_size, cell.state_size.h]
    weight : tf.Variable
    weight = weight of fully connected layer
    bias : tf.Variable
    bias = bias of fully connected layer
    Returns:
    lstm_op : dictionary
    lstm_op = {'output': LSTM output, 'final_state': final state, 'initial_state': state input}
    lstm_param : dictionary
    lstm_param = {'weight': weight, 'bias': bias}
    """
    with tf.variable_scope(name) as lstm_sc:
        cell = tf.nn.rnn_cell.LSTMCell(num_units = num_units,
                            use_peepholes = False, state_is_tuple = True)
        if (initial_state is None):
            c_input = tf.placeholder(tf.float32, [None, cell.state_size.c])
            h_input = tf.placeholder(tf.float32, [None, cell.state_size.h])
            initial_state = [c_input, h_input]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(initial_state[0],
                                                    initial_state[1])
        lstm_output, final_state = tf.nn.dynamic_rnn(cell = cell,
                            inputs = lstm_input, initial_state = state_in,
                            dtype = tf.float32, time_major = False)
        # TODO: check how to replace _kernel and _bias in operators of cell
        if (weight is not None):
            cell._kernel = weight
        if (bias is not None):
            cell._bias = bias
        lstm_op = {'output': lstm_output, 'final_state': final_state,
                                        'initial_state': initial_state}
        lstm_param = {'weight': cell._kernel, 'bias': cell._bias}
    return lstm_op, lstm_param

def dynamic_conv_lstm_layer(name, layer_input, output_channel,
                            kernel_size, add_bias = True,
                            initial_state = None):
    """
    Convolutional LSTM Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer
    output_channel : int
    output_channel = number of output channels
    kernel_size : list, tuple or int
    kernel_size = size of deconvolution filter, when kernel_size is list or tuple, it is [height, width]
    add_bias : bool
    add_bias = when this is True, bias is added
    initial_state : tuple or None
    initial_state = (c, h), their shape should be [batch_size] + cell.state_size.c,  [batch_size] + cell.state_size.h
    Returns:
    lstm_op : dictionary
    lstm_op = {'output': LSTM output, 'final_state': final state, 'initial_state': state input}
    lstm_param : dictionary
    lstm_param = {'weight': weight, 'bias': bias}
    """
    with tf.variable_scope(name) as lstm_sc:
        shape_input = [int(d) for d in layer_input.shape[-3:]]
        if (isinstance(kernel_size, int)):
            kernel_size = [kernel_size, kernel_size]
        cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims = 2,
                                input_shape = shape_input,
                                output_channels = output_channel,
                                kernel_shape = kernel_size,
                                use_bias = add_bias)
        if (initial_state is None):
            c_input = tf.placeholder(tf.float32, [None] +
                                    list(cell.state_size.c))
            h_input = tf.placeholder(tf.float32, [None] +
                                    list(cell.state_size.h))
            initial_state = [c_input, h_input]
        state_in = tf.nn.rnn_cell.LSTMStateTuple(initial_state[0],
                                                    initial_state[1])
        lstm_output, final_state = tf.nn.dynamic_rnn(cell = cell,
                                            inputs = layer_input,
                                            initial_state = state_in,
                                            dtype = tf.float32,
                                            time_major = False)
        param_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope = lstm_sc.name)
        weight, bias = None, None
        for param in param_list:
            if (param.name.find('kernel') >= 0):
                weight = param
            if (param.name.find('biases') >= 0):
                bias = param
        lstm_param = {'weight': weight}
        if (add_bias):
            lstm_param['bias'] = bias
        lstm_op = {'output': lstm_output, 'final_state': final_state,
                    'initial_state': initial_state}
    return lstm_op, lstm_param

