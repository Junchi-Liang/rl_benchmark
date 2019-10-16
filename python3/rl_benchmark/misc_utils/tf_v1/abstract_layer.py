import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np

from rl_benchmark.misc_utils.tf_v1.layer import conv2d_layer, fc_layer, deconv2d_layer, dynamic_lstm_layer, dynamic_conv_lstm_layer
from rl_benchmark.misc_utils.tf_v1.lstm_helper import lstm_output_name, lstm_initial_state_name, lstm_final_state_name
from rl_benchmark.misc_utils.tf_v1.helper import activation_function

def kernel_name_conv2d(layer_id):
    """
    Get Name Of Kernel For Convolution Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    weight_name : string
    weight_name = name of the kernel
    """
    weight_name = 'w_' + layer_id
    return weight_name

def bias_name_conv2d(layer_id):
    """
    Get Name Of Bias For Convolution Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    bias_name : string
    bias_name = name of the bias
    """
    bias_name = 'b_' + layer_id
    return bias_name

def kernel_name_deconv2d(layer_id):
    """
    Get Name Of Kernel For Deconvolution Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    weight_name : string
    weight_name = name of the kernel
    """
    weight_name = 'w_' + layer_id
    return weight_name

def bias_name_deconv2d(layer_id):
    """
    Get Name Of Bias For Deconvolution Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    bias_name : string
    bias_name = name of the bias
    """
    bias_name = 'b_' + layer_id
    return bias_name

def weight_name_fc(layer_id):
    """
    Get Name Of Weight For Fully Connected Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    weight_name : string
    weight_name = name of the weight
    """
    weight_name = 'w_' + layer_id
    return weight_name

def bias_name_fc(layer_id):
    """
    Get Name Of Bias For Fully Connected Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    bias_name : string
    bias_name = name of the bias
    """
    bias_name = 'b_' + layer_id
    return bias_name

def weight_name_lstm(layer_id):
    """
    Get Name Of Weight For LSTM Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    weight_name : string
    weight_name = name of the weight
    """
    weight_name = 'w_' + layer_id
    return weight_name

def bias_name_lstm(layer_id):
    """
    Get Name Of Weight For LSTM Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    bias_name : string
    bias_name = name of the bias
    """
    bias_name = 'b_' + layer_id
    return bias_name

def activation_op_name(layer_input_id, activation_type):
    """
    Get Name Of Activation Operator
    Args:
    layer_input_id : string
    layer_input_id = name of the input layer for activation
    activation_type : string
    activation_type = type/name of activation
    Returns:
    activation_name : string
    activation_name = name of activation operator
    """
    activation_name = layer_input_id + '_' + activation_type
    return activation_name

def dropout_rate_name(layer_input_id):
    """
    Get Name Of Dropout Rate
    Args:
    layer_input_id : string
    layer_input_id = name of input layer of dropout
    Returns:
    dropout_rate_id : string
    dropout_rate_id = name of dropout rate
    """
    dropout_rate_id = 'dropout_rate_' + layer_input_id
    return dropout_rate_id

def dropout_op_name(layer_input_id):
    """
    Get Name Of Dropout Operator
    Args:
    layer_input_id : string
    layer_input_id = name of input layer of dropout
    Returns:
    dropout_name : string
    dropout_name = name of dropout name
    """
    dropout_name = 'dropout_' + layer_input_id
    return dropout_name

def conv2d(name, layer_input, stride, padding,
        kernel_size = None, input_channel = None, output_channel = None,
        add_bias = True, activation_param = None, add_dropout = False,
        weight_initializer = None, bias_initializer = None,
        param = None, ops = None):
    """
    2D Convolution Layer (With Activation And Dropout)
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
    add_bias : bool
    add_bias = when this is True, bias is added
    activation_param : string, tuple or list
    activation_param = when this is not None, it should be parameter for activation (e.g. 'relu', ['leakyrelu', 0.1])
    add_dropout : bool
    add_dropout = when this is True, dropout is included
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer_conv2d() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    param : dictionary
    param = collection of parameters, will look for parameters inside this dictionary, parameters will also be added to this dictionary
    ops : dictionary
    ops = collection of parameters, all operators in this layer will be added to this dictionary
    Returns:
    conv_op : tensorflow.python.framework.ops.Tensor
    conv_op = final output of this layer (may after activation and dropout if included)
    conv_param : dictionary
    conv_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    weight_name = kernel_name_conv2d(name)
    bias_name = bias_name_conv2d(name)
    if (param is not None and weight_name in param.keys()):
        weight = param[weight_name]
    else:
        weight = None
    if (param is not None and add_bias and bias_name in param.keys()):
        bias = param[bias_name]
    else:
        bias = None
    conv_op, conv_param = conv2d_layer(name, layer_input,
            stride, padding, kernel_size, input_channel, output_channel,
            weight, bias, add_bias,
            weight_initializer = weight_initializer,
            bias_initializer = bias_initializer)
    if (ops is not None):
        ops[name] = conv_op
    if (param is not None):
        param[weight_name] = conv_param['weight']
        if (add_bias):
            param[bias_name] = conv_param['bias']
    if (activation_param is not None):
        conv_op, activation_type = activation_function(conv_op,
                                                activation_param)
        if (ops is not None):
            ops[activation_op_name(name, activation_type)] = conv_op
    if (add_dropout):
        dropout_rate_id = dropout_rate_name(name)
        dropout_rate_ph = tf.placeholder(tf.float32, [], dropout_rate_id)
        if (ops is not None):
            ops[dropout_rate_id] = dropout_rate_ph
        conv_op = tf.nn.dropout(conv_op, 1 - dropout_rate_ph)
        dropout_name = dropout_op_name(name)
        if (ops is not None):
            ops[dropout_name] = conv_op
    return conv_op, conv_param

def deconv2d(name, layer_input, stride, padding,
            kernel_size = None,
            input_channel = None, output_channel = None,
            add_bias = True, activation_param = None, add_dropout = False,
            weight_initializer = None, bias_initializer = None,
            output_shape = None, param = None, ops = None):
    """
    2D Deconvolution Layer (With Activation And Dropout)
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
    add_bias : bool
    add_bias = when this is True, bias is added
    activation_param : string, tuple or list
    activation_param = when this is not None, it should be parameter for activation (e.g. 'relu', ['leakyrelu', 0.1])
    add_dropout : bool
    add_dropout = when this is True, dropout is included
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer_conv2d() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    output_shape : list or tuple or string or tensor or None
    output_shape = output shape of this layer.
                    when it is list or tuple ([h, w] or [b, h, w, c]),it is exactly the output size.
                    when it is string, the output size will match ops[output_shape].
                    when it is a tensor, the output size will match this tensor.
                    when this is None, the shape will be infered from deconv_output_length
    param : dictionary
    param = collection of parameters, will look for parameters inside this dictionary, parameters will also be added to this dictionary
    ops : dictionary
    ops = collection of parameters, all operators in this layer will be added to this dictionary
    Returns:
    deconv_op : tensorflow.python.framework.ops.Tensor
    deconv_op = final output of this layer (may after activation and dropout if included)
    deconv_param : dictionary
    deconv_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    weight_name = kernel_name_deconv2d(name)
    bias_name = bias_name_deconv2d(name)
    if (param is not None and weight_name in param.keys()):
        weight = param[weight_name]
    else:
        weight = None
    if (param is not None and add_bias and bias_name in param.keys()):
        bias = param[bias_name]
    else:
        bias = None
    if (output_shape is None):
        deconv_op, deconv_param = deconv2d_layer(name, layer_input,
                stride, padding, kernel_size,
                input_channel, output_channel,
                weight, bias, add_bias,
                weight_initializer, bias_initializer, output_shape)
    elif (isinstance(output_shape, list) or
            isinstance(output_shape, tuple)):
        assert (len(output_shape) == 2 or len(output_shape) == 4)
        if (len(output_shape) == 2):
            output_shape = [array_ops.shape(layer_input)[0],
                            output_shape[0], output_shape[1],
                            output_channel]
        deconv_op, deconv_param = deconv2d_layer(name, layer_input,
                stride, padding, kernel_size,
                input_channel, output_channel,
                weight, bias, add_bias,
                weight_initializer, bias_initializer, output_shape)
    elif (isinstance(output_shape, str)):
        assert (ops is not None and isinstance(ops, dict))
        deconv_op, deconv_param = deconv2d_layer(name, layer_input,
                stride, padding, kernel_size,
                input_channel, output_channel,
                weight, bias, add_bias,
                weight_initializer, bias_initializer,
                matched_tensor = ops[output_shape])
    else:
        deconv_op, deconv_param = deconv2d_layer(name, layer_input,
                stride, padding, kernel_size,
                input_channel, output_channel,
                weight, bias, add_bias,
                weight_initializer, bias_initializer,
                matched_tensor = output_shape)
    if (ops is not None):
        ops[name] = deconv_op
    if (param is not None):
        param[weight_name] = deconv_param['weight']
        if (add_bias):
            param[bias_name] = deconv_param['bias']
    if (activation_param is not None):
        deconv_op, activation_type = activation_function(deconv_op,
                                                    activation_param)
        if (ops is not None):
            ops[activation_op_name(name, activation_type)] = deconv_op
    if (add_dropout):
        dropout_rate_id = dropout_rate_name(name)
        dropout_rate_ph = tf.placeholder(tf.float32, [], dropout_rate_id)
        if (ops is not None):
            ops[dropout_rate_id] = dropout_rate_ph
        deconv_op = tf.nn.dropout(deconv_op, 1 - dropout_rate_ph)
        dropout_name = dropout_op_name(name)
        if (ops is not None):
            ops[dropout_name] = deconv_op
    return deconv_op, deconv_param

def fully_connected(name, layer_input, 
        input_length = None, output_length = None,
        add_bias = True, activation_param = None, add_dropout = False,
        weight_initializer = None, bias_initializer = None,
        param = None, ops = None):
    """
    Fully Connected Layer (With Activation And Dropout)
    Args:
    name : string
    name = name of this layer (will be used as scope)
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer
    input_length : int
    input_length = length of input vector
    output_length : int
    output_length = length of output vector
    add_bias : bool
    add_bias = when this is True, bias is added
    activation_param : string, tuple or list
    activation_param = when this is not None, it should be parameter for activation (e.g. 'relu', ['leakyrelu', 0.1])
    add_dropout : bool
    add_dropout = when this is True, dropout is included
    weight_initializer : function
    weight_initializer = initializer for weight, when this is None, tf.contrib.layers.xavier_initializer() is used
    bias_initializer : function
    bias_initializer = initializer for bias, when this is None, bias is initialized as zero
    param : dictionary
    param = collection of parameters, will look for parameters inside this dictionary, parameters will also be added to this dictionary
    ops : dictionary
    ops = collection of parameters, all operators in this layer will be added to this dictionary
    Returns:
    fc_op : tensorflow.python.framework.ops.Tensor
    fc_op = output fully connected layer
    fc_param : dictionary
    fc_param = {'weight': weight} or {'weight': weight, 'bias': bias} where weight and bias are tf.Variable
    """
    weight_name = weight_name_fc(name)
    bias_name = bias_name_fc(name)
    if (param is not None and weight_name in param.keys()):
        weight = param[weight_name]
    else:
        weight = None
    if (param is not None and add_bias and bias_name in param.keys()):
        bias = param[bias_name]
    else:
        bias = None
    fc_op, fc_param = fc_layer(name, layer_input,
                                input_length, output_length,
                                weight, bias, add_bias,
                                weight_initializer, bias_initializer)
    if (ops is not None):
        ops[name] = fc_op
    if (param is not None):
        param[weight_name] = fc_param['weight']
        if (add_bias):
            param[bias_name] = fc_param['bias']
    if (activation_param is not None):
        fc_op, activation_type = activation_function(fc_op,
                                                    activation_param)
        if (ops is not None):
            ops[activation_op_name(name, activation_type)] = fc_op
    if (add_dropout):
        dropout_rate_id = dropout_rate_name(name)
        dropout_rate_ph = tf.placeholder(tf.float32, [], dropout_rate_id)
        if (ops is not None):
            ops[dropout_rate_id] = dropout_rate_ph
        fc_op = tf.nn.dropout(fc_op, 1 - dropout_rate_ph)
        dropout_name = dropout_op_name(name)
        if (ops is not None):
            ops[dropout_name] = fc_op
    return fc_op, fc_param

def dynamic_lstm(name, layer_input, num_units,
                initial_state = None, param = None, ops = None):
    """
    LSTM Layer
    Args:
    name : string
    name = name of this layer (will be used as scope)
    num_units: int
    num_units = the number of units in the LSTM cell
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer, shape [batch_size, max_time, ...]
    initial_state : tuple or None
    initial_state = (c, h), their shape should be [batch_size, cell.state_size.c],  [batch_size, cell.state_size.h]
    param : dictionary
    param = collection of parameters, will look for parameters inside this dictionary, parameters will also be added to this dictionary
    ops : dictionary
    ops = collection of parameters, all operators in this layer will be added to this dictionary
    Returns:
    lstm_op : dictionary
    lstm_op = {'output': LSTM output, 'final_state': final state, 'initial_state': state input}
    lstm_param : dictionary
    lstm_param = {'weight': weight, 'bias': bias}
    """
    weight_name = weight_name_lstm(name)
    bias_name = bias_name_lstm(name)
    if (param is not None and weight_name in param.keys()):
        weight = param[weight_name]
    else:
        weight = None
    if (param is not None and bias_name in param.keys()):
        bias = param[bias_name]
    else:
        bias = None
    lstm_op, lstm_param = dynamic_lstm_layer(name, num_units,
                                    layer_input, initial_state,
                                                    weight, bias)
    if (ops is not None):
        ops[lstm_output_name(name)] = lstm_op['output']
        ops[lstm_initial_state_name(name)] = lstm_op['initial_state']
        ops[lstm_final_state_name(name)] = lstm_op['final_state']
    if (param is not None):
        param[weight_name] = lstm_param['weight']
        param[bias_name] = lstm_param['bias']
    return lstm_op, lstm_param

def dynamic_conv_lstm(name, layer_input, output_channel, kernel_size,
                                add_bias = True, initial_state = None,
                                param = None, ops = None):
    """
    Convolutional LSTM Layer
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
    param : dictionary
    param = collection of parameters, will look for parameters inside this dictionary, parameters will also be added to this dictionary
    ops : dictionary
    ops = collection of parameters, all operators in this layer will be added to this dictionary
    Returns:
    lstm_op : dictionary
    lstm_op = {'output': LSTM output, 'final_state': final state, 'initial_state': state input}
    lstm_param : dictionary
    lstm_param = {'weight': weight, 'bias': bias}
    """
    weight_name = weight_name_lstm(name)
    bias_name = bias_name_lstm(name)
    lstm_op, lstm_param = dynamic_conv_lstm_layer(name, layer_input,
                                        output_channel, kernel_size,
                                        add_bias, initial_state)
    if (ops is not None):
        ops[lstm_output_name(name)] = lstm_op['output']
        ops[lstm_initial_state_name(name)] = lstm_op['initial_state']
        ops[lstm_final_state_name(name)] = lstm_op['final_state']
    if (param is not None):
        param[weight_name] = lstm_param['weight']
        if (add_bias):
            param[bias_name] = lstm_param['bias']
    return lstm_op, lstm_param

