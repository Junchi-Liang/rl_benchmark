import tensorflow as tf
import numpy as np

def lstm_output_name(layer_id):
    """
    Get Name Of Output For LSTM Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    output_name : string
    output_name = name of LSTM output
    """
    output_name = layer_id + '/output'
    return output_name

def lstm_initial_state_name(layer_id):
    """
    Get Name Of Initial State For LSTM Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    initial_state_name : string
    initial_state_name = name of LSTM initial state
    """
    initial_state_name = layer_id + '/initial_state'
    return initial_state_name

def lstm_final_state_name(layer_id):
    """
    Get Name Of Final State For LSTM Layer
    Args:
    layer_id : string
    layer_id = name of the layer
    Returns:
    final_state_name : string
    final_state_name = name of LSTM final state
    """
    final_state_name = layer_id + '/final_state'
    return final_state_name

def lstm_unroll_size_name(lstm_layer_name):
    """
    Get The Name Of An LSTM Unroll Size
    Args:
    lstm_layer_name : string
    lstm_layer_name = name of the LSTM layer
    Returns:
    name : string
    name = name of an LSTM unroll size
    """
    name = lstm_layer_name + '/unroll_size'
    return name

def lstm_batch_size_name(lstm_layer_name):
    """
    Get The Name Of An LSTM Batch Size
    Args:
    lstm_layer_name : string
    lstm_layer_name = name of the LSTM layer
    Returns:
    name : string
    name = name of an LSTM batch size
    """
    name = lstm_layer_name + '/batch_size'
    return name

def lstm_input_name(lstm_layer_name):
    """
    Get The Name Of An LSTM Input
    Args:
    lstm_layer_name : string
    lstm_layer_name = name of the LSTM layer
    Returns:
    name : string
    name = name of an LSTM input
    """
    name = lstm_layer_name + '/input'
    return name

def to_lstm_input(name, layer_input, ops = None):
    """
    Convert Input Layer To LSTM Input
    Args:
    name : string
    name = name of the layer
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer, shape [None (batch_size * unroll_size), ...]
    ops : dictionary
    ops = collection of operators, if this is not None, generated operators is stored here.
            lstm_unroll_size_name = placeholder for unroll size
            lstm_batch_size_name = placeholder for batch size
            lstm_input_name = converted input
    Returns:
    ops : dictionary
    ops = collection of operators.
            lstm_unroll_size_name = placeholder for unroll size
            lstm_batch_size_name = placeholder for batch size
            lstm_input_name = converted input
    """
    if (ops is None):
        ops = {}
    ops[lstm_unroll_size_name(name)] = tf.placeholder(tf.int32, [])
    ops[lstm_batch_size_name(name)] = tf.placeholder(tf.int32, [])
    feat_size = int(np.prod(layer_input.shape[1:]))
    ops[lstm_input_name(name)] = tf.reshape(layer_input,
                            [ops[lstm_batch_size_name(name)],
                                ops[lstm_unroll_size_name(name)],
                                feat_size])
    return ops

def to_convlstm_input(name, layer_input, ops = None):
    """
    Convert Input Layer To Convolutional LSTM Input
    Args:
    name : string
    name = name of the layer
    layer_input : tensorflow.python.framework.ops.Tensor
    layer_input = input layer, shape [None (batch_size * unroll_size), ...]
    ops : dictionary
    ops = collection of operators, if this is not None, generated operators is stored here.
            lstm_unroll_size_name = placeholder for unroll size
            lstm_batch_size_name = placeholder for batch size
            lstm_input_name = converted input
    Returns:
    ops : dictionary
    ops = collection of operators.
            lstm_unroll_size_name = placeholder for unroll size
            lstm_batch_size_name = placeholder for batch size
            lstm_input_name = converted input
    """
    if (ops is None):
        ops = {}
    ops[lstm_unroll_size_name(name)] = tf.placeholder(tf.int32, [])
    ops[lstm_batch_size_name(name)] = tf.placeholder(tf.int32, [])
    feat_shape = [int(d) for d in layer_input.shape[-3:]]
    ops[lstm_input_name(name)] = tf.reshape(layer_input,
                            [ops[lstm_batch_size_name(name)],
                                ops[lstm_unroll_size_name(name)]] +
                                    feat_shape)
    return ops

def concat_lstm_output(lstm_output):
    """
    Concatenate Batch And Step Axis Of LSTM Output
    Args:
    lstm_output : tensorflow.python.framework.ops.Tensor
    lstm_output = LSTM output, shape [batch_size, unroll_size] + feature_shape
    Returns:
    concat_output: tensorflow.python.framework.ops.Tensor
    concat_output = concatenated output, shape [batch_size * unroll_size] + feature_shape
    """
    feature_shape = [int(d) for d in lstm_output.shape[2:]]
    concat_output = tf.reshape(lstm_output, [-1] + feature_shape)
    return concat_output
