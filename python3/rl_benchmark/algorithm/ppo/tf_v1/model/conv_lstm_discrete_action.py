import tensorflow as tf
import numpy as np
import jason, json

from rl_benchmark.misc_utils.tf_v1.lstm_helper import to_lstm_input, lstm_input_name, lstm_output_name, lstm_initial_state_name, lstm_final_state_name, lstm_unroll_size_name, lstm_batch_size_name, concat_lstm_output
from rl_benchmark.misc_utils.tf_v1.abstract_layer import conv2d, fully_connected, dynamic_lstm

class PPOModel(object):
    """
    Model For PPO
    """
    def __init__(self, scope_name, config_file_path = None,
                 img_resolution = None, n_action = None, lstm_h_size = None,
                 conv_arch = [[8, 32, 4, 'SAME', True, 'relu'],
                              [4, 64, 2, 'SAME', True, 'relu'],
                              [3, 64, 1, 'SAME', True, 'relu']],
                 fc_before_lstm = [[512, True, 'relu']]):
        """
        Args:
        scope_name : string
        scope_name = name of this model, used as scope
        config_file_path : string
        config_file_path = path to config file of the model, when this is not None, other parameters will be ignored
        img_resolution : list
        img_resolution = resolution of input images, [h, w, channels]
        n_action : int
        n_action = number of actions
        lstm_h_size : int
        lstm_h_size = size of hidden state in LSTM
        conv_arch : list
        conv_arch = architecture of convolution layers,
                    each of its element is [kernel size, output_channel, stride, padding, add_bias, activation]
        fc_before_lstm : list
        fc_before_lstm = fully connected layers before LSTM, each of its element is [number of hidden units, add_bias, activation]
        """
        self.scope_name = scope_name
        if (config_file_path is not None):
            self.load_config_from_json(config_file_path)
            if (n_action is not None):
                self.n_action = n_action
        else:
            self.img_resolution = img_resolution
            self.n_action = n_action
            self.lstm_h_size = lstm_h_size
            self.conv_arch = conv_arch
            self.fc_before_lstm = fc_before_lstm
        with tf.variable_scope(self.scope_name) as model_scope:
            self.scope = model_scope
            self.ops, self.param, self.lstm_list = self.build()
            self.var_init_ops = tf.variables_initializer([
                            self.param[k] for k in self.param.keys()])

    def contain_lstm(self):
        """
        Check If This Model Has LSTM Layers
        Returns:
        has_lstm : bool
        has_lstm = this model has LSTM
        """
        if (self.lstm_h_size is not None and self.lstm_h_size > 0):
            return True
        return False

    def lstm_layer(self):
        """
        Get A List Of Names For LSTM Layers
        Returns:
        lstm_name_list : string
        lstm_name_list = list of names of LSTM layers
        """
        return self.lstm_list

    def zero_lstm_state(self, batch_size = 1, dtype = np.float32):
        """
        Get Zero State for LSTM
        Args:
        batch_size : int
        batch_size = batch size
        dtype : type
        dtype = type for the state
        Returns:
        zero_state : dictionary
        zero_state = zero state, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
        zero_state : list
        zero_state = [zero_c, zero_h], shapes of zero_c and zero_h are both [batch_size, ...]
        """
        assert self.contain_lstm()
        zero_state = {}
        for lstm_layer_id in self.lstm_layer():
            zero_c = np.zeros([batch_size, self.lstm_h_size], dtype)
            zero_h = np.zeros([batch_size, self.lstm_h_size], dtype)
            zero_state[lstm_layer_id] = [zero_c, zero_h]
        return zero_state

    def write_config_to_json(self, config_file_path):
        """
        Output Configuration To jason File
        Args:
        config_file_path : string
        config_file_path = path to jason file
        """
        config_output = {'img_resolution': self.img_resolution,
                    'n_action': self.n_action,
                    'lstm_h_size': self.lstm_h_size,
                    'conv_arch': self.conv_arch,
                    'fc_before_lstm': self.fc_before_lstm}
        with open(config_file_path, 'w') as config_file:
            config_file.write(json.dumps(config_output, sort_keys=True))

    def load_config_from_json(self, config_file_path):
        """
        Load Configuration From jason File
        Args:
        config_file_path : string
        config_file_path = path to jason file
        """
        config_input = json.load(open(config_file_path))
        self.img_resolution = config_input['img_resolution']
        self.n_action = config_input['n_action']
        self.lstm_h_size = config_input['lstm_h_size']
        self.conv_arch = config_input['conv_arch']
        self.fc_before_lstm = config_input['fc_before_lstm']

    def build(self, param = None):
        """
        Create Variables And Operators For This Model
        Args:
        param : dictionary
        param = collection of parameters, when this is None, parameters are generated inside this procedure
        Returns:
        ops : dictionary
        ops = collection of operators
        param : dictionary
        param = collection of parameter
        lstm_list : list
        lstm_list = list of lstm layers id
        """
        ops = {}
        lstm_list = []
        if (param is None):
            param = {}
        layer_input = tf.placeholder(tf.float32, [None] +
                                        self.img_resolution, 'input')
        ops['input'] = layer_input
        last_layer = layer_input
        channel_last_layer = int(last_layer.shape[-1])
        # convolution layers
        for i_conv, conv_config in enumerate(self.conv_arch):
            kernel_size, output_channel, stride,\
                    padding, add_bias, activation = conv_config
            layer_id = 'conv%d' % i_conv
            last_layer, _ = conv2d(layer_id, last_layer, stride,
                                    padding, kernel_size,
                                    channel_last_layer, output_channel,
                                    add_bias, activation, False,
                                    param = param, ops = ops)
            channel_last_layer = output_channel
        conv_feat_size = int(last_layer.shape[1]) *\
                            int(last_layer.shape[2]) *\
                            int(last_layer.shape[3])
        last_layer = ops['conv_flat'] = tf.reshape(last_layer,
                                    shape = [-1, conv_feat_size])
        last_size = conv_feat_size
        # fully connected layers
        for i_fc, fc_config in enumerate(self.fc_before_lstm):
            num_hidden_unit, add_bias, activation = fc_config
            layer_id = 'fc%d' % i_fc
            last_layer, _ = fully_connected(layer_id, last_layer,
                            last_size, num_hidden_unit, add_bias,
                            activation, False, param = param, ops = ops)
            last_size = num_hidden_unit
        # LSTM
        if (self.contain_lstm()):
            layer_id = 'lstm'
            lstm_list.append(layer_id)
            to_lstm_input(layer_id, last_layer, ops = ops)
            lstm_ops, _ = dynamic_lstm(layer_id,
                    ops[lstm_input_name(layer_id)],
                    self.lstm_h_size, param = param, ops = ops)
            last_layer = concat_lstm_output(lstm_ops['output'])
            last_size = self.lstm_h_size
        # value and policy
        pi_ops, _ = fully_connected('pi', last_layer,
                                    last_size, self.n_action,
                                    False, None, False,
                                    param = param, ops = ops)
        ops['pi_preactivation'] = pi_ops
        ops['pi'] = tf.nn.softmax(pi_ops)
        value_ops, _ = fully_connected('value', last_layer,
                                        last_size, 1, False, None, False,
                                        param =  param, ops = ops)
        return ops, param, lstm_list

    def build_loss(self, coeff_value, coeff_policy, coeff_entropy,
                value_output, policy_output,
                log_policy_output = None, policy_logits = None):
        """
        Get Operators for Loss
        Args:
        coeff_value : float
        coeff_value = coefficient for value loss
        coeff_policy : float
        coeff_policy = coefficient for policy gradient
        coeff_entropy : float
        coeff_entropy  = coefficient for entropy
        value_output : tensorflow.python.framework.ops.Tensor
        value_output = tensor of output values from the model, shape [batch_size (None), 1]
        policy_output : tensorflow.python.framework.ops.Tensor
        policy_output = tensor of policy distribution from the model, shape [batch_size (None), number of actions]
        log_policy_output : tensorflow.python.framework.ops.Tensor
        log_policy_output = log(policy_output)
        policy_logits = tensorflow.python.framework.ops.Tensor
        policy_logits = logits of policy (input to softmax)
        Returns:
        ops : dictionary
        ops = collection of operators
        """
        if (log_policy_output is None):
            log_policy_output = tf.log(policy_output)
        ops = {}
        ops['action_index_input'] = tf.placeholder(shape = [None],
                                                   dtype = tf.int32)
        ops['target_value_input'] = tf.placeholder(shape = [None],
                                                   dtype = tf.float32)
        ops['advantage_input'] = tf.placeholder(shape = [None],
                                                 dtype = tf.float32)
        ops['old_neg_logpac'] = tf.placeholder(shape = [None],
                                                    dtype = tf.float32)
        ops['old_value'] = tf.placeholder(shape = [None],
                                                    dtype = tf.float32)
        ops['policy_clip_range'] = tf.placeholder(shape = [],
                                                    dtype = tf.float32)
        ops['value_clip_range'] = tf.placeholder(shape = [],
                                                    dtype = tf.float32)
        ops['policy_entropy'] = -tf.reduce_mean(
                                tf.reduce_sum(tf.multiply(
                                    policy_output, log_policy_output),
                                    axis = 1))
        value_squeezed = tf.squeeze(value_output, axis = -1)
        value_clipped = ops['old_value'] + tf.clip_by_value(
                                    value_squeezed - ops['old_value'],
                                    -ops['value_clip_range'],
                                        ops['value_clip_range'])
        ops['value_loss_unclipped'] = tf.square(value_squeezed -
                                            ops['target_value_input'])
        ops['value_loss_clipped'] = tf.square(value_clipped -
                                            ops['target_value_input'])
        ops['value_loss'] = 0.5 * tf.reduce_mean(tf.maximum(
                                        ops['value_loss_unclipped'],
                                        ops['value_loss_clipped']))
        if (policy_logits is not None):
            ops['neg_logpac'] =\
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits = policy_logits,
                                    labels = ops['action_index_input'])
        else:
            ops['action_one_hot'] = tf.one_hot(ops['action_index_input'],
                                                            self.n_action,
                                                on_value = 1, off_value = 0,
                                                        dtype = tf.float32)
            ops['neg_logpac'] = -tf.reduce_sum(tf.multiply(
                                                        log_policy_output,
                                                    ops['action_one_hot']),
                                                                axis = -1)
        ops['policy_ratio'] = tf.exp(ops['old_neg_logpac'] -
                                            ops['neg_logpac'])
        ops['neg_policy_loss_unclipped'] = -tf.multiply(
                                            ops['advantage_input'],
                                            ops['policy_ratio'])
        ops['neg_policy_loss_clipped'] = -tf.multiply(
                                            ops['advantage_input'],
                                    tf.clip_by_value(ops['policy_ratio'],
                                        1. - ops['policy_clip_range'],
                                        1. + ops['policy_clip_range']))
        ops['neg_policy_loss'] = tf.reduce_mean(tf.maximum(
                                    ops['neg_policy_loss_unclipped'],
                                    ops['neg_policy_loss_clipped']))
        ops['loss'] = coeff_value * ops['value_loss'] +\
                        coeff_policy * ops['neg_policy_loss'] -\
                        coeff_entropy * ops['policy_entropy']
        return ops

    def build_update(self, optimizer, clip_norm, loss_func):
        """
        Operators For Optimizing The Model
        Args:
        optimizer : tensorflow.python.training.optimizer.Optimizer
        optimizer = optimizer used for updating parameters
        clip_norm : float
        clip_norm = The clipping ratio
        loss_func : tensorflow.python.framework.ops.Tensor
        loss_func = loss function which should be optimized
        Returns:
        update_ops : tensorflow.python.framework.ops.Tensor
        update_ops = operators for update
        """
        param_key = self.param.keys()
        var_list = [self.param[k] for k in param_key]
        gradient = tf.gradients(loss_func, var_list)
        clipped_grad, grad_norm = tf.clip_by_global_norm(gradient,
                                                        clip_norm)
        grads_and_vars = [(clipped_grad[i], self.param[k])
                          for i, k in enumerate(param_key)]
        update_ops = optimizer.apply_gradients(grads_and_vars)
        return update_ops

    def init_param(self, tf_sess):
        """
        Initialize Parameters Of This Model
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for initialization
        """
        tf_sess.run(self.var_init_ops)

    def load_from_npz(self, tf_sess, npz_path):
        """
        Load Parameters From NPZ FILE
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for variable assignment
        npz_path : string
        npz_path = path to npz file
        """
        self.init_param(tf_sess)
        param_loaded = np.load(npz_path)
        for param_key in self.param.keys():
            if (param_key in param_loaded.keys()):
                tf_sess.run(self.param[param_key].assign(
                                param_loaded[param_key]))

    def save_to_npz(self, tf_sess, npz_path):
        """
        Save Parameters To NPZ File
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for parameter computation
        npz_path : string
        npz_path = path to npz file
        """
        p = {}
        for param_key in self.param.keys():
            p[param_key] = tf_sess.run(self.param[param_key])
        np.savez(npz_path, **p)

    def prepare_feed_dict(self, img_input, lstm_state_input,
            batch_size = None, seq_len = None, loss_ops = None,
            action_index_input = None, target_value_input = None,
            advantage_input = None, old_neg_logpac_input = None,
            old_value_input = None, policy_clip_range_input = None,
            value_clip_range_input = None):
        """
        Preparing Feed Dict
        Args:
        img_input : numpy.ndarray
        img_input = current frame input, shape (batch size * seq len, h, w, c) or (h, w, c)
        lstm_state_input : dictionary
        lstm_state_input = collection of lstm state input, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size) or (hidden size)
        batch_size : int
        batch_size = batch size
        seq_len : int
        seq_len = length of sequences
        loss_ops : dictionary
        loss_ops = collection of loss operators, should be provided for loss evaluation
        action_index_input : numpy.ndarray
        action_index_input = action index input for loss evaluation,
                        shape (batch size * seq len)
        target_value_input : numpy.ndarray
        target_value_input = target value input for loss evaluation,
                        shape (batch size * seq len)
        advantage_input : numpy.ndarray
        advantage_input = input advantage estimation for loss evaluation,
                        shape (batch size * seq len)
        old_neg_logpac_input : numpy.ndarray
        old_neg_logpac_input = old policy negative log(pi[a]) input for loss evaluation,
                        shape (batch size * seq len)
        old_value_input : numpy.ndarray
        old_value_input = value of states from old policy,
                        shape (batch size * seq len)
        policy_clip_range_input : float
        policy_clip_range_input = policy clip range
        value_clip_range_input : float
        value_clip_range_input = value clip range
        Returns:
        feed_dict : dictionary
        feed_dict = feed dictionary for tensorflow
        """
        feed_dict = {}
        if (img_input.ndim == 4):
            feed_dict[self.ops['input']] = img_input
        elif (img_input.ndim == 3):
            feed_dict[self.ops['input']] = np.expand_dims(
                    img_input, axis = 0)
        else:
            raise NotImplementedError
        if ((batch_size is None) or (seq_len is None)):
            batch_size = feed_dict[self.ops['input']].shape[0]
            seq_len = 1
        for lstm_layer_id in self.lstm_layer():
            feed_dict[self.ops[
                lstm_batch_size_name(lstm_layer_id)]] = batch_size
            feed_dict[self.ops[
                lstm_unroll_size_name(lstm_layer_id)]] = seq_len
            for s in range(2):
                if (lstm_state_input[lstm_layer_id][s].ndim == 1):
                    feed_dict[self.ops[
                        lstm_initial_state_name(lstm_layer_id)][s]
                        ] = np.expand_dims(
                            lstm_state_input[lstm_layer_id][s], axis = 0)
                elif (lstm_state_input[lstm_layer_id][s].ndim == 2):
                    feed_dict[self.ops[
                        lstm_initial_state_name(lstm_layer_id)][s]
                        ] = lstm_state_input[lstm_layer_id][s]
                else:
                    raise NotImplementedError
        if (loss_ops is not None):
            if (isinstance(action_index_input, int) or
                    isinstance(action_index_input, float)):
                feed_dict[loss_ops['action_index_input']
                        ] = np.array([action_index_input])
            elif (action_index_input.ndim == 1):
                feed_dict[loss_ops['action_index_input']
                        ] = action_index_input
            else:
                raise NotImplementedError
            if (isinstance(target_value_input, int) or
                    isinstance(target_value_input, float)):
                feed_dict[loss_ops['target_value_input']
                        ] = np.array([target_value_input])
            elif (target_value_input.ndim == 1):
                feed_dict[loss_ops['target_value_input']
                        ] = target_value_input
            else:
                raise NotImplementedError
            if (isinstance(advantage_input, int) or
                    isinstance(advantage_input, float)):
                feed_dict[loss_ops['advantage_input']
                        ] = np.array([advantage_input])
            elif (advantage_input.ndim == 1):
                feed_dict[loss_ops['advantage_input']] = advantage_input
            else:
                raise NotImplementedError
            if (isinstance(old_neg_logpac_input, int) or
                    isinstance(old_neg_logpac_input, float)):
                feed_dict[loss_ops['old_neg_logpac']
                        ] = np.array([old_neg_logpac_input])
            elif (old_neg_logpac_input.ndim == 1):
                feed_dict[loss_ops['old_neg_logpac']
                        ] = old_neg_logpac_input
            else:
                raise NotImplementedError
            if (isinstance(old_value_input, int) or
                    isinstance(old_value_input, float)):
                feed_dict[loss_ops['old_value']
                        ] = np.array([old_value_input])
            elif (old_value_input.ndim == 1):
                feed_dict[loss_ops['old_value']] = old_value_input
            else:
                raise NotImplementedError
            feed_dict[loss_ops['policy_clip_range']
                    ] = policy_clip_range_input
            feed_dict[loss_ops['value_clip_range']
                    ] = value_clip_range_input
        return feed_dict

    def sample_action(self, tf_sess, img_input, lstm_state_input = None):
        """
        Sample Actions
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for computation
        img_input : numpy.ndarray
        img_input = current frame input, shape (batch size * seq len, h, w, c) or (h, w, c)
        lstm_state_input : dictionary or None
        lstm_state_input = collection of lstm state input, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size) or (hidden size)
        Returns:
        action_info : dictionary
        action_info = information of sampled action
            "action_index" : numpy.ndarray or int
            "action_index" = batch of actions, shape [batch size]
            "lstm_state_output" : tuple or None
            "lstm_state_output" = state output of lstm, [c, h],
                c, h : numpy.ndarray
                shape of c is (batch size, lstm_cell.state_size.c) or (lstm_cell.state_size.c),
                shape of h is (batch size, lstm_cell.state_size.h) or (lstm_cell.state_size.h)
            "state_value" : numpy.ndarray or float
            "state_value" = value function of current state, shape [batch size]
            "pi" : numpy.ndarray or float
            "pi" = policy, shape [batch size, number of actions] or [number of actions]
        """
        feed_dict = self.prepare_feed_dict(img_input, lstm_state_input)
        req_list = [self.ops['pi'], self.ops['value']]
        for lstm_layer_id in self.lstm_layer():
            req_list.append(self.ops[lstm_final_state_name(lstm_layer_id)])
        response = tf_sess.run(req_list, feed_dict = feed_dict)
        pi, state_value = response[:2]
        lstm_state_output = {}
        for lstm_layer_id, res_value in zip(self.lstm_layer(), response[2:]):
            lstm_state_tuple = []
            for s in range(2):
                if (lstm_state_input[lstm_layer_id][s].ndim == 1):
                    lstm_state_tuple.append(
                            np.expand_dims(res_value[s], axis = 0))
                elif (lstm_state_input[lstm_layer_id][s].ndim == 2):
                    lstm_state_tuple.append(res_value[s])
            lstm_state_output[lstm_layer_id] = lstm_state_tuple
        state_value = np.squeeze(state_value, axis = -1)
        action_index = []
        for i in range(pi.shape[0]):
            action_index.append(np.random.choice(self.n_action, p = pi[i]))
        if (img_input.ndim == 3):
            action_index = int(action_index[0])
            state_value = state_value[0]
            pi = pi[0]
        else:
            action_index = np.array(action_index, np.int)
        action_info = {'action_index': action_index,
                        'lstm_state_output': lstm_state_output,
                        'state_value': state_value,
                        'pi': pi}
        return action_info

