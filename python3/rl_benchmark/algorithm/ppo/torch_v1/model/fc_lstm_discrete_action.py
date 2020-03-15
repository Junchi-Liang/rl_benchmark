import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from rl_benchmark.misc_utils.torch_v1.conv_helper import conv_output_size
from rl_benchmark.misc_utils.torch_v1.helper import Flatten, convert_to_tuple, get_activation, calculate_gain_from_activation
import jason, json

class PPOModel(nn.Module):
    """
    Model For PPO
    """
    def __init__(self, config_file_path = None,
                 state_shape = None, n_action = None, lstm_h_size = None,
                 fc_config_before_lstm = [],
                 fc_config_after_lstm = [[256, True, 'relu']]):
        """
        Args:
        config_file_path : string
        config_file_path = path to config file of the model, when this is not None, other parameters will be ignored
        state_shape : tuple or list
        state_shape = shape of state input
        n_action : int
        n_action = number of actions
        lstm_h_size : int
        lstm_h_size = size of hidden state in LSTM
        fc_before_lstm : list
        fc_before_lstm = fully connected layers before LSTM, each of its element is [number of hidden units, add_bias, activation]
        fc_config_after_lstm : configuration for fully connected layers after LSTM, each element is [number of hidden units, add_bias, activation]
        """
        super(PPOModel, self).__init__()
        if (config_file_path is not None):
            self.load_config_from_json(config_file_path)
            if (n_action is not None):
                self.n_action = n_action
        else:
            self.state_shape = state_shape
            self.n_action = n_action
            self.lstm_h_size = lstm_h_size
            self.fc_config_before_lstm = fc_config_before_lstm
            self.fc_config_after_lstm = fc_config_after_lstm
        self.build()

    def write_config_to_json(self, config_file_path):
        """
        Output Configuration To jason File
        Args:
        config_file_path : string
        config_file_path = path to jason file
        """
        config_output = {'state_shape': self.state_shape,
                    'n_action': self.n_action,
                    'lstm_h_size': self.lstm_h_size,
                    'fc_config_before_lstm': self.fc_config_before_lstm,
                    'fc_config_after_lstm': self.fc_config_after_lstm}
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
        self.state_shape = config_input['state_shape']
        self.n_action = config_input['n_action']
        self.lstm_h_size = config_input['lstm_h_size']
        self.fc_config_before_lstm = config_input['fc_config_before_lstm']
        self.fc_config_after_lstm = config_input['fc_config_after_lstm']

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

    def zero_lstm_state(self, batch_size = 1,
            dtype = torch.float32, to_numpy = True):
        """
        Get Zero State for LSTM
        Args:
        batch_size : int
        batch_size = batch size
        dtype : type
        dtype = type for the state
        to_numpy : bool
        to_numpy = if returned tensor is converted to numpy.ndarray
        Returns:
        zero_state : dictionary
        zero_state = zero state, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        h0, c0 : numpy.ndarray or torch.Tensor
                        shape of h0, c0 is (1, batch size, hidden size)
        """
        assert self.contain_lstm()
        zero_state = {}
        for lstm_layer_id in self.lstm_layer():
            h0 = torch.zeros([1, batch_size, self.lstm_h_size],
                    dtype = dtype)
            c0 = torch.zeros([1, batch_size, self.lstm_h_size],
                    dtype = dtype)
            if (to_numpy):
                h0 = h0.numpy()
                c0 = c0.numpy()
            zero_state[lstm_layer_id] = (h0, c0)
        return zero_state

    def build(self):
        """
        Construct Submodule And Prepare Parameters
        """
        self.state_ndim = len(self.state_shape)
        self.state_size = np.prod(self.state_shape)
        last_size = self.state_size
        # fully connected before LSTM
        if (self.fc_config_before_lstm is not None and
                len(self.fc_config_before_lstm) > 0):
            submodule = OrderedDict()
            for i_layer, layer_config in enumerate(
                    self.fc_config_before_lstm):
                num_hidden_unit, add_bias, activation = layer_config
                last_layer = submodule['fc%d' % i_layer] = nn.Linear(
                        last_size, num_hidden_unit, bias = add_bias)
                nn.init.xavier_uniform_(last_layer.weight,
                        calculate_gain_from_activation(activation))
                if (add_bias):
                    nn.init.constant_(last_layer.bias, 0)
                last_size = num_hidden_unit
            self.fc_function_before_lstm = nn.Sequential(submodule)
        else:
            self.fc_function_before_lstm = None
        # LSTM
        self.lstm_list = []
        if (self.contain_lstm()):
            module_id = 'lstm'
            self.lstm_list.append(module_id)
            self.__setattr__(module_id, nn.LSTM(last_size, self.lstm_h_size))
            nn.init.xavier_uniform_(self.__getattr__(module_id).weight_hh_l0)
            nn.init.xavier_uniform_(self.__getattr__(module_id).weight_ih_l0)
            nn.init.constant_(self.__getattr__(module_id).bias_ih_l0, 0)
            nn.init.constant_(self.__getattr__(module_id).bias_hh_l0, 0)
            last_size = self.lstm_h_size
        # fully connected after LSTM
        if (self.fc_config_after_lstm is not None and
                len(self.fc_config_after_lstm) > 0):
            submodule = OrderedDict()
            for i_layer, layer_config in enumerate(
                    self.fc_config_after_lstm):
                num_hidden_unit, add_bias, activation = layer_config
                last_layer = submodule['fc%d' % i_layer] = nn.Linear(
                        last_size, num_hidden_unit, bias = add_bias)
                nn.init.xavier_uniform_(last_layer.weight,
                        calculate_gain_from_activation(activation))
                if (add_bias):
                    nn.init.constant_(last_layer.bias, 0)
                last_size = num_hidden_unit
            self.fc_function_after_lstm = nn.Sequential(submodule)
        else:
            self.fc_function_after_lstm = None
        # policy and value
        self.policy_branch = nn.Linear(last_size, self.n_action,
                                        bias = False)
        nn.init.xavier_uniform_(self.policy_branch.weight)
        self.value_branch = nn.Linear(last_size, 1,
                                        bias = False)
        nn.init.xavier_uniform_(self.value_branch.weight)
        self.policy_softmax = nn.Softmax(dim = -1)

    def forward(self, state_batch, lstm_state_batch = None):
        """
        Forward The Architecture
        Args:
        state_batch : torch.tensor
        state_batch = batch input state,
            shape (batch size or sequence size * batch size) + state_shape
        lstm_state_batch : dictionary
        lstm_state_batch = batch input lstm state, indexed by names in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are torch.tensor
                            shape of h0, c0 is (1, batch size, hidden size)
        Returns:
        result : dictionary
        result = collection of forward results
            'pi_logits' : torch.tensor
            'pi_logits' = logits for policy, shape (seq len * batch size, n_action)
            'pi' : torch.tensor
            'pi' = policy, shape (seq len * batch size, n_action)
            'value' : torch.tensor
            'value' = value, shape (seq len * batch size, 1)
            'lstm_state_output' : dictionary
            'lstm_state_output' = output lstm state, indexed by names in self.lstm_layer()
                                each element is (ht, ct)
                                shape of ht, ct is (1, batch size, hidden size)
        """
        device = list(self.parameters())[0].device
        dtype = list(self.parameters())[0].dtype
        module_dict = dict(self.named_modules())
        state_flat_batch = state_batch.view(-1, self.state_size)
        result_tensor = state_flat_batch = state_flat_batch.to(
                device = device, dtype = dtype)
        if (self.fc_function_before_lstm is not None):
            result_tensor = self.fc_function_before_lstm(result_tensor)
        lstm_state_output = {}
        for lstm_layer_id in self.lstm_layer():
            batch_size = lstm_state_batch[lstm_layer_id][0].size(1)
            seq_len = int(result_tensor.size(0) / batch_size)
            lstm_input = result_tensor.view(seq_len, batch_size,
                                            result_tensor.size(1))
            lstm_state_tuple = (
                    lstm_state_batch[lstm_layer_id][0].to(
                        device = device, dtype = dtype),
                    lstm_state_batch[lstm_layer_id][1].to(
                        device = device, dtype = dtype))
            lstm_output, state_t = module_dict[lstm_layer_id](
                        lstm_input, lstm_state_tuple)
            lstm_state_output[lstm_layer_id] = state_t
            result_tensor = lstm_output.view(seq_len * batch_size,
                    lstm_output.size(-1))
        if (self.fc_function_after_lstm is not None):
            result_tensor = self.fc_function_after_lstm(result_tensor)
        pi_logits = self.policy_branch(result_tensor)
        pi = self.policy_softmax(pi_logits)
        value = self.value_branch(result_tensor)
        return {'pi_logits': pi_logits, 'pi': pi, 'value': value,
                'lstm_state_output': lstm_state_output}

    def sample_action(self, state_input, lstm_state_input = None):
        """
        Sample Actions
        Args:
        state_input : numpy.ndarray
        state_input = input of current state,
                        shape (seq len * batch size) + state_shape or state_shape
        lstm_state_input : dictionary
        lstm_state_input = input lstm state batch, indexed by names in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are numpy.ndarray
                            shape of h0, c0 is (1, batch size, hidden size)
        Returns:
        Returns:
        action_info : dictionary
        action_info = information of sampled action
            "action_index" : numpy.ndarray or int
            "action_index" = batch of actions, shape [seq len * batch size]
            "lstm_state_output" : dictionary
            "lstm_state_output" = output lstm state, indexed by names in self.lstm_layer()
                                each element is (h0, c0)
                                h0, c0 are numpy.ndarray
                                shape of h0, c0 is (1, batch size, hidden size)
            "state_value" : numpy.ndarray or float
            "state_value" = value function of current state, shape [seq len * batch size]
            "pi" : numpy.ndarray or float
            "pi" = policy, shape [seq len * batch size, number of actions] or [number of actions]
        """
        if (isinstance(state_input, int) or
                isinstance(state_input, float)):
            assert self.state_ndim == 0
            state_batch = torch.Tensor([state_input])
        elif (state_input.ndim == self.state_ndim):
            state_batch = torch.from_numpy(
                    np.expand_dims(state_input, axis = 0))
        elif (state_input.ndim == 1 + self.state_ndim):
            state_batch = torch.from_numpy(state_input)
        else:
            raise NotImplementedError
        lstm_state_batch = {}
        for lstm_layer_id in self.lstm_layer():
            state = []
            for s in range(2):
                if (lstm_state_input[lstm_layer_id][s].ndim == 3):
                    state.append(torch.from_numpy(
                        lstm_state_input[
                            lstm_layer_id][s].astype(np.float)))
                elif (lstm_state_input[lstm_layer_id][s].ndim == 2):
                    # shape (1, hidden size) => (1, batch size, hidden size)
                    state.append(torch.from_numpy(
                        np.expand_dims(
                            lstm_state_input[
                                lstm_layer_id][s].astype(np.float),
                            axis = 1)))
                elif (lstm_state_input[lstm_layer_id][s].ndim == 1):
                    state.append(torch.from_numpy(
                        lstm_state_input[lstm_layer_id][s][
                            None, None, :].astype(np.float)))
                else:
                    raise NotImplementedError
            lstm_state_batch[lstm_layer_id] = tuple(state)
        with torch.no_grad():
            forward_result = self.forward(state_batch, lstm_state_batch)
            pi_tensor = forward_result['pi']
            value_tensor = forward_result['value']
            lstm_state_output_tensor = forward_result['lstm_state_output']
        lstm_state_output = {}
        for lstm_layer_id in self.lstm_layer():
            state = []
            for s in range(2):
                if (lstm_state_input[lstm_layer_id][s].ndim == 3):
                    state.append(
                        lstm_state_output_tensor[lstm_layer_id
                                        ][s].to('cpu').numpy())
                elif (lstm_state_input[lstm_layer_id][s].ndim == 2):
                    # shape (1, batch size, hidden size) => (1, hidden size)
                    state.append(
                        np.squeeze(
                            lstm_state_output_tensor[lstm_layer_id
                                ][s].to('cpu').numpy(), axis = 1))
                elif (lstm_state_input[lstm_layer_id][s].ndim == 1):
                    state.append(
                        np.squeeze(
                            lstm_state_output_tensor[lstm_layer_id
                                ][s].to('cpu').numpy(), axis = (0, 1)))
            lstm_state_output[lstm_layer_id] = tuple(state)
        pi = pi_tensor.to('cpu').numpy()
        value = np.squeeze(value_tensor.to('cpu').numpy(), axis = -1)
        action_index = []
        for i in range(state_batch.size(0)):
            action_index.append(np.random.choice(self.n_action, p = pi[i]))
        if (isinstance(state_input, int) or
                isinstance(state_input, float) or
                state_input.ndim == self.state_ndim):
            action_index = int(action_index[0])
            value = value[0]
            pi = pi[0]
        else:
            action_index = np.array(action_index, np.int)
        action_info = {'action_index': action_index,
                        'lstm_state_output': lstm_state_output,
                        'state_value': value,
                        'pi': pi}
        return action_info

    def clipped_loss(self, coeff_value, coeff_policy, coeff_entropy,
            policy_clip_range, value_clip_range, data_set,
            value_output = None,
            policy_output = None, policy_logits = None):
        """
        Clipped Loss
        Args:
        coeff_value : float
        coeff_value = coefficient for value loss
        coeff_policy : float
        coeff_policy = coefficient for policy gradient
        coeff_entropy : float
        coeff_entropy  = coefficient for entropy
        policy_clip_range : float
        policy_clip_range = clip range for policy loss
        value_clip_range : float
        value_clip_range = clip range for value loss
        data_set : dictionary
        data_set = sampled data
            'state' : numpy.ndarray
            'state' = batch of input state, shape [sequence size * batch size] + state_shape, when value_output, policy_output, policy_logits are not None, this item is ignored
            'lstm_state_input' : dictionary
            'lstm_state_input' = input batch of lstm state, when value_output, policy_output, policy_logits are not None, this item is ignored, indexed by name in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are numpy.ndarray
                            shape of h0, c0 is (1, batch size, hidden size)
            'action_index' : numpy.ndarray
            'action_index' = batch of indices for applied action, shape [sequence size * batch size]
            'target_value' : numpy.ndarray
            'target_value' = batch of target values, shape [sequence size * batch size]
            'advantage' : numpy.ndarray
            'advantage' = batch of advantage estimation, shape [sequence size * batch size]
            'old_value' : numpy.ndarray
            'old_value' = batch of values for old policy, shape [sequence size * batch size]
            'old_neg_logpac' : numpy.ndarray
            'old_neg_logpac' = -log(pi_old(a_t)), shape [sequence size * batch_size]
        value_output : torch.tensor or None
        value_output = value output by the model, when this is None, it is computed from data_set
        policy_output : torch.tensor or None
        policy_output = policy output by the model, when this is None, it is computed from data_set
        policy_logits : torch.tensor or None
        policy_logits = policy logtis output by the model
        Returns:
        loss : dictionary
        loss = a collection of computation result related to error
        """
        eps = 1e-8
        if (value_output is None or policy_output is None or
                                        policy_logits is None):
            state_batch = torch.from_numpy(data_set['state'])
            lstm_state_batch = {}
            for lstm_layer_id in self.lstm_layer():
                lstm_state_batch[lstm_layer_id] = [
                    torch.from_numpy(data_set['lstm_state_input'
                        ][lstm_layer_id][0].astype(np.float)),
                    torch.from_numpy(data_set['lstm_state_input'
                        ][lstm_layer_id][1].astype(np.float))]
            forward_result = self.forward(state_batch, lstm_state_batch)
            policy_logits = forward_result['pi_logits']
            policy_output = forward_result['pi']
            value_output = forward_result['value']
        device = policy_output.device
        loss = {}
        value_output = torch.squeeze(value_output, dim = 1)
        old_value = torch.from_numpy(data_set['old_value']).to(
                device = device, dtype = value_output.dtype)
        loss['value_clipped'] = old_value + torch.clamp(
                                    value_output - old_value,
                                    -value_clip_range, value_clip_range)
        target_value = torch.from_numpy(data_set['target_value']).to(
                device = device, dtype = value_output.dtype)
        loss['value_loss_unclipped'] = torch.pow(
                value_output - target_value, 2)
        loss['value_loss_clipped'] = torch.pow(
                loss['value_clipped'] - target_value, 2)
        loss['value_loss'] = 0.5 * torch.mean(
                torch.max(loss['value_loss_unclipped'],
                            loss['value_loss_clipped']))
        if (policy_logits is not None):
            action = torch.from_numpy(data_set['action_index']).to(device)
            loss['neg_logpac'] = nn.CrossEntropyLoss(
                                reduction = 'none')(policy_logits, action)
        else:
            action_one_hot = np.eye(
                    self.n_action)[data_set['action_index']]
            action_one_hot = torch.from_numpy(action_one_hot).to(device)
            logpac = torch.log(policy_output) * action_one_hot
            loss['neg_logpac'] = -toch.sum(logpac, dim = -1)
        old_neg_logpac = torch.from_numpy(
                data_set['old_neg_logpac']).to(
                        device = device, dtype = loss['neg_logpac'].dtype)
        loss['policy_ratio'] = torch.exp(old_neg_logpac -
                                        loss['neg_logpac'])
        advantage = torch.from_numpy(data_set['advantage']).to(
                device = device, dtype = loss['policy_ratio'].dtype)
        loss['neg_policy_loss_unclipped'
                ] = -advantage * loss['policy_ratio']
        loss['neg_policy_loss_clipped'] = -advantage * torch.clamp(
                loss['policy_ratio'],
                1. - policy_clip_range, 1. + policy_clip_range)
        loss['neg_policy_loss'] = torch.mean(torch.max(
                        loss['neg_policy_loss_unclipped'],
                        loss['neg_policy_loss_clipped']))
        loss['policy_entropy'] = -torch.mean(torch.sum(policy_output
            * torch.log(torch.max(policy_output,
                other = eps * torch.ones_like(policy_output))), dim = -1))
        loss['loss'] = coeff_value * loss['value_loss'] +\
                coeff_policy * loss['neg_policy_loss'] -\
                coeff_entropy * loss['policy_entropy']
        return loss

