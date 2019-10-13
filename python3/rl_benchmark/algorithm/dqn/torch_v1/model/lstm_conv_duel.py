import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import jason, json

from rl_benchmark.misc_utils.torch_v1.conv_helper import conv_output_size
from rl_benchmark.misc_utils.torch_v1.helper import Flatten, convert_to_tuple, get_activation, calculate_gain_from_activation

class DQNModel(nn.Module):
    """
    DQN Model
    """
    def __init__(self, config_file_path = None,
                 img_resolution = None, n_action = None, lstm_h_size = None,
                 conv_arch = [[8, 32, 4, 0, True, 'relu'],
                              [4, 64, 2, 0, True, 'relu'],
                              [3, 64, 1, 0, True, 'relu']],
                 fc_before_lstm = [[512, True, 'relu']]):
        """
        Args:
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
        super(DQNModel, self).__init__()
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
        self.build()

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
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        """
        assert self.contain_lstm()
        zero_state = {
                'lstm': (
                    np.zeros([1, batch_size, self.lstm_h_size],
                            dtype = dtype),
                    np.zeros([1, batch_size, self.lstm_h_size],
                            dtype = dtype)
                )}
        return zero_state

    def build(self):
        """
        Construct Submodule And Prepare Parameters
        """
        self.lstm_list = []
        submodule = OrderedDict()
        h, w, last_channel = self.img_resolution
        # convolution layers
        for i_conv, conv_config in enumerate(self.conv_arch):
            kernel_size, output_channel, stride,\
                    padding, add_bias, activation = conv_config
            padding = convert_to_tuple(padding, int)
            kernel_size = convert_to_tuple(kernel_size, int)
            stride = convert_to_tuple(stride, int)
            submodule['conv%d' % (i_conv + 1)] = nn.Conv2d(last_channel,
                                    output_channel, kernel_size,
                                    stride, padding, bias = add_bias)
            nn.init.xavier_uniform_(
                    submodule['conv%d' % (i_conv + 1)].weight,
                    calculate_gain_from_activation(activation))
            if (add_bias):
                nn.init.constant_(
                        submodule['conv%d' % (i_conv + 1)].bias, 0)
            if (activation is not None):
                activation_type, activation_module = get_activation(
                                                            activation)
                submodule[activation_type +
                            str(i_conv + 1)] = activation_module
            h = conv_output_size(h, padding[0], kernel_size[0], stride[0])
            w = conv_output_size(w, padding[1], kernel_size[1], stride[1])
            last_channel = output_channel
        # flatten features
        submodule['conv_flat'] = Flatten()
        last_size = h * w * last_channel
        # fully connected layers
        for i_fc, fc_config in enumerate(self.fc_before_lstm):
            num_hidden_unit, add_bias, activation = fc_config
            submodule['fc%d' % i_fc] = nn.Linear(
                        last_size, num_hidden_unit, bias = add_bias)
            nn.init.xavier_uniform_(submodule['fc%d' % i_fc].weight,
                    calculate_gain_from_activation(activation))
            if (add_bias):
                nn.init.constant_(submodule['fc%d' % i_fc].bias, 0)
            if (activation is not None):
                activation_type, activation_module = get_activation(
                                                            activation)
                submodule[activation_type +
                        str(len(self.conv_arch) + i_fc + 1)
                        ] = activation_module
            last_size = num_hidden_unit
        self.encoder = nn.Sequential(submodule)
        # LSTM
        if (self.contain_lstm()):
            module_id = 'lstm'
            self.lstm_list.append(module_id)
            self.__setattr__(module_id, nn.LSTM(last_size, self.lstm_h_size))
            nn.init.xavier_uniform_(self.__getattr__(module_id).weight_hh_l0)
            nn.init.xavier_uniform_(self.__getattr__(module_id).weight_ih_l0)
            nn.init.constant_(self.__getattr__(module_id).bias_ih_l0, 0)
            nn.init.constant_(self.__getattr__(module_id).bias_hh_l0, 0)
            last_size = self.lstm_h_size
        # output q value
        self.advantage_branch_size = int(last_size / 2)
        self.value_branch_size = last_size - self.advantage_branch_size
        self.function_to_advantage = nn.Linear(
                self.advantage_branch_size, self.n_action, bias = True)
        nn.init.xavier_uniform_(self.function_to_advantage.weight)
        nn.init.constant_(self.function_to_advantage.bias, 0)
        self.function_to_value = nn.Linear(
                self.value_branch_size, 1, bias = True)
        nn.init.xavier_uniform_(self.function_to_value.weight)
        nn.init.constant_(self.function_to_value.bias, 0)

    def forward(self, img_batch, lstm_state_batch = None):
        """
        Forward The Architecture
        Args:
        img_batch : torch.tensor
        img_batch = batch input image, shape (batch size or sequence size * batch size, c, h, w)
        lstm_state_batch : dictionary
        lstm_state_batch = batch input lstm state, indexed by names in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are torch.tensor
                            shape of h0, c0 is (1, batch size, hidden size)
        Returns:
        forward_result : dictionary
        forward_result = collection of results
            'q_value' : torch.Tensor
            'q_value' = q value, shape (seq len * batch size, n_action) or (batch size, n_action)
            'lstm_state_output' : dictionary
            'lstm_state_output' = output lstm state, indexed by names in self.lstm_layer()
                                each element is (ht, ct)
                                shape of ht, ct is (1, batch size, hidden size)
        """
        device = list(self.parameters())[0].device
        dtype = list(self.parameters())[0].dtype
        img_batch = img_batch.to(device = device, dtype = dtype)
        encode_batch = self.encoder(img_batch)
        module_dict = dict(self.named_modules())
        lstm_state_output = {}
        for lstm_layer_id in self.lstm_layer():
            batch_size = lstm_state_batch[lstm_layer_id][0].size(1)
            seq_len = int(encode_batch.size(0) / batch_size)
            lstm_input = encode_batch.view(seq_len, batch_size,
                                            encode_batch.size(1))
            lstm_state_tuple = (
                    lstm_state_batch[lstm_layer_id][0].to(
                        device = device, dtype = dtype),
                    lstm_state_batch[lstm_layer_id][1].to(
                        device = device, dtype = dtype))
            lstm_output, state_t = module_dict[lstm_layer_id](
                        lstm_input, lstm_state_tuple)
            lstm_state_output[lstm_layer_id] = state_t
            encode_batch = lstm_output.view(seq_len * batch_size,
                                                lstm_output.size(-1))
        advantage_branch, value_branch = torch.chunk(
                encode_batch, 2, dim = 1)
        assert advantage_branch.size(1) == self.advantage_branch_size
        assert value_branch.size(1) == self.value_branch_size
        value = self.function_to_value(value_branch)
        advantage = self.function_to_advantage(advantage_branch)
        q_value = value + (advantage -
                torch.mean(advantage, dim = 1, keepdim = True))
        q_value = torch.squeeze(q_value, dim = 1)
        return {'q_value': q_value, 'lstm_state_output': lstm_state_output}

    def evaluate_loss(self, data_set,
            seq_len = None, available_seq_len = None):
        """
        Evaluate Loss
        Args:
        data_set : dictionary
        data_set = sampled data
            'target_q' : numpy.ndarray
            'target_q' = target q value batch, shape (seq len * batch size or batch size)
            'action' : numpy.ndarray
            'action' = applied action batch, shape (seq len * batch size or batch size)
            'img' : numpy.ndarray
            'img' = batch input image, shape (batch size or sequence size * batch size, c, h, w)
            'lstm_state_input' : dictionary
            'lstm_state_input' = batch input lstm state, indexed by names in self.lstm_layer()
                                each element is (h0, c0)
                                h0, c0 are numpy.ndarray
                                shape of h0, c0 is (1, batch size, hidden size)
        seq_len : int
        seq_len = length of sequence, when only part of a sequence is used in loss, this should be provided
        available_seq_len : int
        available_seq_len = when only part of a sequence is used in loss, this is provided as the length of available part
        Returns:
        loss : dictionary
        loss = collection of losses
            'loss_whole_seq' : torch.Tensor
            'loss_whole_seq' = loss over whole sequence, shape ()
            'loss_truncated' : torch.Tensor
            'loss_truncated' = loss over available part of sequences, shape (), returned only when seq_len and available_seq_len is provided
            'loss' : torch.Tensor
            'loss' = loss to be optimized, shape ()
        """
        img_batch = torch.from_numpy(
                np.transpose(data_set['img'].astype(np.float),
                    [0, 3, 1, 2]))
        lstm_state_batch = {}
        for lstm_layer_id in self.lstm_layer():
            lstm_state_batch[lstm_layer_id] = [
                torch.from_numpy(data_set['lstm_state_input'
                    ][lstm_layer_id][0].astype(np.float)),
                torch.from_numpy(data_set['lstm_state_input'
                    ][lstm_layer_id][1].astype(np.float))]
        forward_result = self.forward(img_batch, lstm_state_batch)
        q_value = forward_result['q_value']
        action_batch = torch.from_numpy(data_set['action'])
        action_one_hot_batch = nn.functional.one_hot(
                action_batch.to(dtype = torch.int64),
                num_classes = self.n_action).to(
                        device = q_value.device, dtype = q_value.dtype)
        q_chosen = torch.sum(q_value * action_one_hot_batch, dim = 1)
        target_q_batch = torch.from_numpy(data_set['target_q']).to(
                device = q_chosen.device, dtype = q_chosen.dtype)
        mse_loss = nn.MSELoss(reduction = 'none')(
                q_chosen, target_q_batch)
        loss_whole_seq = torch.mean(mse_loss)
        if ((seq_len is not None) and (available_seq_len is not None)):
            batch_size = int(mse_loss.size(0) / seq_len)
            assert mse_loss.size(0) == batch_size * seq_len
            mse_loss_seq = mse_loss.view(seq_len, batch_size)
            mask_seq = torch.zeros(seq_len, batch_size)
            mask_seq[(seq_len - available_seq_len):] = 1.
            mask_seq = mask_seq.to(dtype = mse_loss.dtype,
                    device = mse_loss.device)
            loss_truncated = torch.mean(mse_loss_seq * mask_seq)
            return {'loss_whole_seq': loss_whole_seq,
                    'loss_truncated': loss_truncated,
                    'loss': loss_truncated}
        else:
            return {'loss_whole_seq': loss_whole_seq,
                    'loss': loss_whole_seq}

    def sample_action(self, img_input,
            lstm_state_input = None, exploration_prob = 0.):
        """
        Sample Action
        Args:
        img_input : numpy.ndarray
        img_input = input images, shape (seq len * batch size, h, w, c) or (h, w, c)
        lstm_state_input : dictionary or None
        lstm_state_input = input lstm state, indexed by names in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are numpy.ndarray
                            shape of h0, c0 is (1, batch size, hidden size)
        exploration_prob : float
        exploration_prob = probability of exploration (random action is returned)
        Returns:
        result : dictionary
        result = collection of sampling results
            "action_index" : numpy.ndarray or int
            "action_index" = batch of actions, shape [seq len * batch size]
            "lstm_state_output" : dictionary
            "lstm_state_output" = output lstm state, indexed by names in self.lstm_layer()
                                each element is (h0, c0)
                                h0, c0 are numpy.ndarray
                                shape of h0, c0 is (1, batch size, hidden size)
            "q_value" : numpy.ndarray or float
            "q_value" = q value, shape [seq len * batch size, n_action]
        """
        if (img_input.ndim == 3):
            img_batch = np.expand_dims(img_input, axis = 0)
        elif (img_input.ndim == 4):
            img_batch = img_input.astype(np.float)
        else:
            raise NotImplementedError
        img_batch = torch.from_numpy(
            np.transpose(img_batch, [0, 3, 1, 2]))
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
            forward_result = self.forward(img_batch, lstm_state_batch)
        q_value = forward_result['q_value'].to(device = 'cpu').numpy()
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
        p_sampled = np.random.rand(q_value.shape[0])
        action_random = np.random.randint(self.n_action,
                size = [q_value.shape[0]])
        action_output = np.argmax(q_value, axis = 1)
        action_output[
            p_sampled <= exploration_prob] = action_random[
                    p_sampled <= exploration_prob]
        if (img_input.ndim == 3):
            q_value = q_value[0]
            action_output = action_output[0]
        return {'q_value': q_value, 'lstm_state_output': lstm_state_output,
                'action_index': action_output}

