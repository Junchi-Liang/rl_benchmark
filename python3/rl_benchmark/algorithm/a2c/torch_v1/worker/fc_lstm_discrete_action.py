import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
from scipy.signal import lfilter
import cv2

class ActorCriticWorker(object):
    """
    A2C Worker
    """
    def __init__(self, model, env_list = [], gamma = 0.99,                       
                        lambda_gae = 1., optimizer_type = None):          
        """
        Args:
        model : ActorCriticModel
        model = actor-critic model
        env_list : list of rl_benchmark.env
        env_list = list of environments
        gamma : float
        gamma = discount factor
        lambda_gae : float
        lambda_gae = coefficient in generalized advantage estimation
        optimizer_type : string
        optimizer_type = optimizer type (e.g. 'Adam', 'RMSProp')
        """ 
        self.model = model                              
        self.env_set = env_list       
        self.n_env = len(self.env_set)
        if (self.model.contain_lstm()):
            self.train_lstm_state = model.zero_lstm_state(
                                    batch_size = self.n_env)
        else:
            self.train_lstm_state = None
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        if (optimizer_type is not None):
            self.optimizer = self.get_optimizer(optimizer_type)
        else:
            self.optimizer = None
        for env in self.env_set:
            env.new_episode()
            time.sleep(0.5)

    def get_optimizer(self, optimizer_type):
        """
        Get Optimizer
        Args:
        optimizer_type : string
        optimizer_type = optimizer type (e.g. 'Adam', 'RMSProp')
        Returns:
        optimizer : torch.optim
        optimizer = optimizer
        """
        if (optimizer_type.lower() == 'adam'):
            optimizer = optim.Adam(self.model.parameters())
        elif (optimizer_type.lower() == 'rmsprop'):
            optimizer = optim.RMSprop(self.model.parameters())
        else:
            raise NotImplementedError
        return optimizer

    def reset_lstm_state(self, ind_env = None, lstm_state = None):
        """
        Reset LSTM State Batch With Given Indices
        Args:
        ind_env : int or None
        ind_env = index of environment where LSTM state should be clear, if this is None, clear LSTM states of all environments
        lstm_state : dictionary
        lstm_state = lstm state to be reset, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        """
        if (lstm_state is None):
            lstm_state = self.train_lstm_state
        for lstm_layer_id in self.model.lstm_layer():
            if (ind_env is None):
                lstm_state[lstm_layer_id][0][...] = 0
                lstm_state[lstm_layer_id][1][...] = 0
            else:
                lstm_state[lstm_layer_id][0][:, ind_env, ...] = 0
                lstm_state[lstm_layer_id][1][:, ind_env, ...] = 0

    def update_lstm_state(self, state_new,
                            ind_env = None, lstm_state = None):
        """
        Update LSTM State Batch With Given Indices
        Args:
        state_new : dictionary
        state_new = new state value, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        ind_env : int or None
        ind_env = index of environment where LSTM state should be updated, if this is None, update LSTM states of all environments
        lstm_state : dictionary
        lstm_state = lstm state to be updated, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        """
        if (lstm_state is None):
            lstm_state = self.train_lstm_state
        for lstm_layer_id in self.model.lstm_layer():
            if (ind_env is None):
                lstm_state[lstm_layer_id][0][
                        ...] = state_new[lstm_layer_id][0]
                lstm_state[lstm_layer_id][1][
                        ...] = state_new[lstm_layer_id][1]
            else:
                lstm_state[lstm_layer_id][0][:, ind_env,
                        ...] = state_new[lstm_layer_id][0]
                lstm_state[lstm_layer_id][1][:, ind_env,
                        ...] = state_new[lstm_layer_id][1]

    def get_lstm_state(self, ind_env = None, copy = True,
                        lstm_state = None):
        """
        Get LSTM State Batch With Given Indices
        Args:
        ind_env : int or None
        ind_env = index of environment where LSTM state should be obtained, if this is None, return LSTM states of all environments
        copy : bool
        copy = when this is True, return a copy
        lstm_state : dictionary
        lstm_state = lstm state, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        Returns:
        current_state : dictionary
        current_state = current state, indexed by names in self.lstm_layer()
                        each element is (h0, c0)
                        shape of h0, c0 is (1, batch size, hidden size)
        """
        if (lstm_state is None):
            lstm_state = self.train_lstm_state
        current_state = {}
        for lstm_layer_id in self.model.lstm_layer():
            if (ind_env is None):
                s = [lstm_state[lstm_layer_id][0],
                        lstm_state[lstm_layer_id][1]]
            else:
                s = [lstm_state[lstm_layer_id][0][:, ind_env, ...],
                    lstm_state[lstm_layer_id][1][:, ind_env, ...]]
            if (copy):
                s = [s[0].copy(), s[1].copy()]
            current_state[lstm_layer_id] = s
        return current_state

    def update_model(self, data_batch,
            coeff_value, coeff_policy, coeff_entropy,
            learning_rate, grad_clip_norm = None):
        """
        Update Model With One Batch Of Data
        Args:
        data_batch : dictionary
        data_batch = one batch of data
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
        coeff_value : float
        coeff_value = coefficient for value loss
        coeff_policy : float
        coeff_policy = coefficient for policy gradient
        coeff_entropy : float
        coeff_entropy  = coefficient for entropy
        learning_rate : float
        learning_rate = value of learning rate
        grad_clip_norm : float or None
        grad_clip_norm = norm size for clipping gradients
        """
        self.model.train()
        loss = self.model.evaluate_loss(
                coeff_value, coeff_policy, coeff_entropy, data_batch)
        self.optimizer.zero_grad()
        self.optimizer.param_groups[0]['lr'] = learning_rate
        loss['loss'].backward()
        if (grad_clip_norm is not None):
            nn.utils.clip_grad_norm_(
                self.model.parameters(), grad_clip_norm)
        self.optimizer.step()

    def sample_trajectory(self, env_id_chosen,
                        schedule_local_time_step, num_repeat,
                        get_state_kwargs = {}):
        """
        Sample Trajectories
        Args:
        env_id_chosen : list
        env_id_chosen = list of id for environments where we will sample
        schedule_local_time_step : int
        schedule_local_time_step = scheduled number of local time step, actual number can be smaller
        num_repeat : int
        num_repeat = number of repeated actions
        get_state_kwargs : dictionary
        get_state_kwargs = arguments for get_state
        Returns:
        total_step : int
        total_step = total of steps sampled in this batch
        finished_episode : int
        finished_episode = number of finished episodes in this batch
        trajectory_set : dictionary
        trajectory_set = a set of trajectories
            'state' : numpy.ndarray
            'state' = input states, shape [seq len * batch_size] + state_shape
            'lstm_state_input' : dictionary
            'lstm_state_input' = input batch of lstm state, when value_output, policy_output, policy_logits are not None, this item is ignored, indexed by name in self.lstm_layer()
                            each element is (h0, c0)
                            h0, c0 are numpy.ndarray
                            shape of h0, c0 is (1, batch size, hidden size)
            'action_index' : numpy.ndarray
            'action_index' = selected action indicies, shape [seq len * batch_size]
            'target_value' : numpy.ndarray
            'target_value' = target values, shape [seq len * batch_size]
            'advantage' : numpy.ndarray
            'advantage' = advantage estimation, shape [seq len * batch_size]
        """
        self.model.eval()
        if (get_state_kwargs is None):
            get_state_kwargs = {}
        n_chosen = len(env_id_chosen)
        env_set_chosen = [self.env_set[env_id]
                            for env_id in env_id_chosen]
        if (self.model.contain_lstm()):
            trajectory_set = {
                    'lstm_state_input':
                        self.get_lstm_state(env_id_chosen, True)
                    }
        else:
            trajectory_set = {}
        state_batch = [] # [sequence size][batch size] + state_shape
        action_batch = [] # [sequence size][batch size]
        reward_list = [] # [sequence size][batch size]
        value_batch = [] # [sequence size][batch size]
        total_step = 0
        finished_episode = 0
        ep_done = False
        for i_step in range(schedule_local_time_step):
            state_input_step = np.stack([env.get_state(**get_state_kwargs)
                                for env in env_set_chosen], axis = 0)
            if (self.model.contain_lstm()):
                lstm_state_input_step = self.get_lstm_state(env_id_chosen)
            else:
                lstm_state_input_step = None
            # sample action
            action_info = self.model.sample_action(
                    state_input_step, lstm_state_input_step)
            state_batch.append(state_input_step)
            action_batch.append(action_info['action_index'])
            value_batch.append(action_info['state_value'])
            # update lstm state if necessary
            if (self.model.contain_lstm()): 
                self.update_lstm_state(
                    action_info['lstm_state_output'], env_id_chosen)
            # apply action 
            reward_step = []
            for i_chosen, env_chosen in enumerate(env_set_chosen):
                reward = env_chosen.apply_action(
                    env_chosen.action_set()[
                        action_info['action_index'][i_chosen]],
                    num_repeat)
                reward_step.append(reward)
                if (env_chosen.episode_end()):
                    ep_done = True
            reward_list.append(np.array(reward_step))
            total_step += n_chosen
            if (ep_done):
                break
        # compute target and advantage
        reward_list = np.stack(reward_list, axis = 0) # [sequence size, batch size]
        value_batch = np.stack(value_batch, axis = 0) # [sequence size, batch size]
        target_batch = [] # [batch size][sequence size]
        adv_batch = [] # [batch size][sequence size]
        for i_chosen, env_chosen in enumerate(env_set_chosen):
            id_chosen = env_id_chosen[i_chosen]
            if (env_chosen.episode_end()):
                bootstrap = 0
                finished_episode += 1
                env_chosen.new_episode()
                self.reset_lstm_state(id_chosen)
            else:
                if (self.model.contain_lstm()):
                    bootstrap = self.model.sample_action(
                            env_chosen.get_state(**get_state_kwargs),
                            self.get_lstm_state(id_chosen)
                            )['state_value']
                else:
                    bootstrap = self.model.sample_action(
                            env_chosen.get_state(**get_state_kwargs),
                            None)['state_value']
            reward_plus_bootstrap = np.concatenate([
                reward_list[:, i_chosen], [bootstrap]], axis = 0)
            target_value = lfilter([1], [1, -self.gamma],
                                    reward_plus_bootstrap[::-1])[::-1]
            target_value = target_value[:-1]
            target_batch.append(target_value)
            value_plus_bootstrap = np.concatenate([
                value_batch[:, i_chosen], [bootstrap]], axis = 0)
            td_error = reward_plus_bootstrap[:-1] +\
                            self.gamma * value_plus_bootstrap[1:] -\
                            value_plus_bootstrap[:-1]
            gae = lfilter([1], [1, -self.gamma * self.lambda_gae],
                                                td_error[::-1])[::-1]
            adv_batch.append(gae)
        target_batch = np.stack(target_batch, axis = 1) # [sequence size, batch size]
        adv_batch = np.stack(adv_batch, axis = 1) # [sequence size, batch size]
        # return
        state_batch = np.stack(state_batch, axis = 0) # [sequence size, batch size] + state_shape
        trajectory_set['state'] = state_batch.reshape([-1] +
                            list(self.model.state_shape))
        trajectory_set['action_index'] = np.concatenate(
                                    action_batch, axis = 0)
        trajectory_set['target_value'] = target_batch.reshape(-1)
        trajectory_set['advantage'] = adv_batch.reshape(-1)
        return total_step, finished_episode, trajectory_set

    def schedule_sampling(self, max_local_time_step, remaining_time_step):
        """
        Schedule A Sampling For One Batch From Remaining Time Steps
        Args:
        max_local_time_step : int
        max_local_time_step = max time step for each environment
        remaining_time_step : int
        remaining_time_step = remaining time steps
        Returns:
        env_id_list : list
        env_id_list = list of id for environments where we will sample
        local_time_step : int
        local_time_step = scheduled time steps for each environment
        """
        n_chosen = int(remaining_time_step / max_local_time_step)
        n_chosen = min(n_chosen, self.n_env)
        n_chosen = max(n_chosen, 1)
        env_id_list = np.random.permutation(self.n_env)[:n_chosen]
        local_time_step = min(max_local_time_step, remaining_time_step)
        return env_id_list, local_time_step

    def train_batch(self, max_local_time_step, max_total_step, num_repeat,
                        coeff_value, coeff_policy, coeff_entropy,
                        learning_rate, grad_clip_norm = None,
                        get_state_kwargs = {}):
        """
        Sample One Batch And Update The Model From It
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for computation
        max_local_time_step : int
        max_local_time_step = max time step for each environment
        max_total_step : int
        max_total_step = max total step
        num_repeat : int
        num_repeat = number of repeated actions
        coeff_value : float
        coeff_value = coefficient for value loss
        coeff_policy : float
        coeff_policy = coefficient for policy gradient
        coeff_entropy : float
        coeff_entropy  = coefficient for entropy
        learning_rate : float
        learning_rate = learning rate
        grad_clip_norm : float or None
        grad_clip_norm = norm size for clipping gradients
        get_state_kwargs : dictionary
        get_state_kwargs = arguments for get_state
        Returns:
        total_step : int
        total_step = total of steps sampled in this batch
        finished_episode : int
        finished_episode = number of finished episodes in this batch
        """
        if (max_total_step <= 0):
            return 0, 0
        env_id_list, schedule_local_time_step = self.schedule_sampling(
                                    max_local_time_step, max_total_step)
        total_step, finished_episode, trajectory_set =\
                                self.sample_trajectory(
                                    env_id_list, schedule_local_time_step,
                                    num_repeat, get_state_kwargs)
        self.update_model(trajectory_set,
                coeff_value, coeff_policy, coeff_entropy,
                learning_rate, grad_clip_norm)
        return total_step, finished_episode

    def test(self, env_test, test_episode, num_repeat,
                progress_bar = True, verbose = False,
                get_state_kwargs = {}):
        """
        Test The Model
        Args:
        env_test : reinforcement_learning.env
        env_test = environment for testing
        test_episode : int
        test_episode = number of test episodes
        num_repeat : int
        num_repeat = number of repeated actions
        get_state_kwargs : dictionary
        get_state_kwargs = arguments for get_state
        Returns:
        report : dictionary
        report = collection of statistics
        """
        self.model.eval()
        if (get_state_kwargs is None):
            get_state_kwargs = {}
        report = {}
        score_list = []
        if (progress_bar):
            ep_range = trange(test_episode)
        else:
            ep_range = range(test_episode)
        for i_ep in ep_range:
            env_test.new_episode()
            if (self.model.contain_lstm()):
                lstm_state_test = self.model.zero_lstm_state(
                                                batch_size = 1)
            else:
                lstm_state_test = None
            while (not env_test.episode_end()):
                action_info = self.model.sample_action(
                        env_test.get_state(**get_state_kwargs),
                        lstm_state_test)
                action_index = action_info['action_index']
                if (self.model.contain_lstm()):
                    self.update_lstm_state(
                            action_info['lstm_state_output'],
                            lstm_state = lstm_state_test)
                for _ in range(num_repeat):
                    env_test.apply_action(
                        env_test.action_set()[action_index], 1)
                    if (env_test.episode_end()):
                        break
            score_list.append(env_test.episode_total_score())
            if (verbose):
                print(('Episode %d: ' % i_ep) + str(score_list[-1]))
        score_list = np.array(score_list)
        report['score_mean'] = np.mean(score_list)
        report['score_std'] = np.std(score_list)
        report['score_max'] = np.max(score_list)
        report['score_min'] = np.min(score_list)
        return report


