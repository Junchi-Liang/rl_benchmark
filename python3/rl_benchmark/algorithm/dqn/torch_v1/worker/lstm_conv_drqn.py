import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
import cv2

class DQNWorker(object):
    """
    DQN Worker
    (assuming episode buffer here for drqn)
    """
    def __init__(self, model, replay_buffer, optimizer_type = None):
        """
        Args:
        model : DQNModel
        model = the model to be updated
        replay_buffer : ReplayBuffer
        replay_buffer = replay buffer
        optimizer_type : string
        optimizer_type = optimizer type (e.g. 'Adam', 'RMSProp')
        """
        self.model = model
        self.replay_buffer = replay_buffer
        if (replay_buffer is not None):
            replay_buffer.reset()
        if (optimizer_type is not None):
            self.optimizer = self.get_optimizer(optimizer_type)
        else:
            self.optimizer = None
        self._reset()

    def _reset(self):
        """
        Reset
        """
        self.state_this_episode = []
        self.action_this_episode = []
        self.reward_this_episode = []
        self.done_this_episode = []
        self.train_lstm_state = self.model.zero_lstm_state(
                batch_size = 1)

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

    def update_target_model(self, target, source = None, tau = 1.0):
        """
        Update Target Model
        Args:
        target : DQNWorker
        target = target model
        source : same type as target
        source = source model
        tau : float
        tau = target <- tau * source + (1 - tau) * target
        """
        if (source is None):
            source = self.model
        for target_param, source_param in zip(
                target.parameters(), source.parameters()):
            target_param.data.copy_(
                    target_param.data * (1.0 - tau) +
                    source_param.data * tau)

    def update_model(self, data_batch, learning_rate,
            grad_clip_norm = None):
        """
        Update Model From One Batch
        Args:
        data_batch : dictionary
        data_batch = one batch of data
            'state_current' : numpy.ndarray
            'state_current' = sampled current state, shape [seq len * batch_size] + state_shape
            'action' : numpy.ndarray
            'action' = sampled action, shape [seq len * batch_size] + action_shape
            'target_q' : numpy.ndarray
            'target_q' = target q value, shape [seq len * batch size]
            'seq_len' : numpy.ndarray
            'seq_len' = sequence length of each sampled sequences, shape [batch_size]
        learning_rate : float
        learning_rate = value of learning rate
        grad_clip_norm : float or None
        grad_clip_norm = norm size for clipping gradients
        """
        loss = self.model.evaluate_loss({
            'target_q': data_batch['target_q'],
            'action': data_batch['action'],
            'img': data_batch['state_current'],
            'seq_len': data_batch['seq_len'],
            'lstm_state_input':
                self.model.zero_lstm_state(data_batch['seq_len'].size)},
            seq_len = int(data_batch['target_q'].shape[0] /
                data_batch['seq_len'].size))
        self.optimizer.zero_grad()
        self.optimizer.param_groups[0]['lr'] = learning_rate
        loss['loss'].backward()
        if (grad_clip_norm is not None):
            nn.utils.clip_grad_norm_(
                self.model.parameters(), grad_clip_norm)
        self.optimizer.step()

    def sample_trajectory(self, env, exploration_prob,
            num_step, num_repeat):
        """
        Sample Trajectories
        Args:
        env : rl_benchmark.env.discrete_action
        env = env used for sampling
        exploration_prob : float
        exploration_prob = probability of exploration
        num_step : int
        num_step = number of sampling steps
        num_repeat : int
        num_repeat = number of repeated actions
        Returns:
        total_step : int
        total_step = total of steps sampled
        finished_episode : int
        finished_episode = number of finished episodes in this sampling
        """
        total_step = 0
        finished_episode = 0
        for _ in range(num_step):
            if (env.episode_end()):
                env.new_episode()
                self._reset()
            state_current = env.get_state({'resolution':
                self.model.img_resolution})
            action = self.model.sample_action(state_current,
                    lstm_state_input = self.train_lstm_state,
                    exploration_prob = exploration_prob)
            r = env.apply_action(
                    env.action_set()[action['action_index']], num_repeat)
            self.state_this_episode.append(state_current)
            self.action_this_episode.append(action['action_index'])
            self.reward_this_episode.append(r)
            if (env.episode_end()):
                finished_episode += 1
                self.done_this_episode.append(1)
                self.replay_buffer.append({
                    'state': np.stack(self.state_this_episode, axis = 0),
                    'action': np.array(self.action_this_episode),
                    'reward': np.array(self.reward_this_episode),
                    'done': np.array(self.done_this_episode)
                    })
            else:
                self.done_this_episode.append(0)
                state_next = env.get_state({'resolution':
                    self.model.img_resolution})
                self.train_lstm_state = action['lstm_state_output']
            total_step += 1
        return total_step, finished_episode

    def sample_from_replay_buffer(self, batch_size, seq_len,
            target_model, discount_factor):
        """
        Sample One Data Batch From Replay Buffer
        Args:
        batch_size : int
        batch_size = size of a batch
        seq_len : int
        seq_len = length of sequences
        target : DQNWorker
        target = target model
        discount_factor : float
        discount_factor = discount factor
        Returns:
        data_batch : dictionary or None (when not enough data)
        data_batch = one batch of data
            'state_current' : numpy.ndarray
            'state_current' = sampled current state, shape [batch_size] + state_shape
            'action' : numpy.ndarray
            'action' = sampled action, shape [batch_size] + action_shape
            'target_q' : numpy.ndarray
            'target_q' = target q value, shape [batch size]
            'seq_len' : numpy.ndarray
            'seq_len' = sequence length of each sampled sequences, shape [batch_size]
        """
        if (self.replay_buffer.get_size() < 1):
            return None
        data_raw = self.replay_buffer.sample_batch(batch_size, seq_len)
        state_current = data_raw['state_current'].reshape([-1] +
                list(data_raw['state_current'].shape[2:]))
        state_next = data_raw['state_next'].reshape([-1] +
                list(data_raw['state_next'].shape[2:]))
        action = data_raw['action'].reshape([-1] +
                list(data_raw['action'].shape[2:]))
        reward = data_raw['reward'].reshape([-1] +
                list(data_raw['reward'].shape[2:]))
        done = data_raw['done'].reshape([-1] +
                list(data_raw['done'].shape[2:]))
        q_value_next = target_model.sample_action(
                state_next,
                lstm_state_input =
                self.model.zero_lstm_state(batch_size))['q_value']
        q_value_next = np.max(q_value_next, axis = 1)
        target_q = reward + discount_factor * (1 - done) * q_value_next
        return {'state_current': state_current,
                'action': action, 'target_q': target_q,
                'seq_len': data_raw['seq_len']}

    def test(self, env_test, test_episode, num_repeat,
                visualize = False,
                visualize_pause = None, visualize_size = [480, 640],
                progress_bar = True, verbose = False):
        """
        Test The Model
        Args:
        env_test : rl_benchmark.env
        env_test = environment for testing
        test_episode : int
        test_episode = number of test episodes
        num_repeat : int
        num_repeat = number of repeated actions
        visualize : bool
        visualize = visualize testing 
        visualize_pause : None or float
        visualize_pause = when this is None, screen is not shown, otherwise, each frame is shown for visualize_pause ms
        visualize_size : list or tuple
        visualize_size = size of visualization windows, [h, w]
        Returns:
        report : dictionary
        report = collection of statistics
        """
        report = {}
        score_list = []
        if (progress_bar):
            ep_range = trange(test_episode)
        else:
            ep_range = range(test_episode)
        for i_ep in ep_range:
            env_test.new_episode()
            test_lstm_state = self.model.zero_lstm_state(batch_size = 1)
            while (not env_test.episode_end()):
                action_info = self.model.sample_action(
                    env_test.get_state({'resolution':
                        self.model.img_resolution}),
                    lstm_state_input = test_lstm_state,
                    exploration_prob = 0)
                for _ in range(num_repeat):
                    env_test.apply_action(
                        env_test.action_set()[
                            action_info['action_index']], 1)
                    if (env_test.episode_end()):
                        break
                    if (visualize):
                        img_out = cv2.resize(env_test.get_state({
                            'resolution': self.model.img_resolution[:2]}),
                            tuple(visualize_size[::-1]))
                        cv2.imshow('screen',
                                (img_out[..., ::-1] * 255).astype(np.uint8))
                        cv2.waitKey(visualize_pause)
                test_lstm_state = action_info['lstm_state_output']
            score_list.append(env_test.episode_total_score())
            if (verbose):
                print(('Episode %d: ' % i_ep) + str(score_list[-1]))
        score_list = np.array(score_list)
        report['score_mean'] = np.mean(score_list)
        report['score_std'] = np.std(score_list)
        report['score_max'] = np.max(score_list)
        report['score_min'] = np.min(score_list)
        return report

