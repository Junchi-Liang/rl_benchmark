import time

import tensorflow as tf
import numpy as np
from tqdm import trange, tqdm
from scipy.signal import lfilter
import cv2

class PPOWorker(object):
    """
    PPO Worker
    """
    def __init__(self, model, env_list, gamma = 0.99,
                    lambda_gae = 1., optimizer_type = None,
                    grad_clip_norm = 40, coeff_value = 0.5,
                    coeff_policy = 1., coeff_entropy = 0.01):
        """
        Args:
        model : PPOModel
        model = ppo model
        env_list : list of reinforcement_learning.env
        env_list = list of environments
        gamma : float
        gamma = discount factor
        lambda_gae : float
        lambda_gae = coefficient in generalized advantage estimation
        optimizer_type : string
        optimizer_type = optimizer type (e.g. 'Adam', 'RMSProp')
        grad_clip_norm : float
        grad_clip_norm = norm size for clipping gradients
        n_train_step : int
        n_train_step = number of steps for training
        coeff_value : float
        coeff_value = coefficient for value loss
        coeff_policy : float
        coeff_policy = coefficient for policy gradient
        coeff_entropy : float
        coeff_entropy  = coefficient for entropy
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
            self.learning_rate = tf.placeholder(tf.float32, [],
                                                'learning_rate')
            self.loss = self.model.build_loss(
                            coeff_value, coeff_policy, coeff_entropy,
                            self.model.ops['value'],
                            self.model.ops['pi'],
                            policy_logits =
                                self.model.ops['pi_preactivation'])
            self.optimizer, self.update_ops,\
                    self.var_optimizer = self.get_optimizer(
                            self.loss['loss'], optimizer_type,
                            grad_clip_norm, self.learning_rate)
            self.var_optimizer_init = tf.variables_initializer(
                                            self.var_optimizer)
        else:
            self.optimizer, self.update_ops = None, None
        for env in self.env_set:
            env.new_episode()
            time.sleep(0.5)

    def get_optimizer(self, loss, optimizer_type,
                        grad_clip_norm, learning_rate):
        """
        Get Optimizer
        Args:
        loss : tensorflow.python.framework.ops.Tensor
        loss = loss function to be minimized
        optimizer_type : string
        optimizer_type = optimizer type (e.g. 'Adam', 'RMSProp')
        grad_clip_norm : float
        grad_clip_norm = norm size for clipping gradients
        learning_rate : float or tensorflow.python.framework.ops.Tensor
        learning_rate = learning rate
        Returns:
        optimizer : tensorflow.python.training
        optimizer = optimizer
        update_ops : tensorflow.python.framework.ops.Tensor
        update_ops = operators for updating
        var_optimizer : list
        var_optimizer = collect of optimizer variables
        """
        if (optimizer_type.lower() == 'adam'):
            optimizer = tf.train.AdamOptimizer(
                                learning_rate = learning_rate)
            update_ops = self.model.build_update(
                                optimizer, grad_clip_norm, loss)
            var_optimizer = [v for v in
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    if v.name.find('Adam') >= 0 or
                       v.name.find('beta1_power') >= 0 or
                       v.name.find('beta2_power') >= 0]
        elif (optimizer_type.lower() == 'rmsprop'):
            optimizer = tf.train.RMSPropOptimizer(
                                learning_rate = learning_rate)
            update_ops = self.model.build_update(
                    optimizer, grad_clip_norm, loss)
            var_optimizer = [v for v in
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    if v.name.find('RMSProp') >= 0]
        else:
            raise NotImplementedError
        return optimizer, update_ops, var_optimizer

    def reset_optimizer(self, tf_sess):
        """
        Initialize Variables Of Optimizer
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for initialization
        """
        tf_sess.run(self.var_optimizer_init)

    def reset_lstm_state(self, ind_env = None, lstm_state = None):
        """
        Reset LSTM State
        Args:
        ind_env : int or None
        ind_env = index of environment where LSTM state should be clear, if this is None, clear LSTM states of all environments
        lstm_state : dictionary or None (will be replaced by self.train_lstm_state)
        lstm_state = lstm state to be reset, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
        """
        if (lstm_state is None):
            lstm_state = self.train_lstm_state
        for lstm_layer_id in self.model.lstm_layer():
            if (ind_env is None):
                lstm_state[lstm_layer_id][0][...] = 0
                lstm_state[lstm_layer_id][1][...] = 0
            else:
                lstm_state[lstm_layer_id][0][ind_env, ...] = 0
                lstm_state[lstm_layer_id][1][ind_env, ...] = 0

    def update_lstm_state(self, state_new,
                            ind_env = None, lstm_state = None):
        """
        Update LSTM State
        Args:
        state_new : dictionary
        state_new = new state value, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
        ind_env : int or None
        ind_env = index of environment where LSTM state should be clear, if this is None, clear LSTM states of all environments
        lstm_state : dictionary or None (will be replaced by self.train_lstm_state)
        lstm_state = lstm state to be updated, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
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
                lstm_state[lstm_layer_id][0][ind_env,
                        ...] = state_new[lstm_layer_id][0]
                lstm_state[lstm_layer_id][1][ind_env,
                        ...] = state_new[lstm_layer_id][1]

    def get_lstm_state(self, ind_env = None, copy = True,
                        lstm_state = None):
        """
        Get LSTM State
        Args:
        ind_env : int or None
        ind_env = index of environment where LSTM state should be clear, if this is None, clear LSTM states of all environments
        copy : bool
        copy = when this is True, return a copy
        lstm_state : dictionary or None (will be replaced by self.train_lstm_state)
        lstm_state = lstm state, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
        Returns:
        current_state : dictionary
        current_state = current state, indexed by names in self.lstm_layer()
                        each element is (c0, h0)
                        c0, h0 : numpy.ndarray
                        shape of c0, h0 is (batch size, hidden size)
        """
        if (lstm_state is None):
            lstm_state = self.train_lstm_state
        current_state = {}
        for lstm_layer_id in self.model.lstm_layer():
            if (ind_env is None):
                s = [lstm_state[lstm_layer_id][0],
                        lstm_state[lstm_layer_id][1]]
            else:
                s = [lstm_state[lstm_layer_id][0][ind_env, ...],
                    lstm_state[lstm_layer_id][1][ind_env, ...]]
            if (copy):
                s = [s[0].copy(), s[1].copy()]
            current_state[lstm_layer_id] = s
        return current_state

    def update_model(self, tf_sess, policy_clip_range, value_clip_range,
                    batch_size, unroll_step, data_batch, learning_rate):
        """
        Update Model With One Batch Of Data
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for computation
        policy_clip_range : float
        policy_clip_range = clip range for policy loss
        value_clip_range : float
        value_clip_range = clip range for value loss
        batch_size : int
        batch_size = batch size
        unroll_step : int
        unroll_step = number of steps in each batch
        data_batch : dictionary
        data_batch = a batch of data
                'img' : numpy.ndarray
                'img' = input images, shape [batch_size * unroll_step, h, w, c]
                'lstm_state_input' : dictionary
                'lstm_state_input' = collection of lstm state input,
                        indexed by names in self.lstm_layer()
                                each element is (c0, h0)
                                c0, h0 : numpy.ndarray
                                shape of c0, h0 is (batch size, hidden size) or (hidden size)
                'action' : numpy.ndarray
                'action' = selected actions, shape [batch_size * unroll_step]
                'target' : numpy.ndarray
                'target' = target values, shape [batch_size * unroll_step]
                'adv' : numpy.ndarray
                'adv' = advantage estimation, shape [batch_size * unroll_step]
                'old_value' : numpy.ndarray
                'old_value' = values of old policy, shape [batch_size * unroll_step]
                'old_neg_logpac' : numpy.ndarray
                'old_neg_logpac' = -log(pi_old(a_t)), shape [batch_size * unroll_step]
        learning_rate : float
        learning_rate = value of learning rate
        """
        feed_dict = self.model.prepare_feed_dict(data_batch['img'],
                data_batch['lstm_state_input'],
                batch_size, unroll_step, self.loss,
                data_batch['action'], data_batch['target'],
                data_batch['adv'],
                data_batch['old_neg_logpac'], data_batch['old_value'],
                policy_clip_range, value_clip_range)
        feed_dict[self.learning_rate] = learning_rate
        tf_sess.run(self.update_ops, feed_dict = feed_dict)

    def sample_trajectory(self, tf_sess, env_id_chosen,
                        schedule_local_time_step, num_repeat):
        """
        Sample Trajectories
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for computation
        env_id_chosen : list
        env_id_chosen = list of id for environments where we will sample
        schedule_local_time_step : int
        schedule_local_time_step = scheduled number of local time step, actual number can be smaller
        num_repeat : int
        num_repeat = number of repeated actions
        Returns:
        total_step : int
        total_step = total of steps sampled in this batch
        finished_episode : int
        finished_episode = number of finished episodes in this batch
        trajectory_set : dictionary
        trajectory_set = a set of trajectories
                'img' : numpy.ndarray
                'img' = input images, shape [batch_size * unroll_step, h, w, c]
                'lstm_state_input' : tuple or None
                'lstm_state_input' = collection of lstm state input,
                        indexed by names in self.lstm_layer()
                                each element is (c0, h0)
                                c0, h0 : numpy.ndarray
                                shape of c0, h0 is (batch size, hidden size) or (hidden size)
                'action' : numpy.ndarray
                'action' = selected actions, shape [batch_size * unroll_step]
                'target' : numpy.ndarray
                'target' = target values, shape [batch_size * unroll_step]
                'adv' : numpy.ndarray
                'adv' = advantage estimation, shape [batch_size * unroll_step]
                'old_value' : numpy.ndarray
                'old_value' = values of old policy, shape [batch_size * unroll_step]
                'old_neg_logpac' : numpy.ndarray
                'old_neg_logpac' = -log(pi_old(a_t)), shape [batch_size * unroll_step]
        """
        n_chosen = len(env_id_chosen)
        env_set_chosen = [self.env_set[env_id]
                            for env_id in env_id_chosen]
        img_batch = [[] for _ in range(n_chosen)]
        if (self.model.contain_lstm()):
            trajectory_set = {'lstm_state_input':
                    self.get_lstm_state(env_id_chosen, copy = True)}
        else:
            trajectory_set = {}
        action_batch = [[] for _ in range(n_chosen)]
        r_list = [[] for _ in range(n_chosen)]
        value_list = [[] for _ in range(n_chosen)]
        neg_logpac_list = [[] for _ in range(n_chosen)]
        total_step = 0
        finished_episode = 0
        ep_done = False
        # sample and apply actions
        for i_step in range(schedule_local_time_step):
            img_input = np.stack([env.get_state({'resolution':
                                    self.model.img_resolution[:2]})
                                for env in env_set_chosen], axis = 0)
            if (self.model.contain_lstm()):
                lstm_state_input = self.get_lstm_state(
                        env_id_chosen, copy = False)
            else:
                lstm_state_input = None
            # sample actions
            action_info = self.model.sample_action(
                    tf_sess, img_input, lstm_state_input)
            action_index = action_info['action_index']
            value = action_info['state_value']
            pi = action_info['pi']
            log_pi = np.log(pi)
            # update LSTM state if neg_logpac_list
            if (self.model.contain_lstm()):
                self.update_lstm_state(
                        action_info['lstm_state_output'], env_id_chosen)
            for i_chosen, env_chosen in enumerate(env_set_chosen):
                id_chosen = env_id_chosen[i_chosen]
                img_batch[i_chosen].append(img_input[i_chosen])
                action_batch[i_chosen].append(action_index[i_chosen])
                value_list[i_chosen].append(value[i_chosen])
                neg_logpac_list[i_chosen].append(
                            -log_pi[i_chosen][action_index[i_chosen]])
                # apply actions
                reward = env_chosen.apply_action(
                        env_chosen.action_set()[action_index[i_chosen]],
                        num_repeat)
                r_list[i_chosen].append(reward)
                if (env_chosen.episode_end()):
                    ep_done = True
            total_step += n_chosen
            if (ep_done):
                break
        # compute target and advantage
        target_batch = []
        adv_batch = []
        for i_chosen, env_chosen in enumerate(env_set_chosen):
            id_chosen = env_id_chosen[i_chosen]
            if (env_chosen.episode_end()):
                bootstrap = 0
                finished_episode += 1
                env_chosen.new_episode()
                self.reset_lstm_state(id_chosen)
            else:
                if (self.model.contain_lstm()):
                    bootstrap = self.model.sample_action(tf_sess,
                            env_chosen.get_state({'resolution':
                                self.model.img_resolution[:2]}),
                            self.get_lstm_state(id_chosen,
                                copy = False))['state_value']
                else:
                    bootstrap = self.model.sample_action(tf_sess,
                            env_chosen.get_state({'resolution':
                                self.model.img_resolution[:2]}),
                            None)['state_value']
            reward_plus_bootstrap = np.array(r_list[i_chosen] + [bootstrap])
            target_value = lfilter([1], [1, -self.gamma],
                                    reward_plus_bootstrap[::-1])[::-1]
            target_value = target_value[:-1]
            target_batch.append(target_value)
            value_plus_bootstrap = np.array(value_list[i_chosen] +
                                                            [bootstrap])
            td_error = reward_plus_bootstrap[:-1] +\
                            self.gamma * value_plus_bootstrap[1:] -\
                            value_plus_bootstrap[:-1]
            gae = lfilter([1], [1, -self.gamma * self.lambda_gae],
                                                td_error[::-1])[::-1]
            adv_batch.append(gae)
        # convert format
        img_batch = [np.stack(img_traj, axis = 0)
                            for img_traj in img_batch]
        img_batch = np.concatenate(img_batch, axis = 0)
        action_batch = [np.array(action_traj, np.int)
                                    for action_traj in action_batch]
        action_batch = np.concatenate(action_batch, axis = 0)
        target_batch = np.concatenate(target_batch, axis = 0)
        adv_batch = np.concatenate(adv_batch, axis = 0)
        old_value = np.concatenate(value_list, axis = 0)
        old_neg_logpac = np.concatenate(neg_logpac_list, axis = 0)
        trajectory_set['img'] = img_batch
        trajectory_set['action'] = action_batch
        trajectory_set['target'] = target_batch
        trajectory_set['adv'] = adv_batch
        trajectory_set['old_value'] = old_value
        trajectory_set['old_neg_logpac'] = old_neg_logpac
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

    def train_batch(self, tf_sess, max_local_time_step, max_total_step,
                        num_repeat, value_clip_range,
                        policy_clip_range, learning_rate_value,
                        num_epoch, minibatch_size):
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
        policy_clip_range : float
        policy_clip_range = clip range for policy loss
        value_clip_range : float
        value_clip_range = clip range for value loss
        learning_rate_value : float
        learning_rate_value = vlaue of learning rate
        num_epoch : int
        num_epoch = number of epochs
        minibatch_size : int
        minibatch_size = size of minibatch
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
                self.sample_trajectory(tf_sess, env_id_list,
                        schedule_local_time_step, num_repeat)
        local_step = int(total_step / len(env_id_list))
        for i_epoch in range(num_epoch):
            sample_cnt = 0
            if (self.model.contain_lstm()):
                shuffle = np.random.permutation(len(env_id_list))
            else:
                shuffle = np.random.permutation(total_step)
            while (sample_cnt < shuffle.size):
                next_sample_cnt = min(sample_cnt + minibatch_size,
                                                        shuffle.size)
                if (self.model.contain_lstm()):
                    ind_sample = [list(range(i_chosen * local_step,
                                        (1 + i_chosen) * local_step))
                                for i_chosen in
                                    shuffle[sample_cnt:next_sample_cnt]]
                    ind_sample = np.concatenate(ind_sample, axis = 0)
                else:
                    ind_sample = np.array(shuffle[sample_cnt:
                                                    next_sample_cnt])
                img_batch = trajectory_set['img'][ind_sample]
                action_batch = trajectory_set['action'][ind_sample]
                target_batch = trajectory_set['target'][ind_sample]
                adv_batch = trajectory_set['adv'][ind_sample]
                old_value_batch = trajectory_set['old_value'][ind_sample]
                old_neg_logpac_batch =\
                            trajectory_set['old_neg_logpac'][ind_sample]
                data_batch = {'img': img_batch, 'action': action_batch,
                                'target': target_batch, 'adv': adv_batch,
                                'old_value': old_value_batch,
                                'old_neg_logpac': old_neg_logpac_batch}
                if (self.model.contain_lstm()):
                    data_batch['lstm_state_input'] = self.get_lstm_state(
                            shuffle[sample_cnt:next_sample_cnt], False,
                            trajectory_set['lstm_state_input'])
                    self.update_model(tf_sess, policy_clip_range,
                                        value_clip_range,
                                        next_sample_cnt - sample_cnt,
                                        local_step, data_batch,
                                        learning_rate_value)
                else:
                    self.update_model(tf_sess, policy_clip_range,
                                        value_clip_range,
                                        next_sample_cnt - sample_cnt, 1,
                                        data_batch, learning_rate_value)
                sample_cnt = next_sample_cnt
        return total_step, finished_episode

    def test(self, tf_sess, env_test, test_episode, num_repeat,
                visualize = False,
                visualize_pause = None, visualize_size = [480, 640],
                progress_bar = True, verbose = False):
        """
        Test The Model
        Args:
        tf_sess : tensorflow.Session
        tf_sess = tensorflow session used for computation
        env_test : reinforcement_learning.env
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
            if (self.model.contain_lstm()):
                lstm_state_test = self.model.zero_lstm_state(batch_size = 1)
            else:
                lstm_state_test = None
            while (not env_test.episode_end()):
                action_info = self.model.sample_action(
                        tf_sess, env_test.get_state({'resolution':
                            self.model.img_resolution[:2]}),
                        lstm_state_test)
                action_index = action_info['action_index']
                lstm_state_test = action_info['lstm_state_output']
                for _ in range(num_repeat):
                    env_test.apply_action(
                            env_test.action_set()[action_index], 1)
                    if (env_test.episode_end()):
                        break
                    if (visualize):
                        img_out = cv2.resize(
                                env_test.get_state({'resolution':
                                    self.model.img_resolution[:2]}),
                                tuple(visualize_size[::-1]))
                        if (img_out.max() <= 1):
                            img_out *= 255
                        cv2.imshow('screen', img_out.astype(np.uint8))
                        cv2.waitKey(visualize_pause)
            score_list.append(env_test.episode_total_score())
            if (verbose):
                print(('Episode %d: ' % i_ep) + str(score_list[-1]))
        score_list = np.array(score_list)
        report['score_mean'] = np.mean(score_list)
        report['score_std'] = np.std(score_list)
        report['score_max'] = np.max(score_list)
        report['score_min'] = np.min(score_list)
        return report

