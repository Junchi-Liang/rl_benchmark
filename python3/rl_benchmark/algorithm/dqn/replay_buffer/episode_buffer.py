import numpy as np

class ReplayBuffer(object):
    """
    Replay Buffer For Episodes
    """
    def __init__(self, capacity):
        """
        Args:
        capacity : int
        capacity = capacity, max size of this buffer
        """
        self.capacity = capacity
        self.reset()

    def reset(self):
        """
        Reset This Buffer
        """
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.size = 0
        self.tail = 0

    def get_size(self):
        """
        Get The Size Of This Buffer
        Returns:
        buffer_size : int
        buffer_size = number of items in this buffer
        """
        return self.size

    def append(self, item_new):
        """
        Append New Item
        (insert an episode)
        Args:
        item_new : dictionary
        item_new = new item
            'state' : numpy.ndarray
            'state' = a sequence of states, shape [seq len] + state_size
            'action' : numpy.ndarray
            'action = a sequence of actions, shape [seq len] + action_shape
            'reward' : numpy.ndarray
            'reward' = a sequence of rewards, shape [seq len]
            'done' : numpy.ndarray
            'done' = a sequence of done flag, shape [seq len]
        """
        if (self.size < self.capacity):
            self.state_buffer.append(item_new['state'])
            self.action_buffer.append(item_new['action'])
            self.reward_buffer.append(item_new['reward'])
            self.done_buffer.append(item_new['done'])
            self.size += 1
        else:
            self.state_buffer[self.tail] = item_new['state']
            self.action_buffer[self.tail] = item_new['action']
            self.reward_buffer[self.tail] = item_new['reward']
            self.done_buffer[self.tail] = item_new['done']
        self.tail = (self.tail + 1) % self.capacity

    def sample_batch(self, batch_size, seq_len = 1):
        """
        Sample A Batch Of Sequences
        Args:
        batch_size : int
        batch_size = size of a batch
        seq_len : int
        seq_len = length of sequences
        Returns:
        data_batch : dictionary
        data_batch = a batch of data
            'state_current' : numpy.ndarray
            'state_current' = sampled current state sequence,
                shape [seq_len, batch_size] + state_shape
            'state_next' : numpy.ndarray
            'state_next' = sampled next state sequence,
                shape [seq_len, batch_size] + state_shape
            'action' : numpy.ndarray
            'action' = sampled action sequence,
                shape [seq_len, batch_size] + action_shape
            'reward' : numpy.ndarray
            'reward' = sampled reward sequence,
                shape [seq_len, batch_size]
            'done' : numpy.ndarray
            'done' = sampled done flag sequence,
                shape [seq_len, batch_size]
            'seq_len' : numpy.ndarray
            'seq_len' = sequence length of each sampled sequences,
                shape [batch_size]
        """
        ep_ind_chosen = np.random.randint(0, self.size, size = batch_size)
        state_current_list = [] # [batch_size][seq_len, state_shape]
        state_next_list = [] # [batch_size][seq_len, state_shape]
        action_list = [] # [batch_size][seq_len, action_shape]
        reward_list = [] # [batch_size][seq_len]
        done_list = [] # [batch_size][seq_len]
        seq_len_list = [] # [batch_size]
        for ep_ind in ep_ind_chosen:
            ep_length = self.state_buffer[ep_ind].shape[0]
            assert ep_length > 1
            step_ind = np.random.randint(ep_length - 1)
            state_current_chunk = self.state_buffer[ep_ind][
                    step_ind:(step_ind + seq_len)]
            seq_len_list.append(state_current_chunk.shape[0])
            if (state_current_chunk.shape[0] < seq_len):
                state_current_chunk = np.concatenate(
                        [state_current_chunk,
                        np.zeros(
                            [seq_len - state_current_chunk.shape[0]] +
                            list(state_current_chunk.shape[1:]))
                        ], axis = 0)
            state_next_chunk = self.state_buffer[ep_ind][
                    (step_ind + 1):(step_ind + seq_len + 1)]
            if (state_next_chunk.shape[0] < seq_len):
                state_next_chunk = np.concatenate(
                        [state_next_chunk,
                        np.zeros(
                            [seq_len - state_next_chunk.shape[0]] +
                            list(state_next_chunk.shape[1:]))
                        ], axis = 0)
            action_chunk = self.action_buffer[ep_ind][
                    step_ind:(step_ind + seq_len)]
            if (action_chunk.shape[0] < seq_len):
                action_chunk = np.concatenate(
                        [action_chunk,
                        np.zeros(
                            [seq_len - action_chunk.shape[0]] +
                            list(action_chunk.shape[1:]))
                        ], axis = 0)
            reward_chunk = self.reward_buffer[ep_ind][
                    step_ind:(step_ind + seq_len)]
            if (reward_chunk.shape[0] < seq_len):
                reward_chunk = np.concatenate(
                        [reward_chunk,
                        np.zeros(
                            [seq_len - reward_chunk.shape[0]] +
                            list(reward_chunk.shape[1:]))
                        ], axis = 0)
            done_chunk = self.done_buffer[ep_ind][
                    step_ind:(step_ind + seq_len)]
            if (done_chunk.shape[0] < seq_len):
                done_chunk = np.concatenate(
                        [done_chunk,
                        np.zeros(
                            [seq_len - done_chunk.shape[0]] +
                            list(done_chunk.shape[1:]))
                        ], axis = 0)
            state_current_list.append(state_current_chunk)
            state_next_list.append(state_next_chunk)
            action_list.append(action_chunk)
            reward_list.append(reward_chunk)
            done_list.append(done_chunk)
        state_current = np.stack(state_current_list, axis = 1)
        state_next = np.stack(state_next_list, axis = 1)
        action = np.stack(action_list, axis = 1)
        reward = np.stack(reward_list, axis = 1)
        done = np.stack(done_list, axis = 1)
        return {'state_current': state_current,
                'state_next': state_next, 'action': action,
                'reward': reward, 'done': done,
                'seq_len': np.array(seq_len_list)}

