import numpy as np
from rl_benchmark.algorithm.dqn.replay_buffer.ring_array_impl import RingArrayImpl

class ReplayBuffer(object):
    """
    Basic Replay Buffer
    (sampling sequence data is only available when data is input consecutively)
    """
    def __init__(self, capacity, state_shape, action_shape,
            state_dtype = np.float, action_dtype = np.float,
            reward_dtype = np.float):
        """
        Args:
        capacity : int
        capacity = capacity, max size of this buffer
        state_shape : list or tuple
        state_shape = shape of states
        action_shape : list or tuple
        action_shape = shape of actions
        state_dtype : type
        state_dtype = data type of states
        action_dtype : type
        action_dtype = data type of actions
        reward_dtype : type
        reward_dtype = data type of rewards
        """
        self.capacity = capacity
        self.state_shape = tuple(state_shape)
        self.state_dtype = state_dtype
        self.action_shape = tuple(action_shape)
        self.action_dtype = action_dtype
        self.reward_dtype = reward_dtype
        self.state_current_buffer = RingArrayImpl(
                capacity, state_shape, state_dtype)
        self.state_next_buffer = RingArrayImpl(
                capacity, state_shape, state_dtype)
        self.action_buffer = RingArrayImpl(
                capacity, action_shape, action_dtype)
        self.reward_buffer = RingArrayImpl(capacity, [], reward_dtype)
        self.done_buffer = RingArrayImpl(capacity, [], np.int)
        self.reset()

    def reset(self):
        """
        Reset This Buffer
        """
        self.state_current_buffer.reset()
        self.state_next_buffer.reset()
        self.action_buffer.reset()
        self.reward_buffer.reset()
        self.done_buffer.reset()
        self.size = 0
        self.tail = 0

    def get_size(self):
        """
        Get The Size Of This Buffer
        Returns:
        buffer_size : int
        buffer_size = number of items in this buffer
        """
        return self.state_current_buffer.get_size()

    def append(self, item_new):
        """
        Append New Items
        Args:
        item_new : dictionary
        item_new = new item
            'state_current' : numpy.ndarray
            'state_current' = current state, shape [batch size] + state_shape or state_shape
            'state_next' : numpy.ndarray
            'state_next' = next state, shape [batch size] + state_shape or state_shape
            'action' : numpy.ndarray or int or float
            'action' = action, shpae [batch size] or ()
            'reward' : numpy.ndarray or int or float
            'reward' = reward, shape [batch size] or ()
            'done' : numpy.ndarray or int
            'done' = done flag, shpae [batch size] or ()
        """
        self.state_current_buffer.append(item_new['state_current'])
        self.state_next_buffer.append(item_new['state_next'])
        self.action_buffer.append(item_new['action'])
        self.reward_buffer.append(item_new['reward'])
        self.done_buffer.append(item_new['done'])
        self.size = self.state_current_buffer.get_size()
        self.tail = self.state_current_buffer.tail
        assert self.state_next_buffer.get_size() == self.size
        assert self.action_buffer.get_size() == self.size
        assert self.reward_buffer.get_size() == self.size
        assert self.done_buffer.get_size() == self.size
        assert self.tail == self.state_next_buffer.tail
        assert self.tail == self.action_buffer.tail
        assert self.tail == self.reward_buffer.tail
        assert self.tail == self.done_buffer.tail

    def get_item(self, ind):
        """
        Get Item With Index
        Args:
        ind : numpy.ndarray or int
        ind = indicies of items
        Returns:
        item : dictionary
        item = obtained item
            'state_current' : numpy.ndarray
            'state_current' = current state, shape ind.shape + state_shape or state_shape
            'state_next' : numpy.ndarray
            'state_next' = next state, shape ind.shape + state_shape or state_shape
            'action' : numpy.ndarray or int or float
            'action' = action, shape ind.shape + action_shape or action_shape
            'reward' : numpy.ndarray or int or float
            'reward' = reward, shape ind.shape or ()
            'done' : numpy.ndarray or int
            'done' = done flag, shape ind.shape or ()
        """
        state_current = self.state_current_buffer.get_item(ind)
        state_next = self.state_next_buffer.get_item(ind)
        action = self.action_buffer.get_item(ind)
        reward = self.reward_buffer.get_item(ind)
        done = self.done_buffer.get_item(ind)
        return {'state_current': state_current,
                'state_next': state_next,
                'action': action, 'reward': reward, 'done': done}

    def sample_batch(self, batch_size):
        """
        Sample A Batch
        Args:
        batch_size : int
        batch_size = size of a batch
        Returns:
        data_batch : dictionary
        data_batch = a batch of data
            'state_current' : numpy.ndarray
            'state_current' = sampled current state, shape [batch_size] + state_shape
            'state_next' : numpy.ndarray
            'state_next' = sampled next state, shape [batch_size] + state_shape
            'action' : numpy.ndarray
            'action' = sampled action, shape [batch_size] + action_shape
            'reward' : numpy.ndarray
            'reward' = sampled reward, shape [batch_size]
            'done' : numpy.ndarray
            'done' = sampled done flag, shpae [batch_size]
        """
        ind = np.random.randint(0, self.size, size = batch_size)
        return {'state_current': self.state_current_buffer.get_item(ind),
                'state_next': self.state_next_buffer.get_item(ind),
                'action': self.action_buffer.get_item(ind),
                'reward': self.reward_buffer.get_item(ind),
                'done': self.done_buffer.get_item(ind)}

