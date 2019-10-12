import numpy as np

class RingArrayImpl(object):
    """
    An Implementation Of Ring Array Buffer
    """
    def __init__(self, capacity, item_shape, dtype = np.float):
        """
        Args:
        capacity : int
        capacity = capacity, max size of this buffer
        item_shape : list or tuple
        item_shape = shape of items in this buffer
        dtype : type
        dtype = data type of items in this buffer
        """
        self.capacity = capacity
        self.item_shape = tuple(item_shape)
        self.buffer = np.zeros(
                [self.capacity] + list(self.item_shape)).astype(dtype)
        self.reset()

    def get_item_shape(self):
        """
        Get Shape Of Items
        Returns:
        item_shape : tuple
        item_shape = shape of items in this buffer
        """
        return self.item_shape

    def reset(self):
        """
        Reset This Buffer
        """
        self.buffer[...] = 0
        self.tail = 0
        self.size = 0

    def get_size(self):
        """
        Get The Size Of This Buffer
        Returns:
        buffer_size : int
        buffer_size = number of items in this buffer
        """
        return int(self.size)

    def append(self, item_new):
        """
        Append New Items
        Args:
        item_new : numpy.ndarray or int or float
        item_new = new item, shape item_shape or [batch size] + item_shape
        """
        if (isinstance(item_new, int) or isinstance(item_new, float)):
            if (len(self.item_shape) == 0):
                item_new = np.array([item_new])
            else:
                raise NotImplementedError
        if (item_new.ndim == len(self.item_shape)):
            item_new = np.expand_dims(item_new, axis = 0)
        ind = np.arange(self.tail,
                self.tail + item_new.shape[0]) % self.capacity
        self.buffer[ind] = item_new
        self.tail = (self.tail + item_new.shape[0]) % self.capacity
        self.size = min(self.size + item_new.shape[0], self.capacity)

    def get_item(self, ind):
        """
        Get Item With Index
        Args:
        ind : numpy.ndarray or int
        ind = indicies of items
        Returns:
        item : numpy.ndarray
        item = obtained item, shape ind.shape + item_shape or item_shape
        """
        if (self.size < self.capacity):
            if (isinstance(ind, int)):
                assert ind < self.size, 'unavailable index'
            elif (isinstance(ind, np.ndarray)):
                assert np.sum(ind >= self.size) == 0, 'unavailable index'
            else:
                raise NotImplementedError
        ind = ind % self.capacity
        return self.buffer[ind]

