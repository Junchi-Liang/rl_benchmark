from mlt_pkg.reinforcement_learning.env.abstract_env import AbstractEnvironment
from skimage.transform import resize
import numpy as np
from skimage.color import rgb2grey
import retro
from mlt_pkg.reinforcement_learning.env.gym_retro.reward_helper import reward_reshape

class GymRetroEnvironment(AbstractEnvironment):
    """
    Standard Gym Retro Environment
    """
    def __init__(self, game_name, level_name, task_name, state_size = None):
        """
        Args:
        game_name : string
        game_name = name of the game
        level_name : string
        level_name = name of the level
        task_name : string
        task_name = name of the task
        state_size : list or tuple or None
        state_size = default resolution of states
        """
        self.game_name = game_name
        self.level_name = level_name
        self.task_name = task_name
        self.state_size = state_size
        self.env = retro.make(game = game_name, state = level_name)
        self.new_episode()
        self.n_button = self.env.action_space.sample().size
        self.actions = np.concatenate([np.identity(self.n_button, dtype = bool),
                                    np.zeros([1, self.n_button], dtype = bool)],
                                    axis = 0).tolist()

    def get_state(self, setting = None):
        """
        Get Current State
        Args:
        setting : dictionary or None
        setting = state setting
            'resolution' : list or tuple
            'resolution' = resolution of states, [h, w, c] or [h, w]
        Returns:
        state : numpy.ndarray
        state = current screen, shape [h, w, c], values locate at [0, 1]
        """
        if (setting is None or
                ('resolution' not in setting.keys())):
            resolution = self.state_size
        else:
            resolution = setting['resolution']
        normalized = False
        if (len(resolution) == 3 and resolution[2] == 1):
            state = rgb2grey(self.ob)
            normalized = True
        else:
            state = self.ob
        if (state.ndim == 2):
            state = np.expand_dims(state, axis = -1)
        assert (state.ndim == 3), 'shape of screen should be [h, w, c]'
        state = resize(state, resolution[:2], preserve_range = True)
        state = state.astype(np.float)
        if (not normalized):
            state /= 255.
        return state

    def apply_action(self, action, num_repeat):
        """
        Apply Actions To The Environment And Get Reward
        Args:
        action : list or tuple
        action = applied action
        num_repeat : int
        num_repeat = number of repeated actions
        Returns:
        reward : float
        reward = reward of last action
        """
        assert (not self.done), 'The episode is done'
        reward = 0
        for _ in range(num_repeat):
            self.ob, reward, self.done, _ = self.env.step(action)
            self.score += reward
            if (self.done):
                break
        reward = reward_reshape(reward, self.game_name,
                            self.level_name, self.task_name)
        return reward

    def new_episode(self):
        """
        Start A New Episode
        """
        self.ob = self.env.reset()
        self.done = False
        self.score = 0

    def episode_end(self):
        """
        Check If The Episode Ends
        Returns:
        ep_end : bool
        ep_end = when the episode finishes, return True
        """
        return self.done

    def action_set(self):
        """
        Get Actions Set
        Returns:
        actions : list
        actions = list of actions
        """
        return self.actions

    def available_action(self):
        """
        Get Indices of Available Actions For Current State
        Returns:
        available_ind : list
        available_ind = indices of available action
        """
        return range(self.actions)

    def episode_total_score(self):
        """
        Get Total Score For Last Episode
        """
        return self.score

    def close(self):
        """
        Close The Environment
        """
        self.env.close()
        return True


