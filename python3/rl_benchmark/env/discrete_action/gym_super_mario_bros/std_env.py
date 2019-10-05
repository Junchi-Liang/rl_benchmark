from mlt_pkg.reinforcement_learning.env.discrete_action.abstract_env import AbstractEnvironment
from skimage.transform import resize
import numpy as np
from skimage.color import rgb2grey
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from mlt_pkg.reinforcement_learning.env.gym_super_mario_bros.reward_helper import reward_reshape

class MarioEnvironment(AbstractEnvironment):
    """
    Standard Super Mario Bros Environment
    https://github.com/Kautenja/gym-super-mario-bros
    """
    def __init__(self, game_name, task_name,
                    action_mode = SIMPLE_MOVEMENT,
                    state_size = None):
        """
        Args:
        game_name : string
        game_name = name of the game (e.g. SuperMarioBros-5-1-v0)
        task_name : string
        task_name = name of the task
        state_size : list or tuple or None
        state_size = size of state, [h, w] or [h, w, c]
        """
        self.game_name = game_name
        self.task_name = task_name
        self.action_mode = action_mode
        self.env = gym_super_mario_bros.make(game_name)
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env,
                                                self.action_mode)
        self.n_action = self.env.action_space.n
        self.actions = [a for a in range(self.n_action)]
        self.new_episode()

    def get_state(self, setting = None):
        """
        Get Current State
        Args:
        setting : dictionary
        setting = setting for states
            'resolution' : list or tuple or None
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
        action : int
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
        reward = reward_reshape(reward, self.game_name, self.task_name)
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


