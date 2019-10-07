from vizdoom import *
import numpy as np
from skimage.color import rgb2grey
import cv2
from rl_benchmark.env.discrete_action.abstract_env import AbstractEnvironment

class VizdoomEnvironment(AbstractEnvironment):
    """
    Standard Vizdoom Environment
    """
    def __init__(self, config_file_path, task_name,
                screen_format = vizdoom.ScreenFormat.RGB24,
                screen_resolution = vizdoom.ScreenResolution.RES_640X480,
                window_visible = False, mode = Mode.PLAYER,
                state_size = [84, 84]):
        """
        Args:
        config_file_path : string
        config_file_path = path to cfg file
        screen_format : vizdoom.vizdoom.ScreenFormat
        screen_format = format of screen
        screen_resolution : vizdoom.vizdoom.ScreenFormat
        screen_resolution = resolution of screen
        window_visible : bool
        window_visible = when this is True, visualization window is shown
        mode : vizdoom.vizdoom.Mode
        mode = PLAYER or SPECTATOR
        state_size : list or tuple
        state_size = size of state, [h, w]
        """
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_screen_format(screen_format)
        self.game.set_screen_resolution(screen_resolution)
        self.game.set_window_visible(window_visible)
        self.game.set_mode(mode)
        self.game.init()
        self.task = task_name
        self.state_size = state_size
        n_button = self.game.get_available_buttons_size()
        self.actions = np.concatenate([np.identity(n_button, dtype = bool),
                                    np.zeros([1, n_button], dtype = bool)],
                                    axis = 0).tolist()

    def get_state(self, setting = None):
        """
        Get Current State
        Args:
        setting : dictionary
        setting = setting for states
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
        screen = self.game.get_state().screen_buffer
        if (len(resolution) == 3 and resolution[2] == 1):
            screen = rgb2grey(screen)
        if (screen.ndim == 2):
            screen = np.expand_dims(screen, axis = -1)
        assert (screen.ndim == 3), 'shape of screen should be [h, w, c]'
        state = cv2.resize(screen, tuple(resolution[:2][::-1]))
        state = state.astype(np.float)
        if (state.max() > 1):
            state /= 255.
        return state
        
    def apply_action(self, action, num_repeat = 1):
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
        self.game.set_action(action)
        for _ in range(num_repeat):
            self.game.advance_action()
            if (self.game.is_episode_finished()):
                break
        reward = self.game.get_last_reward()
        if (self.task == 'basic'):
            reward /= 100.
        elif (self.task == 'basic_raw'):
            reward = reward * 1.
        elif (self.task == 'defend_the_center'):
            reward = reward * 1.
        elif (self.task == 'defend_the_centerx100'):
            reward = reward * 100.
        elif (self.task == 'defend_the_line_move'):
            reward = reward * 1.
        elif (self.task == 'defend_the_line_movex100'):
            reward = reward * 100.
        elif (self.task == 'defend_the_line_movex10'):
            reward = reward * 10.
        elif (self.task == 'predict_position'):
            reward = reward * 1.
        elif (self.task == 'deadly_corridor'):
            reward = reward * 1.
        elif (self.task == 'my_way_home'):
            reward = reward * 1.
        elif (self.task == 'health_gathering'):
            reward = reward * 1.
        elif (self.task == 'take_cover'):
            reward = reward * 1.
        else:
            raise NotImplementedError
        return reward

    def new_episode(self):
        """
        Start A New Episode
        """
        self.game.new_episode()

    def episode_end(self):
        """
        Check If The Episode Ends
        Returns:
        ep_end : bool
        ep_end = when the episode finishes, return True
        """
        return self.game.is_episode_finished()

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
        return self.game.get_total_reward()

    def close(self):
        """
        Close The Environment
        """
        self.game.close()
        return True

