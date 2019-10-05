from mlt_pkg.reinforcement_learning.env.discrete_action.gym_super_mario_bros.std_env import MarioEnvironment as BasicEnvironment
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

class MarioEnvironment(BasicEnvironment):
    """
    Only Left, Right, And Jump
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
        self.actions = [0, 1, 5, 6]
        self.n_action = len(self.actions)
        self.new_episode()


