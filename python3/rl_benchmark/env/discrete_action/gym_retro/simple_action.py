from mlt_pkg.reinforcement_learning.env.gym_retro.std_env import GymRetroEnvironment as BasicEnv
from mlt_pkg.reinforcement_learning.env.gym_retro.reward_helper import reward_reshape
import retro

class GymRetroEnvironment(BasicEnv):
    """
    Only Left, Right, And Shoot Actions
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
        self.actions = [[False for _ in range(12)] for _ in range(4)]
        self.actions[1][0] = True # shoot
        self.actions[2][6] = True # left
        self.actions[3][7] = True # right

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
        action = list(action)
        reward = 0
        for _ in range(num_repeat):
            self.ob, reward, self.done, _ = self.env.step(action)
            action[0] = False
            self.score += reward
            if (self.done):
                break
        reward = reward_reshape(reward, self.game_name,
                            self.level_name, self.task_name)
        return reward


