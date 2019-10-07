from multiprocessing import Process, Pipe
from rl_benchmark.env.discrete_action.abstract_env import AbstractEnvironment
from rl_benchmark.env.discrete_action.gym_retro.std_env import GymRetroEnvironment as EnvCore

def env_process(game_name, level_name, task_name, conn):
    """
    One Process For The Environment
    Args:
    game_name : string
    game_name = name of the game
    level_name : string
    level_name = name of the level
    task_name : string
    task_name = name of the task
    conn : multiprocessing.connection.Connection
    conn = pipe to communicate with main process
    """
    env = EnvCore(game_name, level_name, task_name)
    env_func = {'get_state': env.get_state,
            'apply_action': env.apply_action,
            'new_episode': env.new_episode,
            'episode_end': env.episode_end,
            'action_set': env.action_set,
            'available_action': env.available_action,
            'episode_total_score': env.episode_total_score,
            'close': env.close}
    while (True):
        cmd_tuple = conn.recv()
        func_name, func_args = cmd_tuple
        func_return = env_func[func_name](**func_args)
        conn.send(func_return)
        if (func_name == 'close'):
            conn.close()
            break

class GymRetroEnvironment(AbstractEnvironment):
    """
    Wrapper For Standard Gym Retro Environment
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
        self.conn_wrapper, self.conn_process = Pipe(duplex = True)
        self.env_process = Process(target = env_process, args =
                                            (game_name, level_name,
                                            task_name, self.conn_process,))
        self.env_process.start()
        self.new_episode()

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
        func_name = 'get_state'
        func_args = {'setting': setting}
        self.conn_wrapper.send((func_name, func_args))
        state = self.conn_wrapper.recv()
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
        func_name = 'apply_action'
        func_args = {'action': action, 'num_repeat': num_repeat}
        self.conn_wrapper.send((func_name, func_args))
        reward = self.conn_wrapper.recv()
        return reward

    def new_episode(self):
        """
        Start A New Episode
        """
        func_name = 'new_episode'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        _ = self.conn_wrapper.recv()

    def episode_end(self):
        """
        Check If The Episode Ends
        Returns:
        ep_end : bool
        ep_end = when the episode finishes, return True
        """
        func_name = 'episode_end'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        ep_end = self.conn_wrapper.recv()
        return ep_end

    def action_set(self):
        """
        Get Actions Set
        Returns:
        actions : list
        actions = list of actions
        """
        func_name = 'action_set'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        actions = self.conn_wrapper.recv()
        return actions

    def available_action(self):
        """
        Get Indices of Available Actions For Current State
        Returns:
        available_ind : list
        available_ind = indices of available action
        """
        func_name = 'available_action'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        available_ind = self.conn_wrapper.recv()
        return available_ind

    def episode_total_score(self):
        """
        Get Total Score For Last Episode
        """
        func_name = 'episode_total_score'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        score = self.conn_wrapper.recv()
        return score

    def close(self):
        """
        Close The Environment
        """
        func_name = 'close'
        func_args = {}
        self.conn_wrapper.send((func_name, func_args))
        ret = self.conn_wrapper.recv()
        self.env_process.join()
        return ret


