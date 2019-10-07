from rl_benchmark.env.discrete_action.gym_retro.std_wrapper import GymRetroEnvironment as BasicWrapper
from rl_benchmark.env.discrete_action.gym_retro.simple_action import GymRetroEnvironment as EnvCore
from multiprocessing import Process, Pipe

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

class GymRetroEnvironment(BasicWrapper):
    """
    Wrapper For Gym Retro Environment With Simplified Actions
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


