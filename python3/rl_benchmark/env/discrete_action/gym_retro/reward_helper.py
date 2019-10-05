import numpy as np

def reward_reshape(raw_reward, game_name, level_name, task_name):
    """
    Reshape Reward
    Args:
    raw_reward : float
    raw_reward = raw reward
    game_name : string
    game_name = name of the game
    level_name : string
    level_name = name of the level
    task_name : string
    task_name = name of the task
    Returns:
    reward : float
    reward = reshaped reward
    """
    if (game_name == 'Airstriker-Genesis'):
        if (level_name == 'Level1'):
            if (task_name == 'basic'):
                reward = raw_reward / 20.
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return reward

