import numpy as np

def reward_reshape(raw_reward, game_name, task_name):
    """
    Reshape Reward
    Args:
    raw_reward : float
    raw_reward = raw reward
    game_name : string
    game_name = name of the game
    task_name : string
    task_name = name of the task
    Returns:
    reward : float
    reward = reshaped reward
    """
    if (task_name == 'basic'):
        reward = raw_reward
    elif (task_name == 'normalize'):
        reward = raw_reward / 15.
    else:
        raise NotImplementedError
    return reward

