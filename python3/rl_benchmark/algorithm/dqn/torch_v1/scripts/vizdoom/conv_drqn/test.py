import argparse
import time
import os
import json
import torch
import tensorflow as tf
from tqdm import trange

from rl_benchmark.algorithm.dqn.torch_v1.model.lstm_conv_duel import DQNModel
from rl_benchmark.algorithm.dqn.torch_v1.worker.lstm_conv_drqn import DQNWorker
from rl_benchmark.algorithm.dqn.replay_buffer.base import ReplayBuffer
from rl_benchmark.env.discrete_action.vizdoom.std_env import VizdoomEnvironment
from rl_benchmark.algorithm.dqn.misc_utils.exploration_schedule import linear_decay as exploration_schedule

# get training configuration from argument
parser = argparse.ArgumentParser()
parser.add_argument('--model_config',
        help = 'Path to model configuration file')
parser.add_argument('--model_file', help = 'Path to model file')
parser.add_argument('--task_file', help = 'Path to cfg file')
parser.add_argument('--task', help = 'Name of task')
parser.add_argument('--num_repeat',
        help = 'Number of repeated actions',
        type = int, default = 4)
parser.add_argument('--test_episode',
        help = 'Number of test episodes',
        type = int, default = 20)
parser.add_argument('--gpu_id', help = 'GPU id')
parser.add_argument('--visualize_pause',
        help = 'Pause time in ms visualization', type = int, default = -1)
parser.add_argument('--visualize_height',
                help = 'Height for visualization', type = int)
parser.add_argument('--visualize_width',
                help = 'Width for visualization', type = int)
args = parser.parse_args()

if __name__ == '__main__':
    # set up model and worker
    env_test = VizdoomEnvironment(args.task_file, args.task)
    print('Environment is ready')
    model_config = json.load(open(args.model_config))
    model_config['n_action'] = len(env_test.action_set())
    model = DQNModel(**model_config).to(dtype = torch.float32)
    print('Model is ready')
    print('Load model from', args.model_file)
    model.load_state_dict(torch.load(
        args.model_file, map_location = 'cpu'))
    print('Model loaded')
    worker = DQNWorker(model, None, None)
    print('Worker is ready')
    # set up device
    print('Setup device')
    if (args.gpu_id is not None):
        device = torch.device('cuda:' + str(args.gpu_id))
        model.to(device = device)
    else:
        model.to(device = 'cpu')
    print('Device setup done')
    # test
    print('Start Test')
    if (args.visualize_pause == 0):
        report = worker.test(env_test, args.test_episode,
                                args.num_repeat)
    else:
        report = worker.test(env_test, args.test_episode,
                        args.num_repeat,
                        True, args.visualize_pause,
                        [args.visualize_height, args.visualize_width],
                        False, True)
    print("Results: score mean: %.5f(%.5f)," %
            (report['score_mean'], report['score_std']),
            "min: %.5f" % report['score_min'],
            "max: %.5f" % report['score_max'])

    print('Closing environments')
    env_test.close()
    print('done')

