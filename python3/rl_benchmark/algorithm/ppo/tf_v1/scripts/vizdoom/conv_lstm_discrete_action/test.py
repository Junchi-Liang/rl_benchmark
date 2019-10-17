import argparse
import time
import os
import json

import tensorflow as tf
from tqdm import trange

from rl_benchmark.algorithm.ppo.tf_v1.model.conv_lstm_discrete_action import PPOModel
from rl_benchmark.algorithm.ppo.tf_v1.worker.conv_lstm_discrete_action import PPOWorker
from rl_benchmark.env.discrete_action.vizdoom.std_env import VizdoomEnvironment

# get testing configuration from argument
parser = argparse.ArgumentParser()
parser.add_argument('--model_config',
                        help = 'Path to model configuration file')
parser.add_argument('--model_file', help = 'Path to model NPZ file')
parser.add_argument('--task_file', help = 'Path to cfg file')
parser.add_argument('--task', help = 'Name of task')
parser.add_argument('--num_repeat', help = 'Number of repeated actions',
                                                    type = int, default = 4)
parser.add_argument('--gpu_id', help = 'GPU id')
parser.add_argument('--gpu_usage', help = 'Percentage of GPU usage',
                                type = float, default = 0.2)
parser.add_argument('--test_episode', help = 'Number of test episodes',
                                                type = int, default = 20)
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
    model = PPOModel('ppo', **model_config)
    print('Model is ready')
    worker = PPOWorker(model, [env_test])
    print('Worker is ready')
    # set up tensorflow session
    tf_config = tf.ConfigProto()
    if (args.gpu_id is not None):
        tf_config.gpu_options.visible_device_list = args.gpu_id
        if (args.gpu_usage is not None):
            tf_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_usage
    sess = tf.Session(config = tf_config)
    # load model
    print('Loading trained model')
    model.load_from_npz(sess, args.model_file)
    # test
    print('Test')
    print('Start Test')
    if (args.visualize_pause == 0):
        report = worker.test(sess, env_test, args.test_episode,
                                args.num_repeat)
    else:
        report = worker.test(sess, env_test, args.test_episode,
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

