import argparse
import time
import os
import json

import tensorflow as tf
from tqdm import trange

from rl_benchmark.algorithm.ppo.tf_v1.model.conv_lstm_discrete_action import PPOModel
from rl_benchmark.algorithm.ppo.tf_v1.worker.conv_lstm_discrete_action import PPOWorker
from rl_benchmark.env.discrete_action.vizdoom.std_env import VizdoomEnvironment

# get training configuration from argument
parser = argparse.ArgumentParser()
parser.add_argument('--n_env', type = int,
        help = 'Number of environments', default = 8)
parser.add_argument('--n_step', type = int,
        help = 'Number of steps in each sampling',
        default = 4)
parser.add_argument('--model_config',
        help = 'Path to model configuration file')
parser.add_argument('--task_file', help = 'Path to cfg file')
parser.add_argument('--task', help = 'Name of task')
parser.add_argument('--optimizer', help = 'Type of optimizer',
                        default = 'Adam')
parser.add_argument('--gamma', help = 'Discount factor',
                        type = float, default = 0.99)
parser.add_argument('--statistics_output',
        help = 'Output path for statistics',
        default = None)
parser.add_argument('--num_repeat',
        help = 'Number of repeated actions',
        type = int, default = 4)
parser.add_argument('--max_total_step',
        help = 'Max total steps', type = int, default = 100000)
parser.add_argument('--report_per_it',
        help = 'Report every report_per_it iteration',
        type = int, default = 10000)
parser.add_argument('--gpu_id', help = 'GPU id')
parser.add_argument('--gpu_usage', help = 'Percentage of GPU usage',
                                type = float, default = 0.3)
parser.add_argument('--model_output_prefix',
                help = 'Prefix for model output path', default = None)
parser.add_argument('--coeff_value', help = 'Coefficient for critic',
        type = float, default = 0.5)
parser.add_argument('--coeff_policy', help = 'Coefficient for policy',
        type = float, default = 1.)
parser.add_argument('--coeff_entropy',
        help = 'Coefficient for entropy of policy',
        type = float, default = 0.01)
parser.add_argument('--grad_clip_norm',
        help = 'Size of gradient clip',
        type = float, default = 40)
parser.add_argument('--value_clip_range',
        help = 'Clip range for value loss',
        type = float, default = 0.2)
parser.add_argument('--policy_clip_range',
        help = 'Clip range for policy loss',
        type = float, default = 0.2)
parser.add_argument('--test_episode', help = 'Number of test episodes',
        type = int, default = 20)
parser.add_argument('--learning_rate', help = 'Initial learning rate',
        type = float, default = 1e-4)
parser.add_argument('--lambda_gae',
        help = 'Lambda for generalized advantage estimation',
        type = float, default = 1)
parser.add_argument('--learning_rate_decay',
        help = 'Decay coefficient for learning rate',
        type = float, default = 1.)
parser.add_argument('--learning_rate_decay_per_it',
        help = 'Decay learning rate per learning_rate_decay_per_it',
        type = int, default = -1)
parser.add_argument('--n_epoch',
        help = 'Number of epochs after each sampling',
        type = int, default = 4)
parser.add_argument('--minibatch_size',
        help = 'Size of minibatch',
        type = int, default = 2)
args = parser.parse_args()

# set up model and worker
env_set = [VizdoomEnvironment(args.task_file, args.task)
                for _ in range(args.n_env)]
env_test = VizdoomEnvironment(args.task_file, args.task)
print('Environment is ready')
model_config = json.load(open(args.model_config))
model_config['n_action'] = len(env_test.action_set())
model = PPOModel('ppo', **model_config)
print('Model is ready')
worker = PPOWorker(model, env_set, args.gamma, args.lambda_gae,
                        optimizer_type = args.optimizer,
                        grad_clip_norm = args.grad_clip_norm,
                        coeff_value = args.coeff_value,
                        coeff_policy = args.coeff_policy,
                        coeff_entropy = args.coeff_entropy)
print('Worker is ready')
# set up tensorflow session
tf_config = tf.ConfigProto()
if (args.gpu_id is not None and int(args.gpu_id) >= 0):
    tf_config.gpu_options.visible_device_list = args.gpu_id
    if (args.gpu_usage is not None):
        tf_config.gpu_options.per_process_gpu_memory_fraction = args.gpu_usage
sess = tf.Session(config = tf_config)
print('Tensorflow session ready')
# set up summary writer if necessary
if (args.statistics_output is not None):
    if (not os.path.exists(args.statistics_output)):
        os.mkdir(args.statistics_output)
    summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(
                                        args.statistics_output, 'test'))
else:
    summary_writer = None
# initialize variables
worker.reset_optimizer(sess)
print('Optimizer ready')
model.init_param(sess)
print('Model initialized')
# initial test
print('Initial Test')
report = worker.test(sess, env_test, args.test_episode, args.num_repeat)
print("Results: score mean: %.5f(%.5f)," %
        (report['score_mean'], report['score_std']),
        "min: %.5f" % report['score_min'],
        "max: %.5f" % report['score_max'])
print(time.ctime())
if (summary_writer is not None):
    summary = tf.compat.v1.Summary()
    summary.value.add(tag = 'Average Total Reward',
                        simple_value = report['score_mean'])
    summary.value.add(tag = 'Score Standard Variance',
                        simple_value = report['score_std'])
    summary_writer.add_summary(summary, 0)
    summary_writer.add_graph(sess.graph)
    summary_writer.flush()
if (args.model_output_prefix is not None):
    model_output_path = args.model_output_prefix + '_it_0.npz'
    model.save_to_npz(sess, model_output_path)
    print('Model saved to ', model_output_path)
# training
print('Start Training')
learning_rate = args.learning_rate
time_start = time.time()
total_step = 0
total_ep = 0
last_lr_decay_it = 0
while (total_step < args.max_total_step):
    step_total_next = min(args.max_total_step - total_step,
                                        args.report_per_it)
    print('-----Itertion %d/%d ~ %d/%d-----' %
                (total_step, args.max_total_step,
                total_step + step_total_next, args.max_total_step))
    it_cnt = 0
    print('Learning rate: ' + str(learning_rate))
    for learning_step in trange(step_total_next):
        if (it_cnt <= learning_step):
            step_sampled, finished_episode = worker.train_batch(sess,
                        args.n_step,
                        step_total_next - it_cnt, args.num_repeat,
                        args.value_clip_range, args.policy_clip_range,
                        learning_rate, args.n_epoch, args.minibatch_size)
            it_cnt += step_sampled
            total_ep += finished_episode
        if (args.learning_rate_decay_per_it > 0 and
                total_step + it_cnt - last_lr_decay_it >=
                                args.learning_rate_decay_per_it):
            learning_rate *= args.learning_rate_decay
            last_lr_decay_it = total_step + it_cnt
    total_step += step_total_next
    print('Test After %d Iterations' % total_step)
    report = worker.test(sess, env_test, args.test_episode, args.num_repeat)
    print("Results: score mean: %.5f(%.5f)," %
            (report['score_mean'], report['score_std']),
            "min: %.5f" % report['score_min'],
            "max: %.5f" % report['score_max'])
    print('Total training episode: %d' % total_ep)
    total_minute = (time.time() - time_start) / 60.0
    if (total_minute < 60):
        print("Total elapsed time: %.2f minutes" % total_minute)
    else:
        total_hour = int(total_minute / 60)
        total_minute -= 60 * total_hour
        print("Total elapsed time: %d hour %.2f minutes" %
                                        (total_hour, total_minute))
    print(time.ctime())
    if (summary_writer is not None):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag = 'Average Total Reward',
                            simple_value = report['score_mean'])
        summary.value.add(tag = 'Score Standard Variance',
                            simple_value = report['score_std'])
        summary.value.add(tag = 'Training Episode', simple_value = total_ep)
        summary_writer.add_summary(summary, total_step)
        summary_writer.flush()
    if (args.model_output_prefix is not None):
        model_output_path = args.model_output_prefix + ('_it_%d.npz' %
                                                                total_step)
        model.save_to_npz(sess, model_output_path)
        print('Model saved to ', model_output_path) 

print('Closing environments')
env_test.close()
for env in env_set:
    env.close()
print('done')

