import argparse
import time
import os
import json
import torch
import tensorflow as tf
from tqdm import trange

from rl_benchmark.algorithm.dqn.torch_v1.model.lstm_conv_duel import DQNModel
from rl_benchmark.algorithm.dqn.torch_v1.worker.conv_dqn import DQNWorker
from rl_benchmark.algorithm.dqn.replay_buffer.base import ReplayBuffer
from rl_benchmark.env.discrete_action.vizdoom.std_env import VizdoomEnvironment
from rl_benchmark.algorithm.dqn.misc_utils.exploration_schedule import linear_decay as exploration_schedule

# get training configuration from argument
parser = argparse.ArgumentParser()
parser.add_argument('--model_config',
        help = 'Path to model configuration file')
parser.add_argument('--n_step', type = int,
        help = 'Number of steps in each sampling',
        default = 4)
parser.add_argument('--task_file', help = 'Path to cfg file')
parser.add_argument('--task', help = 'Name of task')
parser.add_argument('--optimizer',
        help = 'Type of optimizer', default = 'Adam')
parser.add_argument('--gamma',
        help = 'Discount factor',
        type = float, default = 0.99)
parser.add_argument('--statistics_output',
        help = 'Output path for statistics',
        default = None)
parser.add_argument('--model_output_prefix',
        help = 'Prefix for model output path', default = None)
parser.add_argument('--num_repeat',
        help = 'Number of repeated actions',
        type = int, default = 4)
parser.add_argument('--max_total_step',
        help = 'Max total steps', type = int, default = 100000)
parser.add_argument('--begin_expolration_prob',
        help = 'Initial exploration probability',
        type = float, default = 1.)
parser.add_argument('--end_exploration_prob',
        help = 'Final exploration probability',
        type = float, default = 0.6)
parser.add_argument('--begin_expolration_decay_step',
        help = 'Number of steps to begin exploration decay',
        type = int, default = 1000)
parser.add_argument('--end_exploration_decay_step',
        help = 'Number of steps to step exploration decay',
        type = int)
parser.add_argument('--report_per_it',
        help = 'Report every report_per_it iteration',
        type = int, default = 10000)
parser.add_argument('--grad_clip_norm',
        help = 'Size of gradient clip',
        type = float)
parser.add_argument('--test_episode',
        help = 'Number of test episodes',
        type = int, default = 20)
parser.add_argument('--learning_rate',
        help = 'Initial learning rate',
        type = float, default = 1e-4)
parser.add_argument('--learning_rate_decay',
        help = 'Decay coefficient for learning rate',
        type = float, default = 1.)
parser.add_argument('--learning_rate_decay_per_it',
        help = 'Decay learning rate per learning_rate_decay_per_it',
        type = int, default = -1)
parser.add_argument('--minibatch_size',
        help = 'Size of minibatch',
        type = int, default = 2)
parser.add_argument('--replay_buffer_size',
        help = 'Size of replay buffer', type = int, default = 2000)
parser.add_argument('--update_target_per_main_update',
        help = 'Update target model each update_target_per_main_update main update',
        type = int, default = 5)
parser.add_argument('--tau',
        help = 'Tau in target model update', type = float, default = 1.)
parser.add_argument('--gpu_id', help = 'GPU id')
args = parser.parse_args()

if __name__ == '__main__':
    # set up model and worker
    env_train = VizdoomEnvironment(args.task_file, args.task)
    env_test = VizdoomEnvironment(args.task_file, args.task)
    print('Environment is ready')
    model_config = json.load(open(args.model_config))
    model_config['n_action'] = len(env_test.action_set())
    model = DQNModel(**model_config)
    target_model = DQNModel(**model_config)
    print('Model is ready')
    replay_buffer = ReplayBuffer(args.replay_buffer_size,
            model.img_resolution, [])
    print('Replay buffer is ready')
    worker = DQNWorker(model, replay_buffer, args.optimizer)
    worker.update_target_model(target_model, model, tau = 1.)
    print('Worker is ready')
    # set up device
    print('Setup device')
    if (args.gpu_id is not None):
        device = torch.device('cuda:' + str(args.gpu_id))
        model.to(device = device, dtype = torch.float32)
        target_model.to(device = device, dtype = torch.float32)
    else:
        model.to(device = 'cpu', dtype = torch.float32)
        target_model.to(device = 'cpu', dtype = torch.float32)
    print('Device setup done')
    # set up summary writer if necessary
    if (args.statistics_output is not None):
        if (not os.path.exists(args.statistics_output)):
            os.mkdir(args.statistics_output)
        summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(
                                            args.statistics_output, 'test'))
    else:
        summary_writer = None
    # initial test
    print('Initial Test')
    report = worker.test(env_test, args.test_episode, args.num_repeat)
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
        summary_writer.flush()
    if (args.model_output_prefix is not None):
        model_output_path = args.model_output_prefix + '_it_0.pth'
        torch.save(model.state_dict(), model_output_path)
        print('Model saved to ', model_output_path)
    # training
    print('Start Training')
    env_train.new_episode()
    learning_rate = args.learning_rate
    time_start = time.time()
    total_step = 0
    total_ep = 0
    last_lr_decay_it = 0
    total_update_cnt = 0
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
                exploration_prob = exploration_schedule(
                        total_step + it_cnt,
                        args.begin_expolration_decay_step,
                        args.end_exploration_decay_step,
                        args.begin_expolration_prob,
                        args.end_exploration_prob)
                sample_step = min(args.n_step, step_total_next - it_cnt)
                sample_step, finished_episode = worker.sample_trajectory(
                        env_train, exploration_prob,
                        sample_step, args.num_repeat)
                total_ep += finished_episode
                it_cnt += sample_step
                data_batch = worker.sample_from_replay_buffer(
                        args.minibatch_size, target_model, args.gamma)
                if (data_batch is not None):
                    worker.update_model(data_batch, learning_rate,
                            args.grad_clip_norm)
                    total_update_cnt += 1
                    if (total_update_cnt %
                            args.update_target_per_main_update == 0):
                        worker.update_target_model(
                                target_model, tau = args.tau)
            if (args.learning_rate_decay_per_it > 0 and
                    total_step + it_cnt - last_lr_decay_it >=
                                    args.learning_rate_decay_per_it):
                learning_rate *= args.learning_rate_decay
                last_lr_decay_it = total_step + it_cnt
        total_step += step_total_next
        print('Test After %d Iterations' % total_step)
        report = worker.test(env_test, args.test_episode, args.num_repeat)
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
            model_output_path = args.model_output_prefix + ('_it_%d.pth' %
                                                                total_step)
            torch.save(model.state_dict(), model_output_path)
            print('Model saved to ', model_output_path)

    print('Closing environments')
    env_test.close()
    env_train.close()
    print('done')

