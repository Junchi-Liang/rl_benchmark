python3 train.py --n_step 5 --num_repeat 1\
 --model_config basic_dynamic.dqn.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/basic_dynamic.cfg\
 --task basic\
 --max_total_step 300000 --report_per_it 10000 --test_episode 50\
 --statistics_output /tmp/vizdoom/dqn_stat_basic_dynamic\
 --model_output_prefix /tmp/vizdoom/dqn_basic_dynamic\
 --gpu_id 0\
 --optimizer Adam --gamma 0.99 --learning_rate 1e-4 --minibatch_size 32\
 --begin_expolration_prob 1 --end_exploration_prob 0.1\
 --begin_expolration_decay_step 1000 --end_exploration_decay_step 100000\
 --replay_buffer_size 1000 --update_target_per_main_update 100 --tau 1

