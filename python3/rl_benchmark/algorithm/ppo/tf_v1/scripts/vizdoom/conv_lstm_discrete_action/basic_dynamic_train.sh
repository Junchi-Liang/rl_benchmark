python3 train.py --n_env 8 --n_step 5 --num_repeat 1\
 --model_config basic_dynamic.ppo.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/basic_dynamic.cfg\
 --task basic\
 --max_total_step 300000 --report_per_it 10000 --test_episode 50\
 --statistics_output /tmp/vizdoom/ppo_stat_basic_dynamic\
 --model_output_prefix /tmp/vizdoom/ppo_basic_dynamic\
 --gpu_id 0 --gpu_usage 0.3\
 --optimizer Adam --gamma 0.99 --lambda_gae 1 --learning_rate 1e-4\
 --grad_clip_norm 40 --coeff_value 0.5 --coeff_policy 1 --coeff_entropy 0.01\
 --value_clip_range 0.05 --policy_clip_range 0.01 --n_epoch 2 --minibatch_size 2 


