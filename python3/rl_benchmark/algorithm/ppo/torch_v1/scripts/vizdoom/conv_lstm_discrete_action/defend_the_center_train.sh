python3 train.py --n_env 16 --n_step 5 --num_repeat 1\
 --model_config defend_the_center.ppo.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/defend_the_center.cfg\
 --task defend_the_center\
 --max_total_step 3000000 --report_per_it 10000 --test_episode 20\
 --statistics_output /tmp/vizdoom/ppo_stat_defend_the_center\
 --model_output_prefix /tmp/vizdoom/ppo_defend_the_center\
 --gpu_id 0\
 --optimizer Adam --gamma 0.99 --lambda_gae 1 --learning_rate 1e-4\
 --grad_clip_norm 40 --coeff_value 0.5 --coeff_policy 1 --coeff_entropy 0.01\
 --value_clip_range 0.05 --policy_clip_range 0.01 --n_epoch 1 --minibatch_size 4


