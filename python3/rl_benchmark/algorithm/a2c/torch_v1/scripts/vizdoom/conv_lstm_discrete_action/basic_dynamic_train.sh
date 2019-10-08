python3 train.py --n_env 8 --n_step 5 --num_repeat 1\
 --model_config basic_dynamic.a2c.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/basic_dynamic.cfg\
 --task basic\
 --max_total_step 300000 --report_per_it 10000\
 --statistics_output /tmp/vizdoom/a2c_stat_basic_dynamic\
 --model_output_prefix /tmp/vizdoom/a2c_basic_dynamic\
 --gpu_id 0\
 --optimizer Adam --gamma 0.99 --lambda_gae 1 --learning_rate 5e-5\
 --grad_clip_norm 40 --coeff_value 0.5 --coeff_policy 1 --coeff_entropy 0.01

