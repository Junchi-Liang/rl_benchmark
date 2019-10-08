python3 train.py --n_env 8 --n_step 5 --num_repeat 1\
 --model_config defend_the_center.a2c.json\
 --task_file ~/robotics_workspace/ViZDoom/scenarios/defend_the_center.cfg\
 --task defend_the_center\
 --max_total_step 3000000 --report_per_it 10000\
 --statistics_output /tmp/vizdoom/a2c_stat_defend_the_center\
 --model_output_prefix /tmp/vizdoom/a2c_defend_the_center\
 --gpu_id 0\
 --optimizer Adam --gamma 0.99 --lambda_gae 1 --learning_rate 1e-4\
 --grad_clip_norm 40 --coeff_value 0.5 --coeff_policy 1 --coeff_entropy 0.01

