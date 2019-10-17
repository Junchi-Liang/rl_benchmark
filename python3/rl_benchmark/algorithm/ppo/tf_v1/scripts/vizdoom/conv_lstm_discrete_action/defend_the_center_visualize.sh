python3 test.py --num_repeat 1\
 --model_config defend_the_center.ppo.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/defend_the_center.cfg\
 --task defend_the_center\
 --test_episode 50\
 --gpu_id 0 --gpu_usage 0.2\
 --model_file ~/robotics_workspace/rl_benchmark/trained_model/python3/tf_v1/ppo_defend_the_center.npz\
 --visualize_pause 10 --visualize_height 360 --visualize_width 480

