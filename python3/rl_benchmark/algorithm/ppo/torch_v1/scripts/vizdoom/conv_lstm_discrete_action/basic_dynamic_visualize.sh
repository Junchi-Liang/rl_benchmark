python3 test.py --num_repeat 1\
 --model_config basic_dynamic.ppo.json\
 --task_file ~/robotics_workspace/vizdoom_experiment/scenarios/basic_dynamic.cfg\
 --task basic\
 --test_episode 50\
 --gpu_id 0\
 --model_file /tmp/vizdoom/ppo_basic_dynamic_it_30000.pth\
 --visualize_pause 10 --visualize_height 360 --visualize_width 480

