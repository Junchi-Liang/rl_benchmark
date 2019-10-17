# Instructions

Here are some scripts for training and testing PPO in Vizdoom.

## Training

To train on "basic dynamic" task, please run:

```
source basic_dynamic_train.sh
```

To train on "defend the center" task, please run:

```
source defend_the_center_train.sh
```

You may want to check arguments in the shell scripts. Details explanation can be found in [train.py](https://github.com/Junchi-Liang/rl_benchmark/blob/master/python3/rl_benchmark/algorithm/ppo/torch_v1/scripts/vizdoom/conv_lstm_discrete_action/train.py).

A few things need to be checked before you run:

`--task_file` should point to the cfg file. While you can use different reward designs in the same scenario through `--task`, you can find detailed setting [here](https://github.com/Junchi-Liang/rl_benchmark/blob/master/python3/rl_benchmark/env/discrete_action/vizdoom/std_env.py#L74). Basically, you can just use task name in `--task`.

`--statistics_output` specifies the directory under where TensorBoard log files locate, please prepare this directory before you run this script.

`--model_output_prefix` specifies the output prefix of trained model file name. For example, if you put `\tmp\model` here, you will get trained model files like `\tmp\model_it_10000.pth` and `\tmp\model_it_20000.pth` for result after 10000 and 20000 iterations separately. Please prepare directory properly before you run this script.

`--gpu_id` specifies the ID of which GPU you want to use for training, if you want to use CPU for training, just ignore this argument and remove this argument.

## Testing

To test on "basic dynamic" task, please run:

```
source basic_dynamic_visualize.sh
```

To test on "defend the center" task, please run:

```
source defend_the_center_visualize.sh
```

You may want to check arguments in the shell scripts. Details explanation can be found in [test.py](https://github.com/Junchi-Liang/rl_benchmark/blob/master/python3/rl_benchmark/algorithm/ppo/torch_v1/scripts/vizdoom/conv_lstm_discrete_action/test.py).

A few things need to be checked before you run:

`--task_file`, `--task`, and `--gpu_id` are the same as they are described in training section.

`--model_file` specifies trained model file, you can get them from training, which is specified by `--model_output_prefix` in training scripts.

`--visualize_pause` specifies how long you want to wait in each frame in millisecond, if you just want to see the statistics without the visualization, just give `--visualize_pause 0`.

`--visualize_height` and `--visualize_width` specify visualization window size.

## Additional Note

### basic dynamic

Using provided training script, you should be able to see average score above 80 within 100000 iterations. (Of course, this is reinforcement learning, I cannot guarantee success in each training here :sweat_smile: )

### defend the center

Provided script trains for 3000000 iterations, it may take quite a long time to converge. You may want to try a trained model [here](https://www.dropbox.com/s/62k9e3pdnp3udhw/ppo_defend_the_center.pth?dl=0).

