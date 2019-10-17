# Reinforcement Learning Benchmark
Implementations of some modern (deep) reinforcement learning algorithms.

As Python 2 will be officially deprecated at the end of 2019, only Python 3 implementations are provided, unless there is a clear reason for Python 2 implementations.

For now, implementations focus on Pytorch, some Tensorflow versions are provided. Hopefully, Tensorflow 2 versions will be updated in the future.

| Algorithm | Python3 Implementation |
| --- | --- |
|D(R)QN | [Pytorch]() |
|A2C| [Pytorch]() |
|PPO| [Pytorch]() / [Tensorflow]()|

## Dependencies

This repo is tested under Ubuntu 18.04, CUDA 10.0, Python 3.6.

Tested third-party library versions:

Pytorch == 1.2.0

Tensorflow == 1.14.0

OpenCV == 4.1.1

Numpy == 1.17.2

Scipy == 1.3.1

Vizdoom == 1.1.7

Other versions may also work.

## How To Use
Before running any scripts in the repo, please run:

```
source setup3_init.sh
```

This shell script will set up PYTHONPATH in your terminal session so that Python packages in this repo are available in this terminal session.

Next, please follow links above for each algorithm.

### additional notes for Vizdoom

You can easily install Vizdoom with commands like:

```
pip3 install vizdoom
```

More details can be found in [official repo](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#pypi), you may need some [dependencies](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#deps) to use it.

I usually tested my implementations on an extended version of the basic task, [cfg file](https://www.dropbox.com/s/udurqw2slsv2zxp/basic_dynamic.cfg?dl=0) [wad file](https://www.dropbox.com/s/znt95k8xxa765fc/basic_dynamic.wad?dl=0). They should be placed under the same directory so that cfg file can find the wad file. You may also want to clone Vizdoom repo so that you can access their scenarios (you can find them under their [scenarios folder](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios)).

