# TODO: Complete README.md

# Combined Dynamic Autoencoder (CDAE)
## Dynamic Autoencoder for Learning Autonomous Driving Task

## Prerequisites

The implementation is based on python3 and [PyTorch](https://pytorch.org). The requirements are included in the [requirements file](requirements.txt):
```bash
pip3 install -r requirements.txt
```

## Datasets

The following datasets are used:

1) Simulated using [CARLA](https://carla.org/) simulator.
2) [Udacity](https://github.com/udacity/self-driving-car/tree/master/datasets) dataset with additional post-processing

TODO: Upload simulated dataset

TODO: Explain additional post-processing

## Model Overview

Combined Dynamic Autoencoder consists of three main parts:

1) Autoencoder
2) Gated Recurrent Unit
3) Imitation / Reinforcement learning controller network (multilayer perceptron)

## Training Procedure

TODO: Complete explanation

### 1) Train the autoencoder

```bash
python train_ae.py --config config/model/carla_ae.yaml
```

The default config is provided in the example.

### 2) Train the GRU

```bash
python train_rnn.py --config config/model/carla_rnn.yaml
```

Note that autoencoder is not frozen and fine-tuned as a part of GRU

### 3) Train the imitation learning network

```bash
python train_imitation.py --config config/model/carla_imitation.yaml
```

