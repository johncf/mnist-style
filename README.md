# MNIST Auto-Encoder

## Roadmap

- [x] Simple auto-encoder
- [ ] A script to visualize latent feature-space.
- [ ] Adverserial auto-encoder for the feature-space to have a gaussian distribution.

## Setup

```
$ python3 -m venv pyenv
$ . pyenv/bin/activate
$ pip3 install -r requirements.txt
```

## Usage

```
$ ./mnist-aae.py --help
usage: mnist-aae.py [-h] [--batch-size B] [--epochs E] [--lr LR]
                    [--feature-size N] [--state-prefix pre]

MXNet Gluon MNIST Autoencoder

optional arguments:
  -h, --help          show this help message and exit
  --batch-size B      batch size for training and testing (default: 100)
  --epochs E          number of epochs to train (default: 5)
  --lr LR             learning rate with adam optimizer (default: 0.005)
  --feature-size N    rank of the latent feature vector (default: 8)
  --state-prefix pre  path-prefix of state files (default: mnist) state files
                      will be of the form "prefixN.key.params"
```

```
$ ./mnist-aae.py --feature-size 4 --epochs 2
... ignore warnings that show up here ...
[Epoch 1 Batch 100] Training: mse=0.023652
[Epoch 1 Batch 200] Training: mse=0.023718
[Epoch 1 Batch 300] Training: mse=0.023698
[Epoch 1 Batch 400] Training: mse=0.023704
[Epoch 1 Batch 500] Training: mse=0.023715
[Epoch 1 Batch 600] Training: mse=0.023761
[Epoch 1] Training: mse=0.023761
50 test images written to /tmp/mnist
[Epoch 1] Validation: mse=0.023958
[Epoch 2 Batch 100] Training: mse=0.023553
[Epoch 2 Batch 200] Training: mse=0.023681
[Epoch 2 Batch 300] Training: mse=0.023674
[Epoch 2 Batch 400] Training: mse=0.023656
[Epoch 2 Batch 500] Training: mse=0.023677
[Epoch 2 Batch 600] Training: mse=0.023669
[Epoch 2] Training: mse=0.023669
50 test images written to /tmp/mnist
[Epoch 2] Validation: mse=0.023812
```
