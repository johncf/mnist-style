# MNIST Auto-Encoder

## Roadmap

- [x] Simple auto-encoder
- [x] A script to visualize latent feature-space.
- [x] Adverserial auto-encoder for the feature-space to have a gaussian distribution.

## Setup

```
$ python3 -m venv pyenv
$ source pyenv/bin/activate
$ pip3 install -r requirements.txt
```

## Usage

```
$ ./mnist-sae.py --help
usage: mnist-sae.py [-h] [--batch-size B] [--epochs E] [--lr LR]
                    [--feature-size N] [--state-prefix pre]

MNIST Simple Auto-Encoder

optional arguments:
  -h, --help          show this help message and exit
  --batch-size B      batch size for training and testing (default: 100)
  --epochs E          number of epochs to train (default: 5)
  --lr LR             learning rate with adam optimizer (default: 0.005)
  --feature-size N    dimensions of the latent feature vector (default: 8)
  --state-prefix pre  path-prefix of state files (default: mnist) state files
                      will be of the form "prefixN.key.params"
```

```
$ ./mnist-sae.py --feature-size 4
... ignore warnings that show up here ...
[Epoch 1 Batch 100] Training: mse=0.070375
[Epoch 1 Batch 200] Training: mse=0.058631
[Epoch 1 Batch 300] Training: mse=0.052309
[Epoch 1 Batch 400] Training: mse=0.048182
[Epoch 1 Batch 500] Training: mse=0.045372
[Epoch 1 Batch 600] Training: mse=0.043215
[Epoch 1] Training: mse=0.043215
50 test images written to /tmp/mnist
[Epoch 1] Validation: mse=0.032171
[Epoch 2 Batch 100] Training: mse=0.031460
... output redacted for brevity ...
[Epoch 5 Batch 500] Training: mse=0.025346
[Epoch 5 Batch 600] Training: mse=0.025322
[Epoch 5] Training: mse=0.025322
50 test images written to /tmp/mnist
[Epoch 5] Validation: mse=0.024895
```

## Feature Space Visualisation

If you're interested in visualizing the feature space which your trained
encoder maps to, there's a script to do just that.

For that, we first need to install the visualisation requirements. Note that
PyQt5 is listed in it, which you may skip if you manually replace `plt.show()`
with `plt.savefig('plot.svg')` as well as remove `mpl.use('PyQt5Agg')` right
after `import matplotlib`.

```
$ pip install -r vis-requirements.txt
```

Executing the following script creates an encoder model with saved parameters,
runs it on 500 test images (by default), and displays the feature space as a
scatter matrix.

```
$ ./encoder-vis.py --feature-size 4
```

[![vis-sample](https://i.imgur.com/6IN5FDkl.png)](https://i.imgur.com/6IN5FDk.png)

_(click to enlarge)_
