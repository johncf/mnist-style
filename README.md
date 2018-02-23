# MNIST Style-based Auto-Encoder

## Roadmap

- [x] Simple auto-encoder
- [x] A script to visualize latent feature-space (style-space).
- [x] Adverserial auto-encoder to fit the style-space into a gaussian distribution.
- [ ] A script to generate images of all digits from a random style-vector.

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
                    [--feature-size N] [--ckpt-dir ckpt]

MNIST Simple Auto-Encoder

optional arguments:
  -h, --help        show this help message and exit
  --batch-size B    batch size for training and testing (default: 100)
  --epochs E        number of epochs to train (default: 5)
  --lr LR           learning rate with adam optimizer (default: 0.005)
  --feature-size N  dimensions of the latent feature vector (default: 4)
  --ckpt-dir ckpt   training session directory (default: mnistN.ckpt) for
                    storing model parameters and trainer states
```

```
$ ./mnist-aae.py --feature-size 4
... ignore warnings that show up here ...
[Epoch 1 Batch 100] Training: mse=0.0671
[Epoch 1 Batch 200] Training: mse=0.0572
[Epoch 1 Batch 300] Training: mse=0.0510
[Epoch 1 Batch 400] Training: mse=0.0468
[Epoch 1 Batch 500] Training: mse=0.0440
[Epoch 1 Batch 600] Training: mse=0.0419
[Epoch 1] Training:
  AutoEncoder: mse=0.0419
  GaussDiscriminator: actual gauss detection accuracy=0.8523
  GaussDiscriminator: feature space detection accuracy=0.9687
[Epoch 1] Validation:
  AutoEncoder: mse=0.0300
  GaussDiscriminator: feature space satisfaction accuracy=0.0484
  test images written to /tmp/mnist
Model parameters and trainer state saved.
[Epoch 2 Batch 100] Training: mse=0.0303
[Epoch 2 Batch 200] Training: mse=0.0300
... output redacted for brevity ...
[Epoch 24 Batch 500] Training: mse=0.0231
[Epoch 24 Batch 600] Training: mse=0.0231
[Epoch 24] Training:
  AutoEncoder: mse=0.0231
  GaussDiscriminator: actual gauss detection accuracy=0.3636
  GaussDiscriminator: feature space detection accuracy=0.6523
[Epoch 24] Validation:
  AutoEncoder: mse=0.0235
  GaussDiscriminator: feature space satisfaction accuracy=0.1660
  test images written to /tmp/mnist
Model parameters and trainer state saved.
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

The following image is the output from an auto-encoder model that was trained
with the simple approach ([`mnist-sae.py`](./mnist-sae.py)).

[![sae-sample](https://i.imgur.com/fBaF6tcl.png)](https://i.imgur.com/fBaF6tc.png)

_(click to enlarge)_

And below is the same from an adverserially trained auto-encoder model
([`mnist-aae.py`](./mnist-aae.py)).

[![aae-sample](https://i.imgur.com/pI3iQyBl.png)](https://i.imgur.com/pI3iQyB.png)

_(click to enlarge)_
