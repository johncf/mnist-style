# MNIST Style-based Auto-Encoder

An [Adversarial Auto-Encoder][aae] (AAE) model that encode style information of
MNIST images on a Gaussian distributed multivariate.

The model used here is slightly different from the one discussed in section 4
(Supervised Adversarial Autoencoders) of [the paper][aae]. In the paper, only
the decoder is provided with a label indicating the digit. Whereas here, we
also provide the encoder with the label.

[aae]: https://arxiv.org/abs/1511.05644

## Roadmap

- [x] Simple auto-encoder
- [x] A script to visualize latent feature-space (style-space).
- [x] Adversarial auto-encoder to fit the style-space into a Gaussian distribution.
- [ ] A script to generate images of all digits from a random style-vector.

## Setup

```
$ python3 -m venv pyenv
$ source pyenv/bin/activate
$ pip install .
```

## Usage

```
$ train-sae --help
[...]
```

```
$ train-aae --feature-size 4
[...]
[Epoch 1 Batch 100] Training: mse=0.0671
[...]
Model parameters and trainer state saved.
```

## Feature Space Visualisation

If you're interested in visualizing the feature space which your trained
encoder maps to, there's a script to do just that.

For that, we first need to install the visualisation requirements.

```
$ pip install -r requirements-vis.txt
```

Executing the following script creates an encoder model with saved parameters,
runs it on 500 test images (by default), and displays the feature space as a
scatter matrix.

```
$ encoder-vis --feature-size 4
```

The following image is the output from an auto-encoder model that was trained
with the simple approach ([`mnist-sae.py`](./mnist-sae.py)).

[![sae-sample](https://i.imgur.com/fBaF6tcl.png)](https://i.imgur.com/fBaF6tc.png)

_(click to enlarge)_

And below is the same from an adversarially trained auto-encoder model
([`mnist-aae.py`](./mnist-aae.py)).

[![aae-sample](https://i.imgur.com/pI3iQyBl.png)](https://i.imgur.com/pI3iQyB.png)

_(click to enlarge)_
