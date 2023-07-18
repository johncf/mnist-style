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
- [x] Make Encoder a digit-classifier!
- [x] Generate images of digits from a random style-vector (see `vis.ipynb`)
- [x] Simple CI pipeline
- [ ] GPU support

## Setup

1.  Create a virtual environment:

    ```
    $ python3 -m venv pyenv
    $ source pyenv/bin/activate
    ```

2.  Install the package with one of `cpu` or `gpu` extra.
    Note: using `gpu` installs the default `torch` package from PyPI, but GPU training is not supported yet, and the package is an order of magnitude bigger than the `cpu` version.

    ```
    $ pip install '.[cpu]' --extra-index-url https://download.pytorch.org/whl/cpu
    ```

    Alternatively, you may install from `requirements.txt` (containing CPU version of `torch`):

    ```
    $ pip install -r requirements.txt
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

If you're interested in visualizing the feature space which your trained encoder maps to, there's a notebook to do that.

We first need to install the visualisation requirements.

```
$ pip install -r requirements-vis.txt
```

Open `vis.ipynb` in JupyterLab, modify parameters if needed and run all cells.

```
$ jupyter lab
```

The following image is the output from an auto-encoder model that was trained
with the simple approach (`train-sae` script).

[![sae-sample](https://i.imgur.com/fBaF6tcl.png)](https://i.imgur.com/fBaF6tc.png)

And below is the same from an adversarially trained auto-encoder model
(`mnist-aae` script).

[![aae-sample](https://i.imgur.com/pI3iQyBl.png)](https://i.imgur.com/pI3iQyB.png)
