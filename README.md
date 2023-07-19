# MNIST Style-based Auto-Encoder

An [Adversarial Auto-Encoder][aae] (AAE) model that encode style information of
MNIST images on a Gaussian distributed multivariate.

The training method implemented here is a slight variant of what's discussed in
section 5 (Semi-Supervised Adversarial Autoencoders) of [the paper][aae]. In the
paper, the encoder generated softmax is trained using another adversarial network.
In contrast, we simply use cross-entropy loss instead. We also implement a simpler
variant of the idea presented in section 2.3.

[aae]: https://arxiv.org/abs/1511.05644

## Goals

- Given an MNIST image, extract a representation of its "handwriting style".
- Be able to generate MNIST-like images, given a digit and an arbitrary style-vector.

## Strategy

For the model to learn a representation of handwriting style, we need the model to
separate the concept of a digit from the visual representation of that digit.
With an autoencoder design, this can be accomplished by having the encoder learn to
classify the digit, given an image, in addition to producing a latent vector to
encode variations in style. The class of digit and the style vector is then passed
to the decoder for reconstruction.

To be able to use the decoder as an image generator, we need the style-vector to
have a known distribution of values to sample from. Using the idea from [the paper][aae],
we could use adversarial training on the style-vector to make the encoder restrict
values to a predefined distribution, consequently making the decoder accept values
from the same range.

## Roadmap

- [x] Simple auto-encoder (without classification)
- [x] A script to visualize latent feature-space (style-space) (see `vis.ipynb`)
- [x] Adversarial training to fit the style-space into a Gaussian distribution
- [x] Make the Encoder a digit-classifier too!
- [x] Generate images of digits from a random style-vector (see `vis.ipynb`)
- [x] Simple CI pipeline
- [x] Make the discriminator digit-aware (see [#17](https://github.com/johncf/mnist-style/issues/17))
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
$ train-aae --help
usage: train-aae [-h] [--batch-size B] [--epochs E] [--lr LR] [--feat-size N] [--ckpt-dir ckpt] [--data-dir data]

MNIST Adverserial Auto-Encoder

options:
  -h, --help       show this help message and exit
  --batch-size B   batch size for training and testing (default: 64)
  --epochs E       number of epochs to train (default: 12)
  --lr LR          learning rate with adam optimizer (default: 0.0004)
  --feat-size N    dimensions of the latent feature vector (default: 4)
  --ckpt-dir ckpt  training session directory (default: ./pt-aae) for storing model parameters and trainer states
  --data-dir data  MNIST data directory (default: ./data) (gets created and downloaded to, if doesn't exist)
```

```
$ train-aae --feat-size 4 --epochs 15
[...]
Epoch 15 training:
  Mean AutoEncoder Loss: 0.1253
  Mean Classification Loss: 0.0095
  Mean Generator Loss: 0.6964 * 0.100
  Mean Discriminator Fake Loss: 0.6922
  Mean Discriminator Real Loss: 0.6924
Epoch 15 validation:
  Mean AutoEncoder Loss: 0.1252
  Mean Classification Loss: 0.0357
  Median Encoded Distribution Error: 10.6533
[time] training: 38.9s, validation: 3.8s
Done!
```

## Visualization Notebook

To analyze and visualize various aspects of the latent space of a trained auto-encoder,

1.  First install the visualisation requirements:

    ```
    $ pip install '.[cpu,vis]' --extra-index-url https://download.pytorch.org/whl/cpu
    ```
    Alternatively, install dependencies with pinned versions:
    ```
    $ pip install -r requirements-vis.txt
    ```

2.  Launch Jupyter Lab, and open `vis.ipynb` from it. Run all cells and modify variables as needed.

    ```
    $ jupyter lab
    ```

## Results and Discussion

Visualizing the style-feature encoding of MNIST test dataset using an encoder model trained with simple autoencoder training method (`train-sae` script) gives the following result:

![sae-vis](https://github.com/johncf/mnist-style/assets/21051830/ff2d089b-6869-4a87-b343-942676597ee5)

Notice that the distributions are not centered around zero, so if we try to generate images using the decoder model by sampling a random style-vector centered around zero, we get mostly garbage results:

![sae-gen](https://github.com/johncf/mnist-style/assets/21051830/c23d00b8-2e0a-46f0-8092-d03b230c23fe)

Visualizing the same using a model trained with adversarial autoencoder training method (`train-aae` script) gives the following result:

![aae2-vis](https://github.com/johncf/mnist-style/assets/21051830/9347b2de-ba56-4c3a-97ba-0d9f2f90fef1)

Notice that the distributions are now well-centered around zero, and we get much better results from the decoder model using random style-vectors centered around zero:

![aae2-gen](https://github.com/johncf/mnist-style/assets/21051830/e25ab9d7-1840-4eac-bc2f-77c9e0d92521)
