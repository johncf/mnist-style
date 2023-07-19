# MNIST Style-based Auto-Encoder

An [Adversarial Auto-Encoder][aae] (AAE) model that encode style information of
MNIST images on a Gaussian distributed multivariate.

The training method implemented here is a slight variant of what's discussed in
section 5 (Semi-Supervised Adversarial Autoencoders) of [the paper][aae]. In the
paper, the encoder generated softmax is trained using another adversarial network.
In contrast, we simply use cross-entropy loss instead.

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

Visualizing the style-feature-space of MNIST test dataset using an autoencoder model trained with simple training method (`train-sae` script) gives the following result:

![sae-vis](https://github.com/johncf/mnist-style/assets/21051830/ff2d089b-6869-4a87-b343-942676597ee5)

Notice that the distributions are not centered around zero, so if we try to generate images by sampling a random style-vector centered around zero,
we get mostly garbage results:

![sae-gen](https://github.com/johncf/mnist-style/assets/21051830/c23d00b8-2e0a-46f0-8092-d03b230c23fe)

Visualizing the same using an autoencoder model trained with adversarial training method (`train-aae` script) gives the following result:

![aae-vis](https://github.com/johncf/mnist-style/assets/21051830/7db1b9fb-b7c3-4e0a-8d0e-f21b895bff53)

Notice that the distributions are now well-centered around zero, and we get much better results from random style-vectors centered around zero:

![aae-gen](https://github.com/johncf/mnist-style/assets/21051830/161a3208-bee2-40f8-a118-19c7f428248c)

However, note that some digits are not constructed well even now. This is because, even though the overall distribution for each feature-component is nicely centered around zero, if we look at it separately for each digit, some of them are still skewed. This is (likely) because the discriminator is digit-agnostic, and thus can't enforce the distribution in a per-digit manner.

## Future Work

Make the discriminator digit-aware (basically using a simpler variant of the idea from section 2.3 of [the paper][aae]).
When training the discriminator:
- "Fake" inputs should be the one-hot representation of the label + the Encoder's style encoding output.
- "Real" inputs should be a random one-hot representation + a prior-distribution random-sampled style vector.
