#!/bin/env python3

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import os

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd

from encoder import ImgEncoder, encode
from decoder import ImgDecoder, decode
from util import restore_block, save_images


def main():
    parser = argparse.ArgumentParser(description='MNIST Simple Auto-Encoder')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate with adam optimizer (default: 0.005)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='dimensions of the latent feature vector (default: 8)')
    parser.add_argument('--state-prefix', default='mnist', metavar='pre',
                        help='path-prefix of state files (default: mnist) ' +
                             'state files will be of the form "prefixN.key.params"')
    opt = parser.parse_args()

    # data
    def transformer(image, label):
        image = image.reshape((1,28,28)).astype(np.float32)/255
        return image, mx.nd.one_hot(mx.nd.array([label]), 10)[0]

    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=True, transform=transformer),
        batch_size=opt.batch_size, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=False, transform=transformer),
        batch_size=opt.batch_size, shuffle=False)

    # network
    enc = ImgEncoder(opt.feature_size)
    dec = ImgDecoder()

    save_paths = dict([(key, '{}{}.{}.params'.format(opt.state_prefix, opt.feature_size, key))
                        for key in ['enc', 'dec', 'enc_tr', 'dec_tr']])

    train(enc, dec, train_data, test_data, save_paths,
          lr=opt.lr, epochs=opt.epochs)


def train(enc, dec, train_data, test_data, save_paths, lr=0.01, epochs=40):
    # computation context
    ctx = mx.cpu()

    # initialize parameters
    xavinit = mx.init.Xavier(magnitude=2.24)
    init_block(enc, xavinit, save_paths['enc'], ctx)
    init_block(dec, xavinit, save_paths['dec'], ctx)

    enc_trainer = gluon.Trainer(enc.collect_params(), 'adam', {'learning_rate': lr})
    dec_trainer = gluon.Trainer(dec.collect_params(), 'adam', {'learning_rate': lr})

    # try to restore trainer states
    restore_trainer(enc_trainer, save_paths['enc_tr'])
    restore_trainer(dec_trainer, save_paths['dec_tr'])

    loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    metric = mx.metric.MSE()

    for epoch in range(epochs):
        metric.reset()
        for i, (images, labels) in enumerate(train_data):
            images = images.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            # record computation graph for differentiating with backward()
            with autograd.record():
                features = encode(enc, images, labels)
                images_out = decode(dec, features, labels)
                L = loss(images_out, images)
                L.backward()
            # weights train step
            batch_size = images.shape[0]
            enc_trainer.step(batch_size)
            dec_trainer.step(batch_size)

            metric.update([images], [images_out])

            if (i+1) % 100 == 0:
                name, mse = metric.get()
                print('[Epoch {} Batch {}] Training: {}={:.4f}'.format(epoch+1, i+1, name, mse))

        name, mse = metric.get()
        print('[Epoch {}] Training: {}={:.4f}'.format(epoch+1, name, mse))

        name, test_mse = test(ctx, enc, dec, test_data)
        print('[Epoch {}] Validation: {}={:.4f}'.format(epoch+1, name, test_mse))

        # save states and parameters
        enc.save_params(save_paths['enc'])
        dec.save_params(save_paths['dec'])
        enc_trainer.save_states(save_paths['enc_tr'])
        dec_trainer.save_states(save_paths['dec_tr'])
        print('Model parameters and trainer state saved to:')
        print('  {}  {}\n  {}  {}'.format(save_paths['enc'], save_paths['enc_tr'],
                                          save_paths['dec'], save_paths['dec_tr']))


def init_block(block, initializer, param_file, ctx):
    if not restore_block(block, param_file, ctx):
        block.initialize(initializer, ctx)


def restore_trainer(trainer, state_file):
    if os.path.isfile(state_file):
        trainer.load_states(state_file)


test_idx = 1
def test(ctx, enc, dec, test_data):
    global test_idx
    metric = mx.metric.MSE()
    samples = []
    for images, labels in test_data:
        features = encode(enc, images, labels)
        images_out = decode(dec, features, labels)
        metric.update([images], [images_out])

        idx = np.random.randint(images.shape[0])
        samples.append(mx.nd.concat(images[idx], images_out[idx], dim=2)[0].asnumpy())

    try:
        imgdir = '/tmp/mnist'
        save_images(samples[::2], imgdir, test_idx*1000)
        test_idx += 1
    except Exception as e:
        print("writing images failed:", e)

    return metric.get()


if __name__ == '__main__':
    main()
