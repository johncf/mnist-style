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
from persistence import TrainingSession
from util import save_images


def main():
    parser = argparse.ArgumentParser(description='MNIST Simple Auto-Encoder')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate with adam optimizer (default: 0.005)')
    parser.add_argument('--feature-size', type=int, default=4, metavar='N',
                        help='dimensions of the latent feature vector (default: 4)')
    parser.add_argument('--ckpt-dir', default=None, metavar='ckpt',
                        help='training session directory (default: mnistNs.ckpt) ' +
                             'for storing model parameters and trainer states')
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

    ckpt_dir = opt.ckpt_dir if opt.ckpt_dir is not None \
                            else 'mnist{}s.ckpt'.format(opt.feature_size)
    sess = TrainingSession(ckpt_dir)
    sess.add_block('enc', ImgEncoder(opt.feature_size), opt.lr)
    sess.add_block('dec', ImgDecoder(), opt.lr)

    train(sess, train_data, test_data, epochs=opt.epochs)


def train(sess, train_data, test_data, epochs=40):
    ctx = mx.cpu()
    sess.init_all(ctx)

    enc = sess.get_block('enc')
    dec = sess.get_block('dec')

    enc_trainer = sess.get_trainer('enc')
    dec_trainer = sess.get_trainer('dec')

    loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    metric = mx.metric.MSE()

    for epoch in range(epochs):
        metric.reset()
        for i, (images, labels) in enumerate(train_data):
            images = images.as_in_context(ctx)
            labels = labels.as_in_context(ctx)

            batch_size = images.shape[0]

            # record computation graph for differentiating with backward()
            with autograd.record():
                features = encode(enc, images, labels)
                images_out = decode(dec, features, labels)
                L = loss(images_out, images)
                L.backward()

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

        sess.save_all()
        print('Model parameters and trainer state saved.')


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
        print("test images written to", imgdir)
    except Exception as e:
        print("writing images failed:", e)

    return metric.get()


if __name__ == '__main__':
    main()
