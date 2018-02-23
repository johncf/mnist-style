#!/bin/env python3

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import os

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd

from encoder import ImgEncoder, encode
from decoder import ImgDecoder, decode
from gaussian import GaussDiscriminator, GaussSampler
from persistence import TrainingSession
from util import save_images


def main():
    parser = argparse.ArgumentParser(description='MNIST Adverserial Auto-Encoder')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=24, metavar='E',
                        help='number of epochs to train (default: 24)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate with adam optimizer (default: 0.005)')
    parser.add_argument('--feature-size', type=int, default=4, metavar='N',
                        help='dimensions of the latent feature vector (default: 4)')
    parser.add_argument('--ckpt-dir', default=None, metavar='ckpt',
                        help='training session directory (default: mnistN.ckpt) ' +
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
    gauss_data = GaussSampler(opt.feature_size, batch_size=opt.batch_size, variance=2)

    ckpt_dir = opt.ckpt_dir if opt.ckpt_dir is not None \
                            else 'mnist{}.ckpt'.format(opt.feature_size)
    sess = TrainingSession(ckpt_dir)
    sess.add_block('enc', ImgEncoder(opt.feature_size), opt.lr)
    sess.add_block('dec', ImgDecoder(), opt.lr)
    sess.add_block('gdc', GaussDiscriminator(base_size=opt.feature_size*3//2), opt.lr*2)

    train(sess, train_data, test_data, gauss_data, epochs=opt.epochs)


def train(sess, train_data, test_data, gauss_data, epochs=40):
    ctx = mx.cpu()
    sess.init_all(ctx)

    enc = sess.get_block('enc')
    dec = sess.get_block('dec')
    gdc = sess.get_block('gdc')

    enc_trainer = sess.get_trainer('enc')
    dec_trainer = sess.get_trainer('dec')
    gdc_trainer = sess.get_trainer('gdc')

    ae_loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    gd_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ae_metric = mx.metric.MSE()
    gd_metric_real = mx.metric.Accuracy()
    gd_metric_fake = mx.metric.Accuracy()

    for epoch in range(epochs):
        ae_metric.reset()
        gd_metric_real.reset()
        gd_metric_fake.reset()
        for i, (images, labels) in enumerate(train_data):
            # prepare data batch
            images = images.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            gauss_sample = gauss_data.next().as_in_context(ctx)

            batch_size = images.shape[0]
            assert gauss_sample.shape[0] == batch_size

            # labels for gauss discriminator training
            gauss_yes = nd.ones((batch_size, 1), ctx=ctx)
            gauss_not = nd.zeros((batch_size, 1), ctx=ctx)

            # train autoencoder
            with autograd.record():
                features = encode(enc, images, labels)
                images_out = decode(dec, features, labels)
                loss_dec = ae_loss(images_out, images)
                gauss_fit = gdc(features)
                loss_gauss = gd_loss(gauss_fit, gauss_yes)
                if epoch == 0: # too early to teach gauss
                    L = loss_dec
                elif epoch < 12:
                    L = loss_dec + loss_gauss/120
                else: # ready for aggressive immersion!
                    L = loss_dec + loss_gauss/20
                L.backward()

            enc_trainer.step(batch_size)
            dec_trainer.step(batch_size)

            ae_metric.update([images], [images_out])

            # train discriminator
            with autograd.record():
                predict_real = gdc(gauss_sample)
                loss_real = gd_loss(predict_real, gauss_yes)
                predict_fake = gdc(features)
                loss_fake = gd_loss(predict_fake, gauss_not)
                L = loss_real + loss_fake
                L.backward()

            gdc_trainer.step(batch_size)

            gd_metric_real.update([gauss_yes], [predict_real])
            gd_metric_fake.update([gauss_not], [predict_fake])

            if (i+1) % 100 == 0:
                name, mse = ae_metric.get()
                print('[Epoch {} Batch {}] Training: {}={:.4f}'.format(epoch+1, i+1, name, mse))

        name, mse = ae_metric.get()
        print('[Epoch {}] Training:'.format(epoch+1))
        print('  AutoEncoder: {}={:.4f}'.format(name, mse))
        name, mse = gd_metric_real.get()
        print('  GaussDiscriminator: actual gauss detection {}={:.4f}'.format(name, mse))
        name, mse = gd_metric_fake.get()
        print('  GaussDiscriminator: feature space detection {}={:.4f}'.format(name, mse))

        print('[Epoch {}] Validation:'.format(epoch+1))
        test(ctx, enc, dec, gdc, test_data)

        sess.save_all()
        print('Model parameters and trainer state saved.')


test_idx = 1
def test(ctx, enc, dec, gdc, test_data):
    global test_idx
    ae_metric = mx.metric.MSE()
    gd_metric = mx.metric.Accuracy()
    samples = []
    for images, labels in test_data:
        gauss_yes = nd.ones((labels.shape[0], 1), ctx=ctx)

        features = encode(enc, images, labels)
        images_out = decode(dec, features, labels)
        ae_metric.update([images], [images_out])

        gauss_fit = gdc(features)
        gd_metric.update([gauss_yes], [gauss_fit])

        idx = np.random.randint(images.shape[0])
        samples.append(mx.nd.concat(images[idx], images_out[idx], dim=2)[0].asnumpy())

    name, mse = ae_metric.get()
    print('  AutoEncoder: {}={:.4f}'.format(name, mse))
    name, mse = gd_metric.get()
    print('  GaussDiscriminator: feature space satisfaction {}={:.4f}'.format(name, mse))

    try:
        imgdir = '/tmp/mnist'
        save_images(samples[::2], imgdir, test_idx*1000)
        test_idx += 1
        print("  test images written to", imgdir)
    except Exception as e:
        print("  writing images failed:", e)


if __name__ == '__main__':
    main()
