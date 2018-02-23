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
from util import restore_block, save_images


def main():
    parser = argparse.ArgumentParser(description='MNIST Adverserial Auto-Encoder')
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
    gauss_data = GaussSampler(opt.feature_size, batch_size=opt.batch_size, variance=1.6)

    # network
    enc = ImgEncoder(opt.feature_size)
    dec = ImgDecoder()
    gdc = GaussDiscriminator(base_size=opt.feature_size*3//2)

    save_paths = dict([(key, '{}{}.{}.params'.format(opt.state_prefix, opt.feature_size, key))
                        for key in ['enc', 'dec', 'gdc', 'enc_tr', 'dec_tr', 'gdc_tr']])

    train(enc, dec, gdc, train_data, test_data, gauss_data, save_paths,
          lr=opt.lr, epochs=opt.epochs)


def train(enc, dec, gdc, train_data, test_data, gauss_data, save_paths, lr=0.01, epochs=40):
    # computation context
    ctx = mx.cpu()

    # initialize parameters
    xavinit = mx.init.Xavier(magnitude=2.24)
    init_block(enc, xavinit, save_paths['enc'], ctx)
    init_block(dec, xavinit, save_paths['dec'], ctx)
    init_block(gdc, xavinit, save_paths['gdc'], ctx)

    enc_trainer = gluon.Trainer(enc.collect_params(), 'adam', {'learning_rate': lr})
    dec_trainer = gluon.Trainer(dec.collect_params(), 'adam', {'learning_rate': lr})
    gdc_trainer = gluon.Trainer(gdc.collect_params(), 'adam', {'learning_rate': lr*2})

    # try to restore trainer states
    restore_trainer(enc_trainer, save_paths['enc_tr'])
    restore_trainer(dec_trainer, save_paths['dec_tr'])
    restore_trainer(gdc_trainer, save_paths['gdc_tr'])

    ae_loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    gd_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ae_metric = mx.metric.MSE()
    gd_metric = mx.metric.Accuracy()

    for epoch in range(epochs):
        ae_metric.reset()
        gd_metric.reset()
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
                L = loss_dec + loss_gauss
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

            gd_metric.update([gauss_yes], [predict_real])
            gd_metric.update([gauss_not], [predict_fake])

            if (i+1) % 100 == 0:
                name, mse = ae_metric.get()
                print('[Epoch {} Batch {}] Training: {}={:.4f}'.format(epoch+1, i+1, name, mse))

        name, mse = ae_metric.get()
        print('[Epoch {}] Training:\n  AutoEncoder: {}={:.4f}'.format(epoch+1, name, mse))
        name, mse = gd_metric.get()
        print('  GaussDiscriminator: {}={:.4f}'.format(name, mse))

        name, test_mse = test(ctx, enc, dec, test_data)
        print('[Epoch {}] Validation: {}={:.4f}'.format(epoch+1, name, test_mse))

        # save states and parameters
        enc.save_params(save_paths['enc'])
        dec.save_params(save_paths['dec'])
        gdc.save_params(save_paths['gdc'])
        enc_trainer.save_states(save_paths['enc_tr'])
        dec_trainer.save_states(save_paths['dec_tr'])
        gdc_trainer.save_states(save_paths['gdc_tr'])
        print('Model parameters and trainer state saved to:')
        print('\n'.join(['  {}  {}']*3).format(save_paths['enc'], save_paths['enc_tr'],
                                               save_paths['dec'], save_paths['dec_tr'],
                                               save_paths['gdc'], save_paths['gdc_tr']))


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
