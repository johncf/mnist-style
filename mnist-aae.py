#!/bin/env python3

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd

from encoder import ImgEncoderPart1, ImgEncoderPart2, encode
from decoder import ImgDecoder, decode


def save_images(images, imgdir, startid=1, nwidth=6):
    import os
    from PIL import Image
    os.makedirs(imgdir, exist_ok=True)
    global img_idx
    for img in images:
        img = Image.fromarray(img*255)
        img.convert('L').save(os.path.join(imgdir, str(startid).zfill(nwidth) + ".png"))
        startid += 1


def main():
    parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Autoencoder')
    parser.add_argument('--batch-size', type=int, default=100, metavar='B',
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate with adam optimizer (default: 0.005)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='rank of the latent feature vector (default: 8)')
    parser.add_argument('--param-prefix', default='mnist', metavar='pre',
                        help='name-prefix of weight files (default: mnist)')
    opt = parser.parse_args()

    # network
    enc1 = ImgEncoderPart1()
    enc2 = ImgEncoderPart2(opt.feature_size)
    dec = ImgDecoder()

    # data
    def transformer(data, label):
        data = data.reshape((1,28,28)).astype(np.float32)/255
        return data, mx.nd.one_hot(mx.nd.array([label]), 10)[0]

    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=True, transform=transformer),
        batch_size=opt.batch_size, shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=False, transform=transformer),
        batch_size=opt.batch_size, shuffle=False)

    # train
    ctx = mx.cpu()
    train(ctx, enc1, enc2, dec, train_data, test_data,
          lr=opt.lr, epochs=opt.epochs)

    enc1.save_params(opt.param_prefix + '.enc1.params')
    enc2.save_params(opt.param_prefix + '.enc2.params')
    dec.save_params(opt.param_prefix + '.dec.params')


def train(ctx, enc1, enc2, dec, train_data, test_data, lr=0.01, epochs=40):
    enc1.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    enc2.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    dec.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    enc1_trainer = gluon.Trainer(enc1.collect_params(), 'adam', {'learning_rate': lr})
    enc2_trainer = gluon.Trainer(enc2.collect_params(), 'adam', {'learning_rate': lr})
    dec_trainer = gluon.Trainer(dec.collect_params(), 'adam', {'learning_rate': lr})

    loss = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    metric = mx.metric.MSE()

    for epoch in range(epochs):
        metric.reset()
        for i, (data, labels) in enumerate(train_data):
            data = data.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            # record computation graph for differentiating with backward()
            with autograd.record():
                features = encode(enc1, enc2, data, labels)
                data_out = decode(dec, features, labels)
                L = loss(data_out, data)
                L.backward()
            # weights train step
            batch_size = data.shape[0]
            enc1_trainer.step(batch_size)
            enc2_trainer.step(batch_size)
            dec_trainer.step(batch_size)

            metric.update([data], [data_out])

            if (i+1) % 100 == 0:
                name, mse = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i+1, name, mse))

        name, mse = metric.get()
        print('[Epoch %d] Training: %s=%f'%(epoch, name, mse))

        name, test_mse = test(ctx, enc1, enc2, dec, test_data)
        print('[Epoch %d] Validation: %s=%f'%(epoch, name, test_mse))


test_idx = 1
def test(ctx, enc1, enc2, dec, test_data):
    global test_idx
    metric = mx.metric.MSE()
    images = []
    for data, labels in test_data:
        features = encode(enc1, enc2, data, labels)
        data_out = decode(dec, features, labels)
        metric.update([data], [data_out])

        idx = np.random.randint(data.shape[0])
        images.append(mx.nd.concat(data[idx], data_out[idx], dim=2)[0].asnumpy())

    try:
        imgdir = '/tmp/mnist'
        save_images(images, imgdir, test_idx*1000)
        test_idx += 1
        print(len(images), "test images written to", imgdir)
    except Exception as e:
        print("writing images failed:", e)

    return metric.get()


if __name__ == '__main__':
    main()
