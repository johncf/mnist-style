#!/bin/env python3

import argparse

import numpy as np
import mxnet as mx
from mxnet import gluon
import matplotlib as mpl; mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from encoder import ImgEncoder, encode
from util import restore_block


def main():
    parser = argparse.ArgumentParser(description='Visualize encoder feature-vector distribution')
    parser.add_argument('--sample-size', type=int, default=500, metavar='S',
                        help='sample size for plotting (default: 500)')
    parser.add_argument('--feature-size', type=int, default=8, metavar='N',
                        help='rank of the latent feature vector (default: 8)')
    parser.add_argument('--state-prefix', default='mnist', metavar='pre',
                        help='path-prefix of state files (default: mnist) ' +
                             'state files will be of the form "prefixN.key.params"')
    opt = parser.parse_args()

    def transformer(data, label):
        data = data.reshape((1,28,28)).astype(np.float32)/255
        return data, label, mx.nd.one_hot(mx.nd.array([label]), 10)[0]

    test_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=False, transform=transformer),
        batch_size=opt.sample_size, shuffle=True)

    enc = ImgEncoder(opt.feature_size)
    param_file = '{}{}.enc.params'.format(opt.state_prefix, opt.feature_size)

    ctx = mx.cpu()

    if not restore_block(enc, param_file, ctx):
        raise FileNotFoundError("parameter file {} not found".format(param_file))

    for data, labels, labels_1h in test_data:
        data_ctx = data.as_in_context(ctx)
        labels_ctx = labels_1h.as_in_context(ctx)
        features = encode(enc, data_ctx, labels_ctx).asnumpy()
        labels = labels.asnumpy()
        columns = ['f' + chr(i + ord('a')) for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=columns)
        df = df.assign(digit=labels)
        g = sns.pairplot(df, hue="digit")
        plt.show()
        break


if __name__ == '__main__':
    main()
