import mxnet as mx
from mxnet.gluon import nn

class ImgDecoder(nn.HybridBlock):
    def __init__(self, img_size=784, **kwargs):
        super(ImgDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = []
            self._add_layer(nn.Dense(64, activation='relu'))
            self._add_layer(nn.Dense(128, activation='relu'))
            self._add_layer(nn.Dense(256, activation='relu'))
            self._add_layer(nn.Dense(img_size, activation='sigmoid'))

    def _add_layer(self, block):
        self.layers.append(block)
        self.register_child(block)

    def hybrid_forward(self, F, x):
        for layer in self.layers:
            x = layer(x)
        return x

def decode(dec, features, labels, shape=(28,28)):
    if isinstance(labels, mx.nd.NDArray):
        x = mx.nd.concat(features, labels)
    elif isinstance(labels, mx.sym.Symbol):
        x = mx.sym.concat(features, labels)
    else:
        raise TypeError("Incompatible type: " + str(type(labels)))
    return dec(x).reshape((-1, 1, *shape))
