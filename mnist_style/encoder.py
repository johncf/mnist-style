import mxnet as mx
from mxnet.gluon import nn

# Convenience type; cannot be used as a normal HybridBlock
class ImgEncoder(nn.HybridBlock):
    def __init__(self, feature_size=8, **kwargs):
        super(ImgEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.part1 = _Part1()
            self.part2 = _Part2(feature_size)

    def hybrid_forward(self, F, x):
        raise NotImplementedError("Use encode() method instead.")

def encode(enc, images, labels):
    x = enc.part1(images)
    if isinstance(labels, mx.nd.NDArray):
        x = mx.nd.concat(labels, x)
    elif isinstance(labels, mx.sym.Symbol):
        x = mx.sym.concat(labels, x)
    else:
        raise TypeError("Incompatible type: " + str(type(labels)))
    return enc.part2(x)

class _Part1(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(_Part1, self).__init__(**kwargs)
        with self.name_scope():
            self._layers = []
            self._add_layer(nn.Conv2D(channels=4, kernel_size=5, activation='relu'))
            self._add_layer(nn.Conv2D(channels=8, kernel_size=3, activation='relu'))
            self._add_layer(nn.MaxPool2D(pool_size=2, strides=2))
            self._add_layer(nn.Conv2D(channels=16, kernel_size=5, activation='relu'))
            self._add_layer(nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
            self._add_layer(nn.MaxPool2D(pool_size=2, strides=2))
            self._add_layer(nn.Flatten())

    def _add_layer(self, block):
        self._layers.append(block)
        self.register_child(block)

    def hybrid_forward(self, F, x):
        for layer in self._layers:
            x = layer(x)
        return x

class _Part2(nn.HybridBlock):
    def __init__(self, feature_size=8, **kwargs):
        super(_Part2, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(512, activation='relu')
            self.dense1 = nn.Dense(feature_size)

    def hybrid_forward(self, F, x):
        x = self.dense0(x)
        return self.dense1(x)
