import mxnet as mx
from mxnet.gluon import nn

class ImgEncoderPart1(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ImgEncoderPart1, self).__init__(**kwargs)
        with self.name_scope():
            self.layers = []
            self._add_layer(nn.Conv2D(channels=4, kernel_size=5, activation='relu'))
            self._add_layer(nn.Conv2D(channels=8, kernel_size=3, activation='relu'))
            self._add_layer(nn.MaxPool2D(pool_size=2, strides=2))
            self._add_layer(nn.Conv2D(channels=16, kernel_size=5, activation='relu'))
            self._add_layer(nn.Conv2D(channels=32, kernel_size=3, activation='relu'))
            self._add_layer(nn.MaxPool2D(pool_size=2, strides=2))
            self._add_layer(nn.Flatten())

    def _add_layer(self, block):
        self.layers.append(block)
        self.register_child(block)

    def hybrid_forward(self, F, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ImgEncoderPart2(nn.HybridBlock):
    def __init__(self, feature_size=8, **kwargs):
        super(ImgEncoderPart2, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(512, activation='relu')
            self.dense1 = nn.Dense(feature_size)

    def hybrid_forward(self, F, x):
        x = self.dense0(x)
        return self.dense1(x)

def encode(enc1, enc2, images, labels):
    x = enc1(images)
    if isinstance(labels, mx.nd.NDArray):
        x = mx.nd.concat(labels, x)
    elif isinstance(labels, mx.sym.Symbol):
        x = mx.sym.concat(labels, x)
    else:
        raise TypeError("Incompatible type: " + str(type(labels)))
    return enc2(x)
