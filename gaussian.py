from mxnet.gluon import nn
from mxnet import nd

class GaussDiscriminator(nn.HybridBlock):
    def __init__(self, base_size=10, **kwargs):
        super(GaussDiscriminator, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(base_size*4, activation='tanh')
            self.dense1 = nn.Dense(base_size*2, activation='tanh')
            self.dense2 = nn.Dense(base_size, activation='tanh')
            self.dense3 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

class GaussSampler:
    def __init__(self, feature_size, batch_size, variance=1):
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.variance = variance

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return nd.random_normal(shape=(self.batch_size, self.feature_size))*self.variance
