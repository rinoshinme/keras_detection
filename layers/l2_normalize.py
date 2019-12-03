import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np


class L2Normalization(Layer):
    def __init__(self, gamma_init=20, **kwargs):
        self.gamma_init = gamma_init
        if K.image_data_format() == 'channels_last':
            self.axis = 3
        else:
            self.axis = 1
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # define weights in this layer
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis], ))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, **kwargs):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma

    def get_config(self):
        new_config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(L2Normalization, self).get_config()
        
        config = {}
        for k, v in base_config.items():
            config[k] = v
        for k, v in new_config.items():
            config[k] = v
        return config


def test_l2normalize():
    norm = L2Normalization()
    print(norm.get_config())


if __name__ == '__main__':
    test_l2normalize()
