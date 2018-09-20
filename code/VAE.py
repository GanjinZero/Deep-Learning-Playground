# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:25:05 2018

@author: GanJinZERO
@Description: Variational Autoencoder
"""

from load_data import load_data_anime_face
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.models import Model
from keras import backend as K


def sample(args):
    # Use log_sigma to make it into -inf to inf, not 0 to inf
    z_mu, z_log_sigma2 = args
    batch = K.shape(z_mu)[0]
    dim = K.int_shape(z_mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mu + K.exp(0.5 * z_log_sigma2) * epsilon

class Vae:
    def __init__(self, pic_nb=100):
        self.train_data = load_data_anime_face(pic_nb) / 255
        self.model = self._build_model()
        
    def _build_model(self):
        input_shape = (96, 96, 3)
        batch_size = 16
        latent_dim = 10
        epochs = 20
        
        # Encoder
        inputs = Input(shape = input_shape, name = "Encoder_input")
        x = Conv2D(filters = 16,
                   kernel_size = (3, 3),
                   activation = 'relu',
                   strides = (1, 1),
                   data_format = "channel_last")(inputs)
        x = Conv2D(filters = 32,
                   kernel_size = (3, 3),
                   activation = 'relu',
                   strides = (1, 1),
                   data_format = "channel_last")(x)
        x = Flatten()(x)
        x = Dense(16, activation = 'relu')(x)
        
        z_mean = Dense(latent_dim, name = 'z_mean')(x)
        z_log_var = Dense(latent_dim, name = 'z_log_var')(x)
        z = Lambda(sample,
                   output_shape=(latent_dim,),
                   name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        
        # Decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        
        return model
    
    def train_model(self):
        
    def plot_random_result(self):
        return
        
if __name__ == "__main__":
    vae = Vae()
    vae.train_model()