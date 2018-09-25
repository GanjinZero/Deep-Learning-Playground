# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:25:05 2018

@author: GanJinZERO
@Description: Variational Autoencoder
"""

from load_data import load_data_anime_face
from keras.layers import Dense, Input, Reshape, Conv2DTranspose
from keras.layers import Conv2D, Flatten, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import mse


def sample(args):
    # Use log_sigma to make it into -inf to inf, not 0 to inf
    z_mu, z_log_sigma2 = args
    batch = K.shape(z_mu)[0]
    dim = K.int_shape(z_mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mu + K.exp(0.5 * z_log_sigma2) * epsilon

class Vae:
    def __init__(self, pic_train_nb=1000, pic_test_nb=200):
        self.train_data = load_data_anime_face(pic_train_nb) / 255
        self.test_data = load_data_anime_face(pic_test_nb) / 255
        self.input_shape = (96, 96, 3)
        self.batch_size = 64
        self.latent_dim = 10
        self.epochs = 100
        
    def build_model(self):
        # Encoder
        self.inputs = Input(shape = self.input_shape, name = "Encoder_input")
        self.x = Conv2D(filters = 16,
                   kernel_size = (3, 3),
                   activation = 'relu',
                   strides = 2,
                   padding='same',
                   data_format = "channels_last")(self.inputs)
        self.x = Conv2D(filters = 32,
                   kernel_size = (3, 3),
                   activation = 'relu',
                   strides = 2,
                   padding='same',
                   data_format = "channels_last")(self.x)
        self.shape = K.int_shape(self.x)
        self.x = Flatten()(self.x)
        self.x = Dense(16, activation = 'relu')(self.x)
        
        self.z_mean = Dense(self.latent_dim, name = 'z_mean')(self.x)
        self.z_log_var = Dense(self.latent_dim, name = 'z_log_var')(self.x)
        self.z = Lambda(sample,
                   output_shape=(self.latent_dim,),
                   name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        print(self.encoder.summary())
        
        # Decoder
        self.latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        self.x = Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(self.latent_inputs)
        self.x = Reshape((self.shape[1], self.shape[2], self.shape[3]))(self.x)
        self.x = Conv2DTranspose(filters=32,
                            kernel_size=(3, 3),
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format = "channels_last")(self.x)
        self.x = Conv2DTranspose(filters=16,
                            kernel_size=(3, 3),
                            activation='relu',
                            strides=2,
                            padding='same',
                            data_format = "channels_last")(self.x)
        self.outputs = Conv2DTranspose(filters=3,
                            kernel_size=(3, 3),
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(self.x)
        self.decoder = Model(self.latent_inputs, self.outputs, name='decoder')
        print(self.decoder.summary())

        # Instantiate VAE model
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae_model = Model(self.inputs, self.outputs, name='vae')
    
    def train_model(self):
        self.output_loss = mse(K.flatten(self.inputs), K.flatten(self.outputs))
        self.output_loss *= 96 * 96 * 3
        self.kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        self.kl_loss = K.sum(self.kl_loss, axis=-1)
        self.kl_loss *= -0.5
        self.total_loss = K.mean(self.output_loss + self.kl_loss)
        self.vae_model.add_loss(self.total_loss)
        self.vae_model.compile(optimizer='adam')
        print(self.vae_model.summary())
        self.vae_model.fit(self.train_data, 
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           validation_data=(self.test_data, None))
        
    def plot_given_z(self, z):
        return 0
        
    def plot_random_result(self):
        return 0
    
    def plot_train(self, nb=1):
        return 0
    
    def plot_test(self, nb=1):
        return 0
    
    def plot_loss(self):
        return 0
    
    def save_model(self):
        return 0
    
    def load_model(self):
        return 0
        
if __name__ == "__main__":
    vae = Vae()
    vae.build_model()
    vae.train_model()