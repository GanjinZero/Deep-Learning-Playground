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
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mse
import matplotlib.pyplot as pyplot
import numpy as np
from loss_history import LossHistory


def sample(args):
    # Use log_sigma to make it into -inf to inf, not 0 to inf
    z_mu, z_log_sigma2 = args
    batch = K.shape(z_mu)[0]
    dim = K.int_shape(z_mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mu + K.exp(0.5 * z_log_sigma2) * epsilon

def plot(picture):
    pic = picture * 255
    pic = picture.reshape((96, 96, 3))
    pyplot.imshow(pic)
    pyplot.axis('off')
    pyplot.show()

class Vae:
    def __init__(self, pic_train_nb=5000, pic_test_nb=100):
        self.train_data = load_data_anime_face(pic_train_nb) / 255
        self.test_data = load_data_anime_face(pic_test_nb) / 255
        self.input_shape = (96, 96, 3)
        self.batch_size = 512
        self.latent_dim = 50
        self.epochs = 200
        self.learning_rate = 0.01
        
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
        self.adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.vae_model.compile(optimizer=self.adam)
        print(self.vae_model.summary())
        self.history = LossHistory()
        self.vae_model.fit(self.train_data, 
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           validation_data=(self.test_data, None),
                           callbacks=[self.history])
        
    def plot_given_z(self, z):
        pict = self.decoder.predict(z)
        plot(pict)
        
    def plot_random_result(self):
        z = np.array([np.random.normal(0, 1, self.latent_dim)])
        self.plot_given_z(z)
    
    def plot_random_train(self):
        train_size = np.shape(self.train_data)[0]
        train_pic_id = np.random.randint(0, train_size)
        train_pic = np.array([self.train_data[train_pic_id]])
        plot(train_pic)
        predict_pic = self.vae_model.predict(train_pic)
        plot(predict_pic)
    
    def plot_random_test(self):
        test_size = np.shape(self.test_data)[0]
        test_pic_id = np.random.randint(0, test_size)
        test_pic = np.array([self.test_data[test_pic_id]])
        plot(test_pic)
        predict_pic = self.vae_model.predict(test_pic)
        plot(predict_pic)
    
    def plot_loss(self):
        self.history.loss_plot('epoch')
    
    def save_model(self):
        return 0
    
    def load_model(self):
        return 0
        
if __name__ == "__main__":
    vae = Vae()
    vae.build_model()
    vae.train_model()
    # vae.plot_given_z(np.array([[0,0,0,0,0]]))
    # vae.plot_random_result()
    vae.plot_random_train()
    vae.plot_random_test()