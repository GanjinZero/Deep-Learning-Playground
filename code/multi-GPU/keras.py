import random
import os
import keras
import time
from keras.layeres import Input, Dense
from keras.models import Model
from keras.utils import multi_gpu_model


def generate_data(n):
    """
    :n: number of train_data
    """
    x = []
    y = []
    for i in range(n):
        x_new = random.random() * 10
        y_new = x_new + random.random()
        x.append(x_new)
        y.append(y_new)

    return x, y


def set_model():
    inp = Input(shape=())
    x = Dense(1, activation="relu")
    model = Model(inputs=inp, outputs=x)
    return model


def gpu_state():
    os.system("nvidia-smi")


def double_gpu_model():
    parallel_model = multi_gpu_model(set_model(), gpus=2)
    return parallel_model


def train_model(model, x, y):
    start = time.time()
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=opt)
    print(model.summary())
    model.fit(x, y, batch_size=32, epochs=100)
    end = time.time()
    print("Used time %s"%round((end - start), 2))


if __name__ == "__main__":
    x, y = generate_data(1000)
    train_model(set_model(), x, y)

    gpu_state()
    train_model(double_gpu_model(), x, y)
