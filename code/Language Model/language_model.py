# @author: GanjinZero
# @time: 2018/12/25
# @description: language_model.py is a neural network based Japanese language model.


import numpy as np
from utils import load_text, load_embedding, make_word_dictionary
from utils import clear_dictionary, mecab_to_text, generate_train
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, CuDNNLSTM
from keras.layers import Lambda, Add
from keras.models import Model
import keras
from sklearn.model_selection import train_test_split
from LossHistory import LossHistory
from keras.utils import np_utils
import random


embeddings_index = load_embedding()

use_text_length = 20000
japanese_text = load_text(use_text_length)
split_japanese_text = mecab_to_text(japanese_text)
dictionary = make_word_dictionary(split_japanese_text, lower_bound=10)
dictionary = clear_dictionary(dictionary, embeddings_index)

#all_embs = np.stack(embeddings_index.values())
#emb_mean, emb_std = all_embs.mean(), all_embs.std()

## Tokenize the sentences
tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(dictionary)
"""
with open("..\result\tokenizer.pkl", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
    
japanese_text_seq = tokenizer.texts_to_sequences(split_japanese_text)

word_index = tokenizer.word_index
nb_words = len(word_index) + 2
start_index = 0
end_index = len(word_index) + 1
embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
emb_mean, emb_std = embedding_matrix.mean(), embedding_matrix.std()
embedding_matrix[0] = np.random.normal(emb_mean, emb_std, (300))
embedding_matrix[-1] = np.random.normal(emb_mean, emb_std, (300))
# del embeddings_index
    
window = 5
train_x, train_y = generate_train(window, end_index, japanese_text_seq)

train_y_cat = np_utils.to_categorical(train_y)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y_cat, test_size=0.2)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Model
inp = Input(shape=(window - 1,))
x = Embedding(nb_words, 300, trainable = True, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Dropout(0.1)(x)
x = Bidirectional(CuDNNLSTM(128, return_sequences=False))(x)
x = Dropout(0.1)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(nb_words, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = LossHistory()

epoch_nb = 80 # 40 is enough
batch = 64

model.fit(x_train, y_train, batch_size=batch, epochs=epoch_nb, verbose=1,
          validation_data=(x_test, y_test), callbacks=[history])

reverse_index = {v: k for k, v in word_index.items()}
reverse_index[nb_words - 1] = ""
reverse_index[0] = ""
word_ind = np.linspace(0, nb_words - 1, nb_words)

def predict_random_sentence(new=[0] * (window - 1)):
    sentence = reverse_index[new[0]] + reverse_index[new[1]] + reverse_index[new[2]] + reverse_index[new[3]]
    while new[-1] != end_index:
        prob = model.predict(np.asarray([new]))[0]
        new_predict = int(random.choices(word_ind, weights=prob)[0])
        sentence += reverse_index[new_predict]
        new = new[1:] + [new_predict]
    return sentence

predict_random_sentence([0,0,0,0])
