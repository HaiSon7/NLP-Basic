import json
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = r"Sarcam/sarcam.json"

with open(path,'r',encoding= 'utf-8') as f:
    data = json.load(f)

dataset = [x['headline'] for x in data]
label_dataset = [x['is_sarcastic'] for x in data]

dataset = np.array(dataset)
label_dataset = np.array(label_dataset)

train_size = 0.8
size = int(len(dataset) * train_size)

train_sentence = dataset[:size]
test_sentence = dataset[size:]

train_label = label_dataset[:size]
test_label = label_dataset[size:]

vocab_size = 3000
tokenizer = Tokenizer(num_words = vocab_size,oov_token = '<UNK>')
tokenizer.fit_on_texts(train_sentence)

vocab_size = len(tokenizer.word_index)
embedding_size = 64
max_length = 25

train_sequences = tokenizer.texts_to_sequences(train_sentence)
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, truncating="post", padding="post")
test_sequences = tokenizer.texts_to_sequences(test_sentence)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, truncating="post", padding="post")


print(train_label.shape)
print(test_label.shape)
print(padded_train_sequences.shape)
print(padded_test_sequences.shape)

class LSTM(tf.keras.layers.Layer):
    def __init__(self,units,input_shape):
        super(LSTM, self).__init__()
        self.units = units
        self.inp_shape = input_shape
        self.W = self.add_weight(name = "W",shape = (4,self.units,self.inp_shape))
        self.U = self.add_weight(name = "U",shape = (4,self.units,self.units))
    def call(self,pre_layer,x):
        pre_h, pre_c = tf.unstack(pre_layer)
        # print(x.shape)
        # print(self.W.shape)
        # print(self.U.shape)
        # print(pre_h.shape)
        f_t = tf.sigmoid(tf.matmul(x, tf.transpose(self.W[0]) ) + tf.matmul(pre_h, self.U[0]))
        i_t = tf.sigmoid(tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, self.U[1]))
        n_c_t = tf.tanh(tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(pre_h, self.U[2]))
        o_t = tf.sigmoid(tf.matmul(x, tf.transpose(self.W[3])) + tf.matmul(pre_h, self.U[3]))

        c_t = tf.multiply(f_t,pre_c) + tf.multiply(i_t,n_c_t)

        h_t = tf.multiply(o_t,tf.tanh(c_t))

        return tf.stack([h_t,c_t])

class Custom_model(tf.keras.Model):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(Custom_model, self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length)

        self.lstm = LSTM(self.units, embedding_size)

        self.classfication_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, sentence):
        batch_size = tf.shape(sentence)[0]
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))

        embedded_sentence = self.embedding(sentence)

        for word in tf.unstack(embedded_sentence, axis=1):
            lstm_out = self.lstm(tf.stack([h, c]), word)
            h, c = tf.unstack(lstm_out)

        return self.classfication_model(h)


units = 128

embedding_size = 100
vocab_size = len(tokenizer.index_word) + 1
input_length = max_length

model = Custom_model(units,embedding_size,vocab_size,input_length)
model.compile(
    tf.keras.optimizers.Adam(0.0005) , loss='binary_crossentropy', metrics=['acc']
)

model.fit(padded_train_sequences, train_label, validation_data=(padded_test_sequences, test_label), batch_size=32, epochs=10)

model.save('my_model.keras')
