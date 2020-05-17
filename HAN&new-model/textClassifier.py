# author - Zhiyi Cheng/Murphy-Orangemud
# May 17 2020
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from nltk import tokenize

MAX_SENT_LENGTH = 100
MAX_WORD_LENGTH = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
alphabet_size = 69


def clean_str(string):
	string = "".join(list(filter(lambda ch: ch in alphabet, string)))
	return string.strip().lower()


train_path = "./trainingandtestdata/training.1600000.processed.noemoticon.csv"
test_path = "./trainingandtestdata/testdata.manual.2009.06.14.csv"

data_train = pd.read_csv(test_path, delimiter=',', encoding='latin-1', header=None,
                         names=['category', 'id', 'time', 'query', 'user', 'text'])

print("data read")

texts = []
tweets = []
labels = []

for idx in range(data_train.text.shape[0]):
	text = data_train.text[idx]
	text = clean_str(text)
	texts.append(text)
	
	labels.append(data_train.category[idx])
	
	tweet = tokenize.word_tokenize(text)
	MAX_SENT_LENGTH = max(MAX_SENT_LENGTH, len(tweet))
	tweets.append(tweet)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')

for i, tweet in enumerate(tweets):
	for j, word in enumerate(tweet):
		if j < MAX_SENT_LENGTH:
			charTokens = list(word)
			k = 0
			for _, char in enumerate(charTokens):
				if k < MAX_WORD_LENGTH:
					data[i, j, k] = alphabet.index(char)
					k += 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print(data_train.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative tweets in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

# temporarily there are no fittable pre-trained model for character embedding
embedding_layer = Embedding(len(alphabet) - 1,
                            EMBEDDING_DIM,
                            input_length=MAX_WORD_LENGTH,
                            trainable=True,
                            mask_zero=True)

class AttLayer(Layer):
	def __init__(self, attention_dim):
		self.init = initializers.get('normal')
		self.supports_masking = True
		self.attention_dim = attention_dim
		super(AttLayer, self).__init__()
	
	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = K.variable(self.init((input_shape[-1], self.attention_dim)), name='W')
		self.b = K.variable(self.init((self.attention_dim,)), name='b')
		self.u = K.variable(self.init((self.attention_dim, 1)), name='u')
		self.trainable_weights = [self.W, self.b, self.u]
		super(AttLayer, self).build(input_shape)
	
	def compute_mask(self, inputs, mask=None):
		return mask
	
	def call(self, x, mask=None):
		# size of x :[batch_size, sel_len, attention_dim]
		# size of u :[batch_size, attention_dim]
		# uit = tanh(xW+b)
		uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
		ait = K.dot(uit, self.u)
		ait = K.squeeze(ait, -1)
		
		ait = K.exp(ait)
		
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			ait *= K.cast(mask, K.floatx())
		ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		ait = K.expand_dims(ait)
		weighted_input = x * ait
		output = K.sum(weighted_input, axis=1)
		
		return output
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])


word_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(word_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
sentEncoder = Model(word_input, l_att)

tweet_input = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')
tweet_encoder = TimeDistributed(sentEncoder)(tweet_input)
l_lstm_tweet = Bidirectional(GRU(100, return_sequences=True))(tweet_encoder)
l_att_tweet = AttLayer(100)(l_lstm_tweet)
preds = Dense(5, activation='softmax')(l_att_tweet)
model = Model(tweet_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)
