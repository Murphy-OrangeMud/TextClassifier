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
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Concatenate
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import keras.optimizers

from nltk import tokenize

MAX_SENT_LENGTH = 100
MAX_SENTS = 10
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
data_test = pd.read_csv(test_path, delimiter=',', encoding='latin-1', header=None,
                         names=['category', 'id', 'time', 'query', 'user', 'text'])

texts_train = []
tweets_train = []
labels_train = []

texts_test = []
tweets_test = []
labels_test = []

for idx in range(data_train.text.shape[0]):
	text_train = data_train.text[idx]
	text_train = clean_str(text_train)
	texts_train.append(text_train)
	
	labels_train.append(data_train.category[idx])
	
	sentences_train_tmp = tokenize.sent_tokenize(text_train)
	sentences_train = []
	for sentence in sentences_train_tmp:
		sentence = tokenize.word_tokenize(sentence)
		sentences_train.append(sentence)
	MAX_SENTS = max(MAX_SENTS, len(sentences_train))
	tweets_train.append(sentences_train)
	

for idx in range(data_test.text.shape[0]):
	text_test = data_test.text[idx]
	text_test = clean_str(text_test)
	texts_test.append(text_test)
	
	labels_test.append(data_test.category[idx])

	sentences_test_tmp = tokenize.sent_tokenize(text_test)
	sentences_test = []
	for sentence in sentences_test_tmp:
		sentence = tokenize.word_tokenize(sentence)
		sentences_test.append(sentence)
	MAX_SENTS = max(MAX_SENTS, len(sentences_test))
	tweets_test.append(sentences_test)


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train)

x_train = np.zeros((len(texts_train), MAX_SENTS, MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')
x_test = np.zeros((len(texts_test), MAX_SENTS, MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')

for i, tweet in enumerate(tweets_train):
	for j, sentence in enumerate(tweet):
		for l, word in enumerate(sentence):
			if l < MAX_SENT_LENGTH:
				charTokens = list(word)
				k = 0
				for _, char in enumerate(charTokens):
					if k < MAX_WORD_LENGTH:
						x_train[i, j, l, k] = alphabet.index(char)
						k += 1
						
for i, tweet in enumerate(tweets_test):
	for j, sentence in enumerate(tweet):
		for l, word in enumerate(sentence):
			if l < MAX_SENT_LENGTH:
				charTokens = list(word)
				k = 0
				for _, char in enumerate(charTokens):
					if k < MAX_WORD_LENGTH:
						x_test[i, j, l, k] = alphabet.index(char)
						k += 1

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

y_train = to_categorical(np.asarray(labels_train))
y_test = to_categorical(np.asarray(labels_test))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

train_indices = np.arange(x_train.shape[0])
np.random.shuffle(train_indices)
x_train = x_train[train_indices]
y_train = y_train[train_indices]

test_indices = np.arange(x_train.shape[0])
np.random.shuffle(test_indices)
x_test = x_test[test_indices]
y_test = y_test[test_indices]

print('Number of positive and negative tweets in traing and validation set')
print y_train.sum(axis=0)
print y_test.sum(axis=0)

print(x_train.shape)
print(y_train.shape)

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

embedding_char_layer = Embedding(len(alphabet),
                            EMBEDDING_DIM,
                            input_length=MAX_WORD_LENGTH,
                            trainable=True,
                            mask_zero=True)

# char level embedding with attention
word_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int32')
embedded_sequences = embedding_char_layer(word_input)
print(embedded_sequences.shape)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer(100)(l_lstm)
wordEncoder = Model(word_input, l_att)

tweet_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int32')
tweet_encoder = TimeDistributed(sentEncoder)(tweet_input)
l_lstm_tweet = Bidirectional(GRU(100, return_sequences=True))(tweet_encoder)
l_att_tweet = AttLayer(100)(l_lstm_tweet)
preds = Dense(5, activation='softmax')(l_att_tweet)  # modified: 2->5
model = Model(tweet_input, preds)

rmsprop = keras.optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=K.epsilon(), clipvalue=1., decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=10, batch_size=50)
