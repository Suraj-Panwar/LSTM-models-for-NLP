
# coding: utf-8

# In[1]:


################
#Import Modules#
################

#from __future__ import print_function
import numpy as np
np.random.seed(0)
import nltk
from matplotlib import pyplot as plt
import random
from nltk.corpus import gutenberg
import tensorflow as tf
import time
import gensim
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
import pickle


# In[2]:


######################################
#Generating Sets for Gutenberg Corpus#
######################################


#Generate Train set for Gutenberg
guten_train = np.random.choice(list(gutenberg.fileids()), 12)

#Generate Dev set for Gutenberg
guten_dev_1 = [i for i in list(gutenberg.fileids()) if i not in guten_train]
guten_dev = np.random.choice(guten_dev_1, 3)

#Generate Test set for Gutenberg
guten_test = [j for j in list(guten_dev_1) if j not in guten_dev]

for i, j, k in zip(guten_train, guten_dev, guten_test):
    assert i not in guten_dev
    assert i not in guten_test
    assert j not in guten_train
    assert j not in guten_test
    assert k not in guten_dev
    assert k not in guten_train


# In[3]:


###################################
#Preprocessing Data for Tensorflow#
###################################

file = gutenberg.raw(fileids=guten_train)
y = nltk.word_tokenize(file)
str_guten = ' '.join(y)
str_garbage = "$&()*1234567890:;=>[]_`-'" 
for s in str_garbage:
    str_guten = str_guten.replace(str(s), '')
str_guten = str_guten.replace('  ', ' ')
str_guten = str_guten.replace("' '", '')
str_guten = str_guten.replace("''", '')

data = str_guten[:100000]


# In[4]:


###################
#Helper Functions#
##################


# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
    print('\n\n')
    # starting with random character
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

# method for preparing the training data
def load_data(data ,seq_length):
    fileObject = open("testfile_chars.txt",'rb') 
    chars = pickle.load(fileObject)
    VOCAB_SIZE = len(chars)

    #print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
    print('Generated Sentence: ')
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}

    X = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
    y = np.zeros((int(len(data)/seq_length), seq_length, VOCAB_SIZE))
    for i in range(0, int(len(data)/seq_length)-1):
        X_sequence = data[i*seq_length:(i+1)*seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence

        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return X, y, VOCAB_SIZE, ix_to_char


# In[5]:


MODE = 'train'
BATCH_SIZE = 50
HIDDEN_DIM = 500
SEQ_LENGTH = 100
WEIGHTS = ''
nb_epoch = 20
GENERATE_LENGTH = 60
LAYER_NUM = 2
X, y, VOCAB_SIZE, ix_to_char = load_data(data, SEQ_LENGTH)


# In[6]:


################
#Main Function#
###############


# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


if not WEIGHTS == '':
    model.load_weights(WEIGHTS)
    nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
    nb_epoch = 0

if MODE == 'train':
    while (nb_epoch<251):
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1, shuffle=False)
        nb_epoch += 1
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
        if nb_epoch % 10 == 0:
            model.save_weights('Weights{}.hdf5'.format(nb_epoch))
            model.model.save('Weights{}.h5'.format(nb_epoch))


else:
    model = load_model('Weights.h5')
    model.load_weights('Weights.hdf5', by_name=True)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    print('\n\n')


# In[7]:


model = load_model('Weights250.h5')
#model.load_weights('Weights.hdf5', by_name=True)
generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)

