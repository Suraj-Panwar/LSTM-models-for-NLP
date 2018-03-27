
# coding: utf-8

# In[1]:


################
#Import Modules#
################

import numpy as np
import nltk
import matplotlib
from matplotlib import pyplot as plt
import random
from nltk.corpus import brown,gutenberg
from nltk import bigrams
from nltk import trigrams
from nltk import ngrams
import copy
import tensorflow as tf
import gensim
from gensim.models import word2vec as Word2Vec
import pandas as pd
import time
import pickle


# In[3]:


######################################
#Generating Sets for Gutenberg Corpus#
######################################

random.seed(22)
#Generate Train set for Gutenberg
guten_train = random.sample(list(gutenberg.fileids()), 12)

#Generate Dev set for Gutenberg
guten_dev_1 = [i for i in list(gutenberg.fileids()) if i not in guten_train]
guten_dev = random.sample(guten_dev_1, 3)

#Generate Test set for Gutenberg
guten_test = [j for j in list(guten_dev_1) if j not in guten_dev]

for i, j, k in zip(guten_train, guten_dev, guten_test):
    assert i not in guten_dev
    assert i not in guten_test
    assert j not in guten_train
    assert j not in guten_test
    assert k not in guten_dev
    assert k not in guten_train
char_list = list(gutenberg.raw(fileids = guten_train))
char_list_1 = list(gutenberg.raw(fileids = guten_test))


# In[ ]:


lookup_1 = nltk.FreqDist(gutenberg.raw())


# In[4]:


###########################################
#Create word Embedding list for Tensorflow#
###########################################

lookup= []
for p in lookup_1:
    lookup.append(p)
numer_list_1 = []
for k in (char_list_1):
    numer_list_1.append(lookup.index(k))


# In[5]:


###########################################
#Create word Embedding list for Tensorflow#
###########################################

lookup= []
for p in lookup_1:
    lookup.append(p)
numer_list = []
for k in (char_list):
    numer_list.append(lookup.index(k))


# In[6]:


##############################
#Declare Parameters for Model#
##############################

#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient
max_grad_norm = 5
#The number of layers in our model
num_layers = 1
#The total number of recurrence steps
num_steps = 20
# Hidden neurons numbers
hidden_size = 200
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 20
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 33369
#Training flag to separate training from testing
is_training = 1

tule = [numer_list[i:i+600] for i in range(0, len(numer_list), 20)]
tule_1 = [numer_list[i+1:i+601] for i in range(0, len(numer_list), 20)]
x = np.array(tule[1]).reshape([30,20])
y = np.array(tule_1[1]).reshape([30,20])


# In[7]:


class LSTM_word_perplexity(object):

    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        size = hidden_size
        self.vocab_size = vocab_size
        
        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])  
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    
        ############################################
        # Creating the input structure for our RNN #
        ############################################

        ####################################################################################################
        # Instanciating our RNN model and retrieving the structure for returning the outputs and the state #
        ####################################################################################################
        
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=self._initial_state)

        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        #output = tf.reshape(tf.concat(1, outputs), [-1, size])
        output = tf.reshape(outputs, [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) 
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) 
        logits = tf.matmul(output, softmax_w) + softmax_b

        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                      [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        self._final_state = state

        if not is_training:
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def inputs(self):
        return self._inputs
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# In[8]:


##########################################################################################################################
# run_epoch takes as parameters the current session, the model instance, the data to be fed, and the operation to be run #
##########################################################################################################################
def run_epoch(session, m, data, eval_op, verbose=False):

    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    #m.initial_state = tf.convert_to_tensor(m.initial_state) 
    #state = m.initial_state.eval()
    state = session.run(m.initial_state)
    #state = tf.reshape(state, [30,20])
    
#     tule = [data[i:i+600] for i in range(0, len(data), 600)]
#     del tule[-1]
#     tule_1 = [data[i+1:i+601] for i in range(0, len(data), 600)]
#     del tule_1[-1]
    tule = [data[i:i+600] for i in range(0, len(data), 600)]
    del tule[-1]
    tule_1 = [data[i+1:i+601] for i in range(0, len(data), 600)]
    del tule_1[-1]
    #For each step and data point
    for step, x , y in zip(range(len(tule)), tule, tule_1):
         
        x = np.array(x).reshape([30,20])
        y = np.array(y).reshape([30,20])
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        cost, state,  _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        
        #Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += cost
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            #print("\n", step)
            print("%.3f perplexity: %.3f" % (step * 1.0 / epoch_size, np.exp(costs / iters)))
            test_perplexity, state_test = run_epoch_test(session, mtest, test_data, tf.no_op(), state)
            print("Test Perplexity: %.3f" % test_perplexity)

    return np.exp(costs / iters), state


# In[ ]:


##########################################################################################################################
# run_epoch_test takes as parameters the current session, the model instance, the data to be fed, the state of the trained model and the operation to be run #
##########################################################################################################################
def run_epoch_test(session, m, data, eval_op, state_1, verbose=False):

    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    #state = m.initial_state.eval()
    #m.initial_state = tf.convert_to_tensor(m.initial_state) 
    #state = m.initial_state.eval()
    state = state_1
    #state = tf.reshape(state, [30,20])
    
#     tule = [data[i:i+600] for i in range(0, len(data), 600)]
#     del tule[-1]
#     tule_1 = [data[i+1:i+601] for i in range(0, len(data), 600)]
#     del tule_1[-1]
    tule = [data[i:i+600] for i in range(0, len(data), 600)]
    del tule[-1]
    tule_1 = [data[i+1:i+601] for i in range(0, len(data), 600)]
    del tule_1[-1]
    #For each step and data point
    for step, x , y in zip(range(len(tule)), tule, tule_1):
         
        x = np.array(x).reshape([30,20])
        y = np.array(y).reshape([30,20])
        cost, state,  _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        
        costs += cost
        
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            #print("\n", step)
            print("%.3f perplexity: %.3f" % (step * 1.0 / epoch_size, np.exp(costs / iters)))
            
            

    return np.exp(costs / iters), state


# In[10]:


train_data = numer_list
test_data = numer_list_1
with tf.Graph().as_default():
    session = tf.InteractiveSession()
    initializer = tf.random_uniform_initializer(-init_scale,init_scale)
    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = LSTM_word_perplexity(is_training=True)
        
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = LSTM_word_perplexity(is_training=False)
        mtest = LSTM_word_perplexity(is_training=False)

    tf.global_variables_initializer().run()

    for i in range(max_max_epoch):
        lr_decay = decay ** max(i - max_epoch, 0.0)
        
        m.assign_lr(session, learning_rate * lr_decay)

        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        train_perplexity, state = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
        test_perplexity, state_1 = run_epoch(session, mtest, test_data, state, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)

