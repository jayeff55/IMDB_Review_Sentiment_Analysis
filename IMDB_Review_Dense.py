# -*- coding: utf-8 -*-
"""

IMDB Film Review Sentiment Analysis
Natural Language Processing

"""
# tools to build neural network
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Embedding 

from keras.callbacks import ModelCheckpoint

# tools to evaluate/visualise network performance
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


# a) DENSE neural network
# Output directory
output_dir = "Model_output_param\\Dense"

# Training
E = 4                       # NLP models often overfit to the training data in fewer epochs than machine vision models 
batchSize = 128

# vector-space embedding
n_dim = 64

n_unique_words = 5000       # top 5000 most popular token in corpus 
                            # this is an alternative approach to defining the minimum occurrence of token to be used

n_words_to_skip = 50        # treating the top 50 most popular work as stop words
                            # instead of removing words from a curated list of stop words
                            
max_review_length = 100     # defining length of input data, i.e. 100 words

pad_type = "pre"            # padding characters at the start of reviews <100 words 

trunc_type = "pre"          # removing the front part of reviews >100 words (bold assumption!)


# Dense network parameters
n_dense = 64    # number of nodes
D = 0.5         # Dropout proportion




# 2. Loading and pre-processing data
(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words = n_unique_words,
                                                    skip_top = n_words_to_skip)

# convert review from index integers to natural language
word_index = keras.datasets.imdb.get_word_index()

# resetting index value (value + 3 to accommodate 3 new entries below)
word_index = {k:(v+3) for k,v in word_index.items()}

# to the dictionary of word_index, add 3 new entries
# which are customary for representing padding, starting and unknown tokens
word_index["PAD"] = 0
word_index["START"] = 1
word_index["UNK"] = 2

# invert the dictionary 
index_word = {v:k for k,v in word_index.items()}

# standardise length of length of reviews
x_train = pad_sequences(x_train,
                        maxlen = max_review_length,
                        padding = pad_type,
                        truncating = trunc_type,
                        value = 0)

x_val = pad_sequences(x_val,
                      maxlen = max_review_length,
                      padding = pad_type,
                      truncating = trunc_type,
                      value = 0)




# 3a) Define DENSE network architecture
model = Sequential()
model.add(Embedding(n_unique_words,     
                    n_dim,              
                    input_length = max_review_length))

model.add(Flatten())

model.add(Dense(n_dense, activation = "relu"))
model.add(Dropout(D))

model.add(Dense(n_dense, activation = "relu"))
model.add(Dropout(D))

model.add(Dense(1, activation = "sigmoid"))




# 4. Compile model
model.compile(optimizer = "adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])





# 5. Set up model checkpoint
model_checkpoint = ModelCheckpoint(filepath = output_dir+"\\weights.{epoch:02d}.hdf5")

if not os.path.exists(output_dir):  
    os.makedirs(output_dir)        
    




# 6. Fit model
model.fit(x_train,
          y_train,
          batch_size = batchSize,
          epochs = E,
          verbose = 1,
          validation_data = (x_val, y_val),
          callbacks = [model_checkpoint])




# 7. Evalute model prediction
# Lowest validation loss  for dense model (0.3511) and highest validation 
# accuracy (84.5 percent) 
# achieved in the second epoch. 
# In the third and fourth epochs, the model is heavily overfitted, 
# with accuracy on the training set considerably higher than on 
# the validation set. 
# By the fourth epoch, training accuracy stands at 99.02%
# while validation accuracy is much lower, at 82.7%

# so, we load the weights from the second epoch
model.load_weights(output_dir+"\\weights.02.hdf5")

# then predict y_hat for x_val using weights from second epoch
# y_hat is probability sentiment is positive
y_hat = model.predict(x_val)


# Visualise distribution of predicted y_hat
plt.hist(y_hat)
plt.axvline(0.5, color = "orange")

# from the plot, we can see that the model has strong opinions
# on sentiments. Most are either <0.1 (negative sentiment) 
# or >0.9 (positive sentiment)
# the straight line at x = 0.5 is just an arbituary threshold


# To obtained ore nuanced assessment of the model, perform ROC_AUC
pct_auc = roc_auc_score(y_val, y_hat)*100.0
print("ROC AUC = %.2f percent" % pct_auc)

# Dense neural net              : 92.37% (model weights from epoch #2)



