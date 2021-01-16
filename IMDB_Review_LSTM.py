# -*- coding: utf-8 -*-
"""
Jeannie Foo
29/12/2020

IMDB Film Review Sentiment Analysis
Natural Language Processing

"""
# tools to build neural network
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers import LSTM, SpatialDropout1D

from keras.callbacks import ModelCheckpoint

# tools to evaluate/visualise network performance
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


# d) LSTM Recurrent neural network architecture
# Output directory
output_dir = "Model_output_param\\LSTM"

# Training
E = 4                     # small no. epochs because LSTM overfits more quickly than simple RNN
batchSize = 128

# vector-space embedding
n_dim = 64

n_unique_words = 10000      
n_words_to_skip = 50        
max_review_length = 100     
pad_type = "pre"            
trunc_type = "pre"          
drop_embed = 0.2            

# LSTM parameters
n_lstm = 256                 
drop_lstm = 0.2




# 2. Loading and pre-processing data
(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words = n_unique_words,
                                                    skip_top = n_words_to_skip)

# convert review from index integers to natural language
word_index = keras.datasets.imdb.get_word_index()

# resetting index value (value + 3 to accommodate additional entries below)
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




# 3d) Define LSTM RNN architecture
# Apply LSTM when a broader context/longer sequence is needed. 
# As LSTMs cells can hold two types of information, i.e. hidden state and cell state.
# Cell state allows the model to "retain" memory further along the sequence
# And hidden states froms previous cells combined with the updated information
# from cell states allow relevant information to be regulated
model = Sequential()

model.add(Embedding(n_unique_words,
                    n_dim,
                    input_length = max_review_length))
model.add(SpatialDropout1D(drop_embed))

model.add(LSTM(n_lstm, dropout = drop_lstm))

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
# The lowest validation loss and highest validation accuracy occurs in the
# second epoch

# so, we load the weights from the appropriate epoch
model.load_weights(output_dir+"\\weights.02.hdf5")

# Compute predictions
y_hat = model.predict(x_val)


# Visualise distribution of predicted y_hat
plt.hist(y_hat)
plt.axvline(0.5, color = "orange")


# Measure performance with ROC_AUC score
pct_auc = roc_auc_score(y_val, y_hat)*100.0
print("ROC AUC = %.2f percent" % pct_auc)

# LSTM Recurrent neural net             : 92.99% (model weights from epoch #2)






