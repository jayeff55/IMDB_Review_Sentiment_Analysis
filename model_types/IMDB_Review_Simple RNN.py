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
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers import SimpleRNN, SpatialDropout1D

from keras.callbacks import ModelCheckpoint

# tools to evaluate/visualise network performance
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


# c) Simple recurrent neural network architecture
# Output directory
output_dir = "Model_output_param\\RNN"

# Training
E = 16                      # no. of epochs increased because overfitting typically don't occur in the early epochs
batchSize = 128

# vector-space embedding
n_dim = 64

n_unique_words = 10000      
n_words_to_skip = 50        
max_review_length = 100     # lowered this because of vanishing gradient over time
pad_type = "pre"            
trunc_type = "pre"          
drop_embed = 0.2            

# RNN parameters
n_rnn = 256                 
drop_rnn = 0.2




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




# 3c) Define Simple RNN architecture
# Unlike the CNN model where it is only able to consider a word in the context
# of two words (one on each side - since 1D convolutional window = 3 here)
# An RNN enables information to persist over time, since a module will receive
# an additional input from the previous module

model = Sequential()

model.add(Embedding(n_unique_words,
                    n_dim,
                    input_length = max_review_length))
model.add(SpatialDropout1D(drop_embed))

model.add(SimpleRNN(n_rnn, dropout = drop_rnn))
# No dense layer added after a recurrent layer because it provides little 
# performance advantage

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
# The simple RNN performs worst because it is only able to 
# backpropagate through ~10 time steps before the gradient 
# diminishes so much that parameter updates become negligibly 
# small. It is only likely to be useful where the context is small 
# i.e. fewer than 10 timesteps

# so, we load the weights from the appropriate epoch
model.load_weights(output_dir+"\\weights.05.hdf5")

# Compute predictions
y_hat = model.predict(x_val)


# Visualise distribution of predicted y_hat
plt.hist(y_hat)
plt.axvline(0.5, color = "orange")


# Measure performance with ROC_AUC score
pct_auc = roc_auc_score(y_val, y_hat)*100.0
print("ROC AUC = %.2f percent" % pct_auc)

# Simple Recurrent neural net   : 85.42% (model weights from epoch #5)



# 8. Visualise distribution of prediction
# create dataframe with actual and predicted labels, i.e. y_val vs y_hat
ydf = pd.DataFrame(y_hat, columns=['y_hat'])
ydf['y_val'] = pd.DataFrame(y_val, columns=['y_val'])

# Examples of false positive, i.e. y_val = 0, but y_hat>0.9
ydf[(ydf['y_val']==0) & (ydf['y_hat']>0.9)].head(10)

# Examples of false negative, i.e. y_val = 1, but y_hat<0.1
ydf[(ydf['y_val']==1) & (ydf['y_hat']<0.1)].head(10)

