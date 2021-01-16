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
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D, SpatialDropout1D, GlobalMaxPooling1D
from keras.layers import Input, concatenate
from keras.callbacks import ModelCheckpoint

# tools to evaluate/visualise network performance
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


# f) Non-sequential multi convolutional neural architecture
# Output directory
output_dir = "Model_output_param\\MultiConv"

# Training
E = 4                     
batchSize = 128

# vector-space embedding
n_dim = 64

n_unique_words = 5000      
n_words_to_skip = 50        
max_review_length = 400     
pad_type = "pre"            
trunc_type = "pre"          
drop_embed = 0.2            

# Multi Convolutional Neural network parameter
n_conv_1 = n_conv_2 = n_conv_3 = 256    
k_conv_1 = 2
k_conv_2 = 3
k_conv_3 = 4

# Dense layer parameter
n_dense = 256
D = 0.2




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




# 3f) Define Multi CNN architecture
# This non-sequential architecture allows model to specialise in learning 
# word-vector pairs and triplets/quadruplets of word meaning

# Define input layer
input_layer = Input(shape = max_review_length,
                    dtype = "int16",
                    name = "input")

# Define embedding layer
embedding_layer = Embedding(n_unique_words,
                            n_dim,
                            name = "embedding")(input_layer)
drop_embed_layer = SpatialDropout1D(drop_embed,
                                    name = "drop_embed")(embedding_layer)

# Define 3 parallel Conv2D streams
conv_1 = Conv1D(n_conv_1, 
                k_conv_1, 
                activation = "relu",
                name = "conv_1")(drop_embed_layer)
maxp_1 = GlobalMaxPooling1D(name = "maxp_1")(conv_1)

conv_2 = Conv1D(n_conv_2, 
                k_conv_2, 
                activation = "relu",
                name = "conv_2")(drop_embed_layer)
maxp_2 = GlobalMaxPooling1D(name = "maxp_2")(conv_2)

conv_3 = Conv1D(n_conv_3, 
                k_conv_3, 
                activation = "relu",
                name = "conv_3")(drop_embed_layer)
maxp_3 = GlobalMaxPooling1D(name = "maxp_3")(conv_3)

# Concatenate outputs/activations from the 3 parallel streams above
concat = concatenate([maxp_1, maxp_2, maxp_3])

# Define 2 dense layers
dense_1 = Dense(n_dense,
                activation = "relu",
                name = "dense_1")(concat)
drop_dense_1 = Dropout(D,
                       name = "dropout_1")(dense_1)

dense_2 = Dense(int(n_dense/4),
                activation = "relu",
                name = "dense_2")(drop_dense_1)
drop_dense_2 = Dropout(D,
                       name = "dropout_2")(dense_2)

# Define sigmoid layer
predictions = Dense(1, 
                    activation = "sigmoid", 
                    name = "output")(drop_dense_2)

# Finally, assemble the two parts together!
model = Model(input_layer, predictions)





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

# Multi CNN (Non-sequential)            : 95.24% (model weights from epoch #3)






