# IMDB_Review_Sentiment_Analysis
## Introduction
Perform sentiment analysis on IMDB movie reviews.
Natural Language Processing using a range of different neural network architectures, i.e.
- Fully connected dense neural network
- Convolutional (1D) neural network
- Simple Recurrent neural network
- Long Short Term Memory recurrent neural network
- Stacked Bi-directional simple RNN
- Non-sequential multi convolutional neural network

## Findings
Each of these neural networks have their advantages/disadvantages:

| Model | Description |
| ----------- | ----------- |
| Fully connected dense neural network | Associating words to sentiment (unable to recognise sequences/patterns) |
| Convolutional (1D) neural network | Able to recognise a word in the context of the word next to it (window = 3) |
| Simple Recurrent neural network | Enables information to persist over time as inputs as fed onto the next timestep |
| Long Short Term Memory recurrent neural network | Overcome the vanishing gradient issue found in simple RNN (which is able to back propagate approx 10 timesteps|
| Stacked Bi-directional simple RNN | Allow for backpropagation backwards and forwards over time + increased model complexity to recognise abstract representation|
| Non-sequential multi convolutional neural network | Able to lear word-vector pairs and triplets/quadruplets of word meaning |



## Summary
Accuracy of model:
- Fully connected dense neural network **(92.37%)**
- Convolutional (1D) neural network **(95.24%)**
- Simple Recurrent neural network **(85.42%)**
- Long Short Term Memory recurrent neural network **(92.99%)**
- Stacked Bi-directional simple RNN **(92.55%)**
- Non-sequential multi convolutional neural network **(95.24%)**

