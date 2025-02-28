from keras import models, layers

"""
This module contains functions for creating sentiment analysis models.

It provides various model architectures, including simple RNNs, LSTMs, GRUs, and their bidirectional or deep variants. 

Each model begins with an embedding layer to transform input sequences into dense vectors, followed by a 
recurrent layer (SimpleRNN, LSTM, or GRU) that processes temporal dependencies in the data.
The output layer uses a sigmoid activation function for binary classification tasks, and the models 
are compiled with the Adam optimizer and binary cross-entropy loss function for training.
"""

def create_rnn_simple(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.SimpleRNN(64))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_simple(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_simple(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.GRU(64))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_bidirectional(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.Bidirectional(layers.SimpleRNN(64)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_bidirectional(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_bidirectional(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_deep(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.SimpleRNN(64, return_sequences=True))  
    model.add(layers.SimpleRNN(64, return_sequences=True))  
    model.add(layers.SimpleRNN(64))  
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_deep(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.LSTM(64, return_sequences=True))  
    model.add(layers.LSTM(64, return_sequences=True))  
    model.add(layers.LSTM(64))  
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_deep(input_dim, output_dim, input_length):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(layers.GRU(64, return_sequences=True))  
    model.add(layers.GRU(64, return_sequences=True))  
    model.add(layers.GRU(64))  
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model