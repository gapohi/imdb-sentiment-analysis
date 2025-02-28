from preprocess import preprocess_data
from models import *

"""
This is the main script of the IMDB movie reviews sentiment analysis project. 

It coordinates the data preprocessing, model selection, and training processes. First, it loads and preprocesses the IMDB 
dataset using the `preprocess.py` module. Then, it allows for easy selection of different neural network models (such as 
RNN, LSTM, GRU, or their variants) using the `models.py` module, and trains the chosen model using the preprocessed data.
"""

def main():
    ## load data from keras datasets and preprocess
    x_train, y_train, x_test, y_test = preprocess_data()

    ## set parameters
    input_dim = 10000 ## max features
    output_dim = 16 ## embedding dim
    input_length = 500 ## max length of reviews, set to 500 after analysis

    ## choose the model
    model = create_rnn_simple(input_dim, output_dim, input_length) ## accuracy 0.50
    # model = create_lstm_simple(input_dim, output_dim, input_length) ## accuracy 0.70
    # model = create_gru_simple(input_dim, output_dim, input_length) ## accuracy 0.50
    # model = create_rnn_bidirectional(input_dim, output_dim, input_length) ## accuracy 0.63
    # model = create_lstm_bidirectional(input_dim, output_dim, input_length) ## accuracy 0.86
    # model = create_gru_bidirectional(input_dim, output_dim, input_length) ## accuracy 0.86
    # model = create_rnn_deep(input_dim, output_dim, input_length) ## accuracy 0.50
    # model = create_lstm_deep(input_dim, output_dim, input_length) ## accuracy 0.58
    # model = create_gru_deep(input_dim, output_dim, input_length) ## accuracy 0.50
    
    ## train and evaluate the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    ## print model parameters summary
    print(model.summary())
    return

if __name__== '__main__':
    main()