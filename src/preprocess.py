import numpy as np
from keras.datasets import imdb # type: ignore
import matplotlib.pyplot as plt

"""
This module is responsible for preprocessing the IMDB dataset for sentiment analysis. 

It includes functions to load the data, prepare the review sequences (by padding/truncating them to a fixed length), 
and split the dataset into training and testing sets. The module also provides basic analysis on review lengths 
and sample reviews to help understand the dataset better.
"""

def load_data(num_words=10000):
    """
    Load the IMDB dataset from keras datasets and merge training and testing sets.
    """
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    return data, targets

def prepare_sequences(data, targets, max_length=500):
    """
    Pad or truncate sequences to a fixed length and split into train/test sets.
    """
    data_padded = np.zeros((len(data), max_length), dtype=np.int32)
    for i, seq in enumerate(data):
        if len(seq) <= max_length:
            data_padded[i, :len(seq)] = seq
        else:
            data_padded[i, :] = seq[:max_length]
    
    ## split into training and test sets
    x_train = data_padded[:25000]
    y_train = targets[:25000]
    x_test = data_padded[25000:]
    y_test = targets[25000:]
    
    return x_train, y_train, x_test, y_test

def analyze_review_lengths(data):
    """
    Compute and display statistics on review lengths.
    """
    lengths = [len(i) for i in data]
    
    print("\nReview length statistics:")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Standard deviation: {np.std(lengths):.2f}")
    print(f"Minimum length: {np.min(lengths)}")
    print(f"Maximum length: {np.max(lengths)}")

    ## plot histogram of review lengths
    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.75, linewidth=1.5, histtype='step')
    plt.xlabel('Review Length')
    plt.ylabel('Count')
    plt.title("Distribution of Review Lengths")
    plt.show()

def display_sample_reviews(data, targets):
    """
    Display examples of short positive and negative reviews.
    """
    index = imdb.get_word_index()
    reverse_index = {value: key for key, value in index.items()}

    positive_count = 0
    negative_count = 0

    for i, review in enumerate(data):
        if len(review) < 50:  ## only show short reviews for readability
            decoded = " ".join([reverse_index.get(idx - 3, "#") for idx in review])
            if targets[i] == 1 and positive_count < 3:
                print(f"Positive Review - Index {i}: {decoded}\n")
                positive_count += 1
            elif targets[i] == 0 and negative_count < 3:
                print(f"Negative Review - Index {i}: {decoded}\n")
                negative_count += 1
            if positive_count == 3 and negative_count == 3:
                break

def preprocess_data():
    """
    Main function to execute preprocessing steps.
    """  
    print("Loading data...")
    data, targets = load_data()
    
    print("\nPreparing sequences...")
    x_train, y_train, x_test, y_test = prepare_sequences(data, targets)
    
    print("\nAnalyzing review lengths...")
    analyze_review_lengths(data)
    
    print("\nDisplaying sample reviews...\n")
    display_sample_reviews(data, targets)

    return x_train, y_train, x_test, y_test