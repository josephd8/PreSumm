
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout, MaxPooling1D, Conv1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
import gensim
import numpy as np
import statistics


def get_data(rel_path_dir, filename, cols):
    """Retrieve full, unprocessed data as a pandas dataframe"""
    
    wd = os.getcwd()
    os.chdir(rel_path_dir)
    data = pd.read_csv(filename, usecols = cols)
    os.chdir(wd)
    
    return data

def tokenize(text):
    """Tokenize reviews - remove punctuation and send to lower case"""
    
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&?').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

def uni_and_bigrams(text):
    """Create unigram, bigram, and unigrams + bigrams"""
    
    # our unigrams are our tokens
    unigrams=tokenize(text)
    # the bigrams just contatenate 2 adjacent tokens with _ in between
    bigrams=list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))
    # returning a list containing all 1 and 2-grams
    return unigrams, bigrams, unigrams+bigrams

def preprocess(data):
    """Preprocess both text reviews and labels"""
    
    stars = list(map(lambda x: int(x),list(data['Stars'])))
    ratings = [1 if x > 4 else 0 for x in stars]
    
    reviews = list(map(tokenize, data['Review']))
    reviews = list(map(lambda x: " ".join(x), reviews))
    
    return reviews, ratings

def create_grams(reviews, n = "uni"):
    """Create bag of word representations with with either 'uni', 'bi', or 'both'"""
    
    if(n == "uni"):
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range = (1,1), min_df = 5, max_features = 10000)
    elif(n == "bi"):
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range = (2,2), min_df = 5, max_features = 10000)
    else:
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range = (1,2), min_df = 5, max_features = 10000)
        
    grams = vectorizer.fit_transform(reviews)
    
    return grams

def print_lr_scores(cv_results_):
    """Pull scores for each parameter combo out of grid search object and print them"""
    
    scores = list(zip(cv_results_['split0_test_score'], cv_results_['split1_test_score'], cv_results_['split2_test_score']))
    cv_accuracy = list(map(statistics.mean, scores))
    params = cv_results_['params']
    
    for i in range(len(params)):
        print("%s --- %f" % (params[i] , cv_accuracy[i]))
        
def print_svm_scores(cv_results_):
    """Pull scores for each parameter combo out of grid search object and print them"""
    
    scores = list(zip(cv_results_['split0_test_score'], cv_results_['split1_test_score']))
    cv_accuracy = list(map(statistics.mean, scores))
    params = cv_results_['params']
    
    for i in range(len(params)):
        print("%s --- %f" % (params[i] , cv_accuracy[i]))


def create_fasttest_sets(reviews, ratings, train_size, test_size, train_filename, test_filename):
    """Create specific datasets for fastest training and testing"""
    
    test_start = train_size + 1
    test_end = test_start + test_size
    
    ft_data = ["__label__"+str(x[1]) + " " + x[0] for x in zip(reviews[:train_size], ratings[:train_size])]

    with open('../data/yelp/'+test_filename, mode = "w", encoding="utf-8") as outfile:
        for s in ft_data_test:
            outfile.write("%s\n" % s)
        
    with open('../data/yelp/'+train_filename, mode = "w", encoding="utf-8") as outfile:
        for s in ft_data:
            outfile.write("%s\n" % s)
            
def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))


def train(train_texts, train_labels, dictionary, dropout_rate = 0.3, n_layers = 2, epochs = 10, model_file=None, EMBEDDINGS_MODEL_FILE=None):
    """
    Train a word-level CNN text classifier.
    :param train_texts: tokenized and normalized texts, a list of token lists, [['sentence', 'blah', 'blah'], ['sentence', '2'], .....]
    :param train_labels: the label for each train text
    :param dictionary: A gensim dictionary object for the training text tokens
    :param dropout_rate: dropout_rate parameter
    :param n_layers: n_layers parameter
    :param model_file: An optional output location for the ML model file
    :param EMBEDDINGS_MODEL_FILE: An optinal location for pre-trained word embeddings file location
    :return: the produced keras model, the validation accuracy, and the size of the training examples
    """
    
    # static model hyper parameters
    EMBEDDING_DIM = 100
    SEQUENCE_LENGTH_PERCENTILE = 90
    #n_layers = 2
    hidden_units = 500
    batch_size = 100
    pretrained_embedding = False
    # if we have pre-trained embeddings, specify if they are static or non-static embeddings
    TRAINABLE_EMBEDDINGS = True
    patience = 2
    #dropout_rate = 0.3
    n_filters = 100
    window_size = 8
    dense_activation = "relu"
    l2_penalty = 0.0003
    #epochs = 2
    VALIDATION_SPLIT = 0.1
        
    assert len(train_texts)==len(train_labels)
    # compute the max sequence length
    # why do we need to do that?
    lengths=list(map(lambda x: len(x), train_texts))
    a = np.array(lengths)
    MAX_SEQUENCE_LENGTH = int(np.percentile(a, SEQUENCE_LENGTH_PERCENTILE))
    # convert all texts to dictionary indices
    #train_texts_indices = list(map(lambda x: texts_to_indices(x[0], dictionary), train_texts))
    train_texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
    # pad or truncate the texts
    x_data = pad_sequences(train_texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))
    # convert the train labels to one-hot encoded vectors
    train_labels = keras.utils.to_categorical(train_labels)
    y_data = train_labels

    model = Sequential()

    # create embeddings matrix from word2vec pre-trained embeddings, if provided
    if pretrained_embedding:
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
        embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
        for word, i in dictionary.token2id.items():
            embedding_vector = embeddings_index[word] if word in embeddings_index else None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=TRAINABLE_EMBEDDINGS))
    else:
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
    # add drop out for the input layer, why do you think this might help?
    model.add(Dropout(dropout_rate))
    # add a 1 dimensional conv layer
    # a rectified linear activation unit, returns input if input > 0 else 0
    model.add(Conv1D(filters=n_filters,
                     kernel_size=window_size,
                     activation='relu'))
    # add a max pooling layer
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
    model.add(Flatten())

    # add 0 or more fully connected layers with drop out
    for _ in range(n_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units,
                        activation=dense_activation,
                        kernel_regularizer=l2(l2_penalty),
                        bias_regularizer=l2(l2_penalty),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))

    # add the last fully connected layer with softmax activation
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(train_labels[0]),
                    activation='softmax',
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

    # compile the model, provide an optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print a summary
    print(model.summary())


    # train the model with early stopping
    early_stopping = EarlyStopping(patience=patience)
    Y = np.array(y_data)

    with tf.device('/GPU:0'):
        fit = model.fit(x_data,
                        Y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=VALIDATION_SPLIT,
                        verbose=1,
                        callbacks=[early_stopping])

    print(fit.history.keys())
    val_accuracy = fit.history['val_acc'][-1]
    print(val_accuracy)
    # save the model

    if model_file:
        model.save(model_file)
    return model, val_accuracy, len(train_labels)
    
