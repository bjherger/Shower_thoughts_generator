import keras
import numpy
from keras import Model, Sequential
from keras import backend as K
from keras.layers import Dense, Flatten, Embedding, LSTM, Activation, Reshape, Lambda
from keras.optimizers import RMSprop
from tensorflow import one_hot

import lib


def ff_model(X, y):
    if len(X.shape) >= 2:
        embedding_input_length = int(X.shape[1])
    else:
        embedding_input_length = 1

    # Embedding input dimensionality is the same as the number of classes in the input data set
    embedding_input_dim = int(numpy.max(X)) + 1

    # Embedding output dimensionality is determined by heuristic
    embedding_output_dim = int(min((embedding_input_dim + 1) / 2, 50))

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype='int32', name='char_input')

    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                input_length=embedding_input_length,
                                trainable=True,
                                name='char_embedding')

    # Create output layer
    softmax_output_dim = len(y[0])
    output_layer = Dense(units=softmax_output_dim, activation='softmax')

    # Create model architecture
    embedded_sequences = embedding_layer(sequence_input)
    x = Flatten()(embedded_sequences)
    x = Dense(32, activation='linear')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='linear')(x)
    x = output_layer(x)

    char_model = Model(sequence_input, x)
    char_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return char_model


def rnn_embedding_model(X, y):

    if len(X.shape) >= 2:
        embedding_input_length = int(X.shape[1])
    else:
        embedding_input_length = 1

    # Embedding input dimensionality is the same as the number of classes in the input data set
    embedding_input_dim = int(numpy.max(X)) + 1

    # Embedding output dimensionality is determined by heuristic
    embedding_output_dim = int(min((embedding_input_dim + 1) / 2, 50))

    # Use a smaller datatype, if possible. This explicit typing is necessary due to the OHE layer.
    if embedding_input_dim < 250:
        dtype = 'uint8'
    else:
        dtype = 'int32'

    sequence_input = keras.Input(shape=(embedding_input_length,), dtype=dtype, name='char_input')

    embedding_layer = Embedding(input_dim=embedding_input_dim,
                                output_dim=embedding_output_dim,
                                input_length=embedding_input_length,
                                trainable=True,
                                name='char_embedding')

    # Create output layer
    softmax_output_dim = len(y[0])
    output_layer = Dense(units=softmax_output_dim, activation='softmax')

    # Create model architecture
    x = embedding_layer(sequence_input)
    x = LSTM(128, dropout=.2, recurrent_dropout=.2)(x)
    x = output_layer(x)

    optimizer = RMSprop(lr=.001)
    char_model = Model(sequence_input, x)
    char_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return char_model

def rnn_model(X, y):
    chars = sorted(list(set(lib.legal_characters())))
    if len(X.shape) >= 2:
        input_length = int(X.shape[1])
    else:
        input_length = 1

    nb_classes = numpy.max(X) + 1

    if nb_classes < 250:
        dtype = 'uint8'
    else:
        dtype = 'int32'

    sequence_input = keras.Input(shape=(input_length,), dtype=dtype, name='char_input')

    x_ohe = Lambda(K.one_hot,
                   arguments={'num_classes': nb_classes}, output_shape=(input_length,nb_classes))

    # lib.get_conf('ngram_len')
    # Create output layer
    softmax_output_dim = len(y[0])
    output_layer = Dense(units=softmax_output_dim, activation='softmax')

    # Create model architecture
    x = x_ohe(sequence_input)
    x = LSTM(128)(x)
    x = output_layer(x)

    optimizer = RMSprop(lr=.01)
    char_model = Model(sequence_input, x)
    char_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return char_model