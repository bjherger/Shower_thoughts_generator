#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import io
import logging
import random

import sys

import numpy
import tensorflow
from keras import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import get_file
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K

import numpy as np

import lib
import models


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.WARN)

    text = extract()
    text, char_indices, indices_char, x, y = transform(text)
    model(text, char_indices, indices_char, x, y)

    pass

def extract():
    # TODO Extract

    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = io.open(path, encoding='utf-8').read().lower()

    if lib.get_conf('test_run'):
        text = text[:10000]

    return text


def transform(text, false_y=False):
    # TODO Should use universal character set, for inference time
    chars = sorted(list(set(lib.legal_characters())))
    if false_y:
        text +=' '

    text = map(lambda x: x.lower(), text)
    text = filter(lambda x: x in lib.legal_characters(), text)
    text = ''.join(text)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    step = 3
    sentences = []
    next_chars = []
    for observation_index in range(0, len(text) - lib.get_conf('ngram_len'), step):
        sentences.append(text[observation_index: observation_index + lib.get_conf('ngram_len')])
        next_chars.append(text[observation_index + lib.get_conf('ngram_len')])

    x = np.zeros((len(sentences), lib.get_conf('ngram_len')), dtype=numpy.uint16)

    y = np.zeros((len(sentences), len(chars)), dtype=bool)

    for observation_index, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[observation_index, t] = char_indices[char]
        y[observation_index, char_indices[next_chars[observation_index]]] = 1


    return text, char_indices, indices_char, x, y

def model(text, char_indices, indices_char, x, y):

    model = models.rnn_embedding_model(x, y)

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Train the model, output generated text after each iteration
    for iteration in range(1, 60):
        logging.info('Iteration number: {}'.format(iteration))
        print 'Iteration number: {}'.format(iteration)
        model.fit(x, y,
                  batch_size=4096,
                  epochs=1)

        start_index = random.randint(0, len(text) - lib.get_conf('ngram_len') - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:

            generated = ''
            sentence = text[start_index: start_index + lib.get_conf('ngram_len')]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            print(generated)

            # Generate 400 characters, using a rolling window
            for next_char_index in range(400):
                text_text, text_char_indices, text_indices_char, x_pred, text_y = transform(sentence, false_y=True)

                preds = model.predict(x_pred, verbose=0)[-1]

                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            print 'Seed: {}, diversity: {}'.format(text[start_index: start_index + lib.get_conf('ngram_len')], diversity)
            print generated

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Main section
if __name__ == '__main__':
    main()
