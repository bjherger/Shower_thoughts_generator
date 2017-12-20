#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
import random

import cPickle
import numpy
import numpy as np
from keras.optimizers import RMSprop

import lib
import models
from reddit_scraper import scrape_subreddit


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # observations = extract()
    # cPickle.dump(observations, open('../data/pickles/posts_extract.pkl', 'w+'))

    observations = cPickle.load(open('../data/pickles/posts_extract.pkl'))
    observations, char_indices, indices_char, x, y = transform(observations)
    model(observations, char_indices, indices_char, x, y)

    pass

def extract():
    # TODO Extract

    # Extract all posts for given subreddit, going back given number of days
    logging.info('Downloading submissions from Reddit')
    observations = scrape_subreddit(lib.get_conf('subreddit'), lib.get_conf('history_num_days'))
    logging.info('Found {} submissions'.format(len(observations.index)))

    logging.info('End extract')
    lib.archive_dataset_schemas('extract', locals(), globals())
    return observations


def transform(observations, false_y=False):

    # Reference variables
    char_indices = lib.get_char_indices()
    indices_char = lib.get_indices_char()
    x_agg = list()
    y_agg = list()

    if lib.get_conf('test_run'):
        observations = observations.head(100).copy()

    # Create a single field with all text
    # TODO Add start and end tokens
    observations['model_text'] = observations['title'] + ' ' + observations['selftext']

    # Iterate through individual observations
    for text in observations['model_text']:

        # Generate x and y for observations
        observation_x, observation_y = lib.gen_x_y(text, false_y=false_y)
        x_agg.extend(observation_x)
        y_agg.extend(observation_y)

    x = numpy.matrix(x_agg)
    y = numpy.matrix(y_agg)
    return observations, char_indices, indices_char, x, y

def model(observation, char_indices, indices_char, x, y):

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

        for diversity in [0.2, 0.5, 1.0, 1.2]:

            generated = ''
            seed_index = numpy.random.choice(len(x))
            seed_indices = x[seed_index].tolist()[0]
            print len(seed_indices), 'seed_indices'
            seed_chars = ''.join(map(lambda x: lib.get_indices_char()[x], seed_indices))

            sentence = seed_chars
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            print(generated)

            # Generate next characters, using a rolling window
            for next_char_index in range(lib.get_conf('pred_length')):
                
                x_pred, text_y = lib.gen_x_y(sentence, false_y=True)

                preds = model.predict(x_pred, verbose=0)[-1]

                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            print 'Seed: {}, diversity: {}'.format(seed_chars, diversity)
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
