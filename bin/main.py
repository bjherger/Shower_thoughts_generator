#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
import os

import cPickle
import numpy
import numpy as np
import re

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop

import lib
import models
from sentence_callback import SentenceGenerator
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

    # Create a single field with all text. < and > serve as start and end tokens
    observations['model_text'] = observations['title'] + ' ' + observations['selftext']

    # Iterate through individual observations
    for text in observations['user_seed']:

        # Generate x and y for observations
        observation_x, observation_y = lib.gen_x_y(text, false_y=false_y)
        x_agg.extend(observation_x)
        y_agg.extend(observation_y)

    x = numpy.matrix(x_agg)
    y = numpy.matrix(y_agg)
    return observations, char_indices, indices_char, x, y

def model(observation, char_indices, indices_char, x, y):

    model = models.rnn_embedding_model(x, y)

    # Set up model training variables
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Set up callbacks
    tf_log_path = os.path.join(os.path.expanduser('~/.logs'), lib.get_batch_name())
    logging.info('Using Tensorboard path: {}'.format(tf_log_path))
    mc_log_path = os.path.join(lib.get_conf('model_checkpoint_path'), lib.get_batch_name() + '_epoch_{epoch:03d}_loss_{loss:.2f}.h5py')
    logging.info('Using mc_log_path path: {}'.format(mc_log_path))
    sentence_generator = SentenceGenerator(verbose=1)

    callbacks = [TensorBoard(log_dir=tf_log_path),
                 ModelCheckpoint(mc_log_path),
                 sentence_generator]

    # Train the model, output generated text after each iteration
    model.fit(x, y,
              batch_size=4096,
              epochs=2, callbacks=callbacks)

    print sentence_generator.sentences

# Main section
if __name__ == '__main__':
    main()
