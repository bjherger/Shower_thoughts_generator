#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
import os

import cPickle
import zipfile

import numpy
import numpy as np
import re

import pandas
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop

import lib
import models
from bin.clr_callback import CyclicLR
from sentence_callback import SentenceGenerator
from reddit_scraper import scrape_subreddit


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    if lib.get_conf('new_data_pull'):
        observations = extract()
        observations.to_feather(lib.get_conf('post_pickle_path'))

    if not os.path.exists(lib.get_conf('post_pickle_path')):
        zip_ref = zipfile.ZipFile(lib.get_conf('post_pickle_path')+'.zip', 'r')
        zip_ref.extractall(os.path.dirname(lib.get_conf('post_pickle_path')))
        zip_ref.close()

    observations = pandas.read_feather(lib.get_conf('post_pickle_path'))
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
    for text in observations['model_text']:

        # Generate x and y for observations
        observation_x, observation_y = lib.gen_x_y(text, false_y=false_y)
        x_agg.extend(observation_x)
        y_agg.extend(observation_y)

    x = numpy.matrix(x_agg)
    y = numpy.matrix(y_agg)
    return observations, char_indices, indices_char, x, y

def model(observation, char_indices, indices_char, x, y):

    char_model = models.rnn_embedding_model(x, y)

    # Set up model training variables
    optimizer = RMSprop(lr=0.01)
    char_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    batch_size = 4096
    num_epochs = 200

    if lib.get_conf('test_run'):
        num_epochs = 2

    # Set up callbacks
    tf_log_path = os.path.join(os.path.expanduser('~/.logs'), lib.get_batch_name())
    logging.info('Using Tensorboard path: {}'.format(tf_log_path))

    mc_log_path = os.path.join(lib.get_conf('`'), lib.get_batch_name() + '_epoch_{epoch:03d}_loss_{loss:.2f}.h5py')
    logging.info('Using mc_log_path path: {}'.format(mc_log_path))

    sentence_generator = SentenceGenerator(verbose=1)

    clr_step_size = numpy.floor((float(x.shape[0]) / batch_size) * 4)
    clr = CyclicLR(base_lr=.005, max_lr=.02, mode='triangular2', step_size=clr_step_size)
    logging.info('Using CRL step size: {}'.format(clr_step_size))

    callbacks = [TensorBoard(log_dir=tf_log_path),
                 ModelCheckpoint(mc_log_path),
                 sentence_generator,
                 clr]

    # Train the model, output generated text after each iteration
    char_model.fit(x, y,
              batch_size=batch_size,
              epochs=num_epochs, callbacks=callbacks)

    print sentence_generator.sentences

# Main section
if __name__ == '__main__':
    main()
