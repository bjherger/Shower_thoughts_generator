#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging
import os

import cPickle

import keras
import numpy
import numpy as np
import re

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
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

    char_model, observations = extract()
    char_model, observations = transform(char_model, observations)
    load(char_model, observations)

    pass

def extract():

    # TODO Extract appropriate model
    char_model = load_model(filepath=lib.get_conf('generate_model_path'))

    # TODO Extract posts to be completed
    observations = None

    logging.info('End extract')
    lib.archive_dataset_schemas('extract', locals(), globals())
    return char_model, observations


def transform(char_model, observations):

    # TODO Normalize post seed text

    # TODO Transform post_seeds into X, y
    x = None
    y = None

    # TODO Infer sentence, add to to observations

    return char_model, observations


def load(char_model, observations):
    # TODO Export observations

    pass

# Main section
if __name__ == '__main__':
    main()
