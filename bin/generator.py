#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import pandas
from keras.models import load_model

import lib


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

    # Extract appropriate model
    char_model = load_model(filepath=lib.get_conf('generate_model_path'))

    # Extract posts to be completed
    observations = pandas.read_csv(lib.get_conf('post_seed_path'))

    logging.info('End extract')
    lib.archive_dataset_schemas('generate_extract', locals(), globals())
    return char_model, observations


def transform(char_model, observations):
    logging.info('Begin transform')

    # TODO Normalize post seed text

    # TODO Transform post_seeds into X, y
    x = None
    y = None

    # TODO Infer sentence, add to to observations

    logging.info('End transform')
    lib.archive_dataset_schemas('generate_transform', locals(), globals())
    return char_model, observations


def load(char_model, observations):
    logging.info('Begin transform')
    # TODO Export observations

    logging.info('End load')
    lib.archive_dataset_schemas('generate_load', locals(), globals())
    pass

# Main section
if __name__ == '__main__':
    main()
