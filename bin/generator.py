#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import numpy
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

    # Filter out user seeds that are too short
    # pre_filter_len = len(observations.index)
    # observations = observations[observations['user_seed'].apply(lambda x: len(x) <= lib.get_conf('ngram_len'))].copy()
    # post_filter_len = len(observations.index)
    # logging.info('Filtered out user seeds that were too short. Before length: {}, after length: {}'.format(
    #     pre_filter_len, post_filter_len))
    # if pre_filter_len != post_filter_len:
    #     logging.warn('{} user seeds were removed because they were too short.'.format(pre_filter_len - post_filter_len))
    #     print '{} user seeds were removed because they were too short.'.format(pre_filter_len - post_filter_len)

    # Reference variables
    x_strings = list()
    x_agg = list()
    y_agg = list()

    # Normalize post seed text, create windows, and add first window to agg
    for text in observations['user_seed']:

        # Generate x and y for observations
        observation_x, observation_y = lib.gen_x_y(text, false_y=True)
        print text, len(observation_x)
        if not (len(observation_x) > 0):
            logging.warn('Cleaned text was too short to infer. Input seed:  {}'.format(text))
            print('Cleaned text was too short to infer. Input seed:  {}'.format(text))
            observation_x, observation_y = lib.gen_x_y('x'*(lib.get_conf('ngram_len')+1), false_y=True)

        x_agg.extend(observation_x[:1])
        y_agg.extend(observation_y[:1])
        model_text = ''.join(map(lambda x: lib.get_indices_char()[x], observation_x[0]))
        x_strings.append(model_text)


    observations['model_seed'] = x_strings
    observations['user_seed_truncated'] = observations['user_seed'].apply(lambda x: len(x) > 40)

    # Transform post_seeds into X, y
    x = numpy.matrix(x_agg)
    y = numpy.matrix(y_agg)

    # Infer sentence, create sentence predictions
    sentence_agg = list()
    for index, observation in observations.iterrows():
        model_seed = observation['model_seed']

        for diversity in numpy.arange(.1, 1.3, .1):

            generated = ''
            sentence = model_seed
            generated += sentence

            # Generate next characters, using a rolling window
            for next_char_index in range(lib.get_conf('pred_length')):
                x_pred, text_y = lib.gen_x_y(sentence, false_y=True)

                preds = char_model.predict(x_pred, verbose=0)[-1]

                next_index = lib.sample(preds, diversity)
                next_char = lib.get_indices_char()[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            local_dict = dict()
            local_dict['observation_index'] = index
            local_dict['seed'] = model_seed
            local_dict['diversity'] = diversity
            local_dict['generated_post'] = generated
            sentence_agg.append(local_dict)

    generated_sentences = pandas.DataFrame(sentence_agg)
    print generated_sentences

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
