import keras
import pandas

import lib

class SentenceGenerator(keras.callbacks.Callback):

    def __init__(self, output_path=None, verbose=0):
        super(SentenceGenerator, self).__init__()
        self.output_path = output_path
        self.verbose = verbose
        self.epoch = 0
        self.sentences = pandas.DataFrame(columns=['epoch', 'seed', 'diversity', 'generated_post'])


    def on_batch_end(self, batch, logs={}):

        # Reference variables
        sentence_agg = list()

        seed_chars = '<Jiggling around cheap iPhone chargers to find the sweet spot is the millennial version of ' \
                     'tweaking a TV antenna.>'[:lib.get_conf('ngram_len')]

        for diversity in [0.2, 0.5, 1.0, 1.2]:

            generated = ''
            sentence = seed_chars
            generated += sentence

            # Generate next characters, using a rolling window
            for next_char_index in range(lib.get_conf('pred_length')):
                x_pred, text_y = lib.gen_x_y(sentence, false_y=True)

                preds = self.model.predict(x_pred, verbose=0)[-1]

                next_index = lib.sample(preds, diversity)
                next_char = lib.get_indices_char()[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

            local_dict = dict()
            local_dict['epoch'] = self.epoch
            local_dict['seed'] = seed_chars
            local_dict['diversity'] = diversity
            local_dict['generated_post'] = generated
            sentence_agg.append(local_dict)

            if self.verbose >= 1:
                print('Diversity: {}, generated post: {}'.format(local_dict['diversity'], local_dict['generated_post']))

        epoch_sentences = pandas.DataFrame(sentence_agg)
        
        self.sentences = pandas.concat(objs=[self.sentences, epoch_sentences])
        if self.output_path is not None:
            self.sentences.to_csv(path_or_buf=self.output_path, index=False)

        self.epoch += 1