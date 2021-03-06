from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


class LSTMFeaturizer(object):
    def __init__(self, max_nb_words):
        self.max_nb_words = max_nb_words
        self.kerasTokenizer = Tokenizer(
            num_words=max_nb_words, oov_token=1, filters=None
        )
        self.word_index = None

    def fit_transform(self, textarray):
        # get maximum sequence length form padding
        # doc_len = text.apply(lambda words: len(words.split(" ")))
        # self.max_seq_len = 20
        self.kerasTokenizer.fit_on_texts(textarray.tolist())
        self.word_index = self.kerasTokenizer.word_index
        # transform training data to sequence
        word_seq = np.array(
            self.kerasTokenizer.texts_to_sequences(textarray.tolist())
        )
        return word_seq, self.kerasTokenizer

    def transform(self, textarray):
        word_seq = np.array(
            self.kerasTokenizer.texts_to_sequences(textarray.tolist())
        )
        return word_seq
