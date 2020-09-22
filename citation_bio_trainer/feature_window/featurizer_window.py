import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing import *
from util.utils import pad_sequences, sliding_window_list, flatten_3d_list
from feature_window.spacyfeaturizer_window import (
    SpacyFeaturizer_window,
    pad_spacy_feats,
)
from feature_window.lstmfeaturizer_window import LSTMFeaturizer_window
from feature_window.customfeaturizer_window import (
    get_custom_feats,
    pad_custom_feats,
)


class Featurizer_window(object):
    def __init__(self, feat_config):
        self.doc_sequencer = LSTMFeaturizer_window(
            max_nb_words=feat_config["max_vocab"]
        )
        self.tags2index = {"B-CIT": 1, "I-CIT": 0}
        self.feat_config = feat_config
        self.maxlen = feat_config["window"]

    def fit_transform(self, nlp, textlist_s, taglist_s):
        ### LSTM Features
        data_dict = {}
        textlist_win, taglist_win = sliding_window_list(
            textlist_s,
            taglist_s,
            winSize=self.feat_config["window"],
            step=self.feat_config["step"],
            has_tags=True,
        )
        textlist_flatten = flatten_3d_list(textlist_win)
        taglist_flatten = flatten_3d_list(taglist_win)

        data_dict["sentences_window"] = textlist_flatten
        data_dict["tags_window"] = taglist_flatten
        # pad text and tag
        padded_textarray, padded_tagarray = pad_sequences(
            textlist_flatten, self.maxlen, taglist_flatten
        )
        # encode the label
        encodedLabel = np.array(
            [[self.tags2index[w] for w in s] for s in padded_tagarray]
        )
        data_dict["labels"] = encodedLabel

        if self.feat_config["lstm_feats"]:

            # fit and transform sequence
            word_seq, tokenizer = self.doc_sequencer.fit_transform(
                padded_textarray
            )
            # create data dictionary
            data_dict["lstm_feats"] = word_seq

        if self.feat_config["custom_feats"]:
            custom_feats = get_custom_feats(textlist_s)
            custom_feats_win, dummy = sliding_window_list(
                custom_feats,
                winSize=self.feat_config["window"],
                step=self.feat_config["step"],
                has_tags=False,
            )
            custom_feats_win = pad_custom_feats(
                custom_feats_win, 5, self.maxlen
            )
            custom_feats_flatten = flatten_3d_list(custom_feats_win)
            data_dict["custom_feats"] = np.array(custom_feats_flatten)

        ### Spacy Features
        
        if self.feat_config["spacy_feats"]:
            sp = SpacyFeaturizer_window(nlp)
            df = pd.DataFrame([])
            df["text"] = np.array(
                [" ".join(i) for i in padded_textarray], dtype="object"
            )
            spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df["spacy_bin"])
            spacy_feats = [i.tolist() for i in spacy_feats]
            data_dict["spacy_num_feats"] = np.array(spacy_feats)
        return data_dict, tokenizer, self.maxlen

    def transform(self, nlp, textlist_s, taglist_s=[], has_tags=True):
        data_dict = {}
        # pad text and tag
        if len(taglist_s) == 0:
            has_tags = False
        textlist_win, taglist_win = sliding_window_list(
            textlist_s,
            taglist_s,
            winSize=self.feat_config["window"],
            step=self.feat_config["step"],
            has_tags=has_tags,
        )
        textlist_flatten = flatten_3d_list(textlist_win)
        taglist_flatten = flatten_3d_list(taglist_win)
        data_dict["sentences_window"] = textlist_flatten
        data_dict["tags_window"] = taglist_flatten
        # pad text and tag
        padded_textarray, padded_tagarray = pad_sequences(
            textlist_flatten, self.maxlen, taglist_flatten, has_tags=has_tags
        )
        # encode the label
        if has_tags:
            encodedLabel = np.array(
                [[self.tags2index[w] for w in s] for s in padded_tagarray]
            )
        else:
            encodedLabel = np.array(padded_tagarray)
        data_dict["labels"] = encodedLabel

        if self.feat_config["custom_feats"]:
            custom_feats = get_custom_feats(textlist_s)
            custom_feats_win, dummy = sliding_window_list(
                custom_feats,
                winSize=self.feat_config["window"],
                step=self.feat_config["step"],
                has_tags=False,
            )
            custom_feats_win = pad_custom_feats(
                custom_feats_win, 5, self.maxlen
            )
            custom_feats_flatten = flatten_3d_list(custom_feats_win)
            data_dict["custom_feats"] = np.array(custom_feats_flatten)

        ### LSTM Features
        if self.feat_config["lstm_feats"]:
            # transform sequence
            word_seq = self.doc_sequencer.transform(padded_textarray)

            # create data dictionary
            data_dict["lstm_feats"] = word_seq

        ### Spacy Features
        if self.feat_config["spacy_feats"]:
            sp = SpacyFeaturizer_window(nlp)
            df = pd.DataFrame([])
            df["text"] = np.array(
                [" ".join(i) for i in textlist_flatten], dtype="object"
            )
            if self.feat_config["spacy_mode"] == "production":
                spacy_df = sp.get_array_from_df_combined(df)
            else:
                spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df["spacy_bin"])
            spacy_feats = [i.tolist() for i in spacy_feats]
            spacy_feats_win = pad_spacy_feats(spacy_feats, 12, self.maxlen)
            data_dict["spacy_num_feats"] = np.array(spacy_feats_win)

        return data_dict
