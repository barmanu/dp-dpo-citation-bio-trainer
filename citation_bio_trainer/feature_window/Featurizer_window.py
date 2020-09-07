import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing import *
from util.Utils import pad_sequences, sliding_window_list, flatten_3d_list
from feature_window.SpacyFeaturizer_window import SpacyFeaturizer_window, pad_spacy_feats
from feature_window.LSTMFeaturizer_window import LSTMFeaturizer_window
from feature_window.CustomFeaturizer_window import get_custom_feats, pad_custom_feats

class Featurizer_window(object):
    def __init__(self, feat_config):
        self.doc_sequencer = LSTMFeaturizer_window(max_nb_words = feat_config['max_vocab'])
        self.tags2index = {'B-CIT': 1, 'I-CIT': 0}
        self.feat_config = feat_config
        self.maxlen = feat_config['window']

    
    def fit_transform(self, textlist, taglist):
        ### LSTM Features
        data_dict={}
        
        textlist_s = [i.split(" ") for i in textlist]
        taglist_s = [i.split(" ") for i in taglist]
        textlist_win, taglist_win = sliding_window_list(textlist_s, taglist_s)
        textlist_flatten = flatten_3d_list(textlist_win)
        taglist_flatten = flatten_3d_list(taglist_win)
        
        data_dict['sentences_window'] = textlist_flatten
        data_dict['tags_window'] = taglist_flatten
        # pad text and tag
        padded_textarray, padded_tagarray = pad_sequences(textlist_flatten, self.maxlen, taglist_flatten)
        # encode the label
        encodedLabel = np.array([[self.tags2index[w] for w in s] for s in padded_tagarray])
        data_dict['labels'] = encodedLabel
                
        if self.feat_config['lstm_feats']:
            
            # fit and transform sequence
            word_seq, tokenizer = self.doc_sequencer.fit_transform(padded_textarray)
            # create data dictionary
            data_dict['lstm_feats'] = word_seq
        
        if self.feat_config['custom_feats']:
            custom_feats = get_custom_feats(textlist_s)
            custom_feats_win,dummy = sliding_window_list(custom_feats, has_tags=False)
            custom_feats_win = pad_custom_feats(custom_feats_win, 5, self.maxlen)
            custom_feats_flatten = flatten_3d_list(custom_feats_win)
            data_dict['custom_feats'] = np.array(custom_feats_flatten)
            
        ### Spacy Features
        if self.feat_config['spacy_feats']:
            sp = SpacyFeaturizer_window()
            df = pd.DataFrame([])
            df['text'] = np.array([" ".join(i) for i in padded_textarray], dtype='object')
            spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df['spacy_bin']) 
            spacy_feats = [i.tolist() for i in spacy_feats]
            data_dict['spacy_num_feats'] = np.array(spacy_feats)
            
        if self.feat_config['parscit_feats']:
            sp = SpacyFeaturizer_window()
            df = pd.DataFrame([])
            df['text'] = np.array(textlist, dtype='object')
            spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df['spacy_bin']) 
            spacy_feats_padded = []
            for ind in range(len(spacy_feats)):
                spacy_mask = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],]*self.maxlen)
                spacy_mask[0:len(spacy_feats[ind]), :] = spacy_feats[ind][:]
                spacy_feats_padded.append(spacy_mask)
            data_dict['spacy_num_feats'] = np.array([i.tolist() for i in spacy_feats_padded])
        
        
        return data_dict, tokenizer, self.maxlen
    
    def transform(self, textlist, taglist=[], has_tags=True):
        data_dict={}
        # pad text and tag
        if len(taglist) == 0:
            has_tags = False 
        
        textlist_s = [i.split(" ") for i in textlist]
        taglist_s = [i.split(" ") for i in taglist]
        textlist_win, taglist_win = sliding_window_list(textlist_s, taglist_s)
        textlist_flatten = flatten_3d_list(textlist_win)
        taglist_flatten = flatten_3d_list(taglist_win)
        data_dict['sentences_window'] = textlist_flatten
        data_dict['tags_window'] = taglist_flatten
        # pad text and tag
        padded_textarray, padded_tagarray = pad_sequences(textlist_flatten, self.maxlen, taglist_flatten)
        # encode the label
        if has_tags:
            encodedLabel = np.array([[self.tags2index[w] for w in s] for s in padded_tagarray])
        else:
            encodedLabel = np.array(padded_tagarray)
        data_dict['labels'] = encodedLabel
            
        if self.feat_config['custom_feats']:
            custom_feats = get_custom_feats(textlist_s)
            custom_feats_win,dummy = sliding_window_list(custom_feats, has_tags=False)
            custom_feats_win = pad_custom_feats(custom_feats_win, 5, self.maxlen)
            custom_feats_flatten = flatten_3d_list(custom_feats_win)
            data_dict['custom_feats'] = np.array(custom_feats_flatten)
            
            
        ### LSTM Features
        if self.feat_config['lstm_feats']:
            # transform sequence
            word_seq = self.doc_sequencer.transform(padded_textarray)

            # create data dictionary
            data_dict['lstm_feats'] = word_seq
        
        ### Spacy Features
        if self.feat_config['spacy_feats']:
            sp = SpacyFeaturizer_window()
            df = pd.DataFrame([])
            df['text'] = np.array([" ".join(i) for i in textlist_flatten], dtype='object')
            spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df['spacy_bin']) 
            spacy_feats = [i.tolist() for i in spacy_feats]
            spacy_feats_win = pad_spacy_feats(spacy_feats, 12, self.maxlen)
            data_dict['spacy_num_feats'] = np.array(spacy_feats_win)
            
        return data_dict
    
    
    
