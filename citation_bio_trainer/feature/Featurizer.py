import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing import *
from util.Utils import pad_sequences
from feature.SpacyFeaturizer import SpacyFeaturizer
from feature.LSTMFeaturizer import LSTMFeaturizer

class Featurizer(object):
    def __init__(self, feat_config):
        self.doc_sequencer = LSTMFeaturizer(max_nb_words = feat_config['max_vocab'])
        self.tags2index = {'B-CIT': 1, 'I-CIT': 0}
        self.feat_config = feat_config
        #self.maxlen = -1
    
    def fit_transform(self, textlist, taglist):
        ### LSTM Features
        data_dict={}
        # pad text and tag
        self.maxlen = np.max([len(i.split(" ")) for i in textlist])
        padded_textarray, padded_tagarray = pad_sequences(textlist, self.maxlen, taglist)
        # encode the label
        encodedLabel = np.array([[self.tags2index[w] for w in s] for s in padded_tagarray])
        data_dict['labels'] = encodedLabel
        
        if self.feat_config['lstm_feats']:
            
            # fit and transform sequence
            word_seq, tokenizer = self.doc_sequencer.fit_transform(padded_textarray)
            # create data dictionary
            data_dict['lstm_feats'] = word_seq

        ### Spacy Features
        if self.feat_config['spacy_feats']:
            sp = SpacyFeaturizer()
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
            
        if self.feat_config['parscit_feats']:
            sp = SpacyFeaturizer()
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
        padded_textarray, padded_tagarray = pad_sequences(textlist, self.maxlen, taglist, has_tags)
        if has_tags:
            encodedLabel = np.array([[self.tags2index[w] for w in s] for s in padded_tagarray])
        else:
            encodedLabel = np.array(padded_tagarray)
        data_dict['labels'] = encodedLabel
        
        ### LSTM Features
        if self.feat_config['lstm_feats']:
            # transform sequence
            word_seq = self.doc_sequencer.transform(padded_textarray)

            # create data dictionary
            data_dict['lstm_feats'] = word_seq
        
        ### Spacy Features
        if self.feat_config['spacy_feats']:
            sp = SpacyFeaturizer()
            df = pd.DataFrame([])
            df['text'] = np.array(textlist, dtype='object')
            spacy_df = sp.get_spacy_dask(df, blocksize=1000)
            spacy_feats = list(spacy_df['spacy_bin']) 
            spacy_feats_padded = []
            for ind in range(len(spacy_feats)):
                spacy_mask = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],]*self.maxlen)
                if len(spacy_feats[ind]) <= self.maxlen:
                    spacy_mask[0:len(spacy_feats[ind]), :] = spacy_feats[ind][:]
                else:
                    spacy_mask[:] = spacy_feats[ind][0:self.maxlen,:]
                spacy_feats_padded.append(spacy_mask)
            data_dict['spacy_num_feats'] = np.array([i.tolist() for i in spacy_feats_padded])
            
        
        return data_dict
    
    
    
