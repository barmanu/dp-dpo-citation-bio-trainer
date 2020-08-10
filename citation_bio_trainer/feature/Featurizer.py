import numpy as np
import pandas as pd
import time
from tensorflow.keras.preprocessing import *
from tensorflow.keras.preprocessing.text import Tokenizer

from util.Utils import pad_sequences
#from feature.SpacyFeaturizer import get_spacy_feats_from_text
from feature.SpacyFeaturizer import SpacyFeaturizer

class Featurizer(object):
    def __init__(self, max_vocab, spacy_feats=False):
        self.doc_sequencer = Sequencer(max_nb_words = max_vocab)
        self.tags2index = {'B-CIT': 1, 'I-CIT': 0}
        self.spacy_feats = spacy_feats
        #self.maxlen = -1
    
    def fit_transform(self, textlist, taglist):
        start_time = time.time()
        data_dict={}
        # pad text and tag
        self.maxlen = np.max([len(i.split(" ")) for i in textlist])
        padded_textarray, padded_tagarray = pad_sequences(textlist, self.maxlen, taglist)
        
        # encode the label
        encodedLabel = np.array([[self.tags2index[w] for w in s] for s in padded_tagarray])
        
        # fit and transform sequence
        word_seq, tokenizer = self.doc_sequencer.fit_transform(padded_textarray)
        
        # create data dictionary
        data_dict['lstm_feats'] = word_seq
        print(self.maxlen)
        print(time.time() - start_time)
        if self.spacy_feats:
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
        data_dict['labels'] = encodedLabel
        
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
        # transform sequence
        word_seq = self.doc_sequencer.transform(padded_textarray)
        
        # create data dictionary
        data_dict['lstm_feats'] = word_seq
        
        if self.spacy_feats:
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
            
        data_dict['labels'] = encodedLabel
        return data_dict
    
    
    
class Sequencer(object):
    def __init__(self, max_nb_words):
        self.max_nb_words = max_nb_words
        self.kerasTokenizer = Tokenizer(num_words=max_nb_words, oov_token=1, filters=None)
        self.word_index = None

    def fit_transform(self, textarray):
        # get maximum sequence length form padding
        #doc_len = text.apply(lambda words: len(words.split(" ")))
        #self.max_seq_len = 20
        self.kerasTokenizer.fit_on_texts(textarray.tolist())
        self.word_index = self.kerasTokenizer.word_index
        # transform training data to sequence
        word_seq = np.array(self.kerasTokenizer.texts_to_sequences(textarray.tolist()))
        return word_seq, self.kerasTokenizer

    def transform(self, textarray):
        word_seq = np.array(self.kerasTokenizer.texts_to_sequences(textarray.tolist()))
        return word_seq