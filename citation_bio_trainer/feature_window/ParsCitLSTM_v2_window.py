import json
import os
import random
import fasttext
import numpy as np
import spacy
from spacy.attrs import POS, ENT_IOB, ENT_ID
import tensorflow as tf
from keras.models import load_model
from spacy.attrs import *
import time
import fasttext
import pandas as pd
import numpy as np
import mlflow.keras
import os, sys
from tqdm import tqdm
sys.path.append('../citation_bio_trainer')
from util.Utils import calulate_ser_jer, load_from_folder, pad_sequences, load_embedding_matrix, evaluate, log_mlflow_results
from sklearn.model_selection import train_test_split
from spacy.tokenizer import Tokenizer as sTokenizer
seed_value = 123456
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
class ParsCitLSTM:
    def __init__(self, model_config: dict, spacy_nlp: object):
        self.__dict__ = model_config
        self.model = load_model(self.model_file)
        self.model._name='parscit_model'
        #self.model.summary()
        self.idx2lab = self.load_json(self.label_dict_file)
        self.nlp = spacy_nlp
        self.nlp.tokenizer = sTokenizer(self.nlp.vocab)
        self.vec_dim = len(self.nlp(".")[0].vector) + 300
        self.dummy_x = np.array([0.] * self.vec_dim)
        self.ftmodel = fasttext.load_model(self.ft_model_path)
    @staticmethod
    def load_json(path):
        d = None
        with open(path) as f:
            d = json.load(f)
        f.close()
        d = dict(zip(d.values(), d.keys()))
        return d
    @staticmethod
    def reshapex(seq, i, j):
        x = seq[:, :, i:j]
        x = np.reshape(x, (len(x), len(x[0])))
        return x
    def predict(self, doc, fasttext_vextors, return_probs=True):
        #print("inside predict")
        _t = [x.text for x in doc]
        _x = np.array([x.vector for x in doc])
        _ems = [np.array(x) for x in fasttext_vextors]
        x = np.concatenate([_ems, _x], axis=-1)
        spacy_nlp = np.array(doc.to_array([POS, ENT_IOB, ENT_ID]))
        t = np.ones_like(spacy_nlp)
        spacy_nlp = np.add(t, spacy_nlp)
        x = np.array([x])
        spacy_nlp = np.array([spacy_nlp])
        #print("before model predict")
        _y0, _y1 = self.model.predict(
            x=[
                x,
                self.reshapex(spacy_nlp, 0, 1),
                self.reshapex(spacy_nlp, 1, 2),
                self.reshapex(spacy_nlp, 2, 3)
            ]
        )
        _y0 = _y0[0]
        if return_probs:
            return _y0
        else:
            _y0 = (_y0 == _y0.max(axis=1, keepdims=True)).astype(int)
            return _y0 
#         _y0 = (_y0 == _y0.max(axis=1, keepdims=True)).astype(int)
#         _y0 = np.argmax(_y0, axis=-1)
#         _y1 = _y1[0]
#         _y1 = np.argmax(_y1, axis=-1)
#         return _t, _y0, _y1

    def get_parscit_list(self, sent_list, return_probs=True):
        ls = []
        for ind in tqdm(range(len(sent_list))):
            text = sent_list[ind]
            text = text.replace('\n', 'क')
            doc = self.nlp(text)
            vec = []
            for x in doc:
                vec.append(self.ftmodel[x.text])
            p1 = self.predict(doc, vec, return_probs=True)
            ls.append(p1)
        return ls
        
    
def pad_dummy_feats(parscit_feats, maxlen):
    parscit_padded = []
    for ind in range(len(parscit_feats)):
        parscit_mask = np.zeros((maxlen, 14), dtype='int8')
        if len(parscit_feats[ind]) <= maxlen:
            parscit_mask[0:len(parscit_feats[ind]), :] = parscit_feats[ind][:]
        else:
            parscit_mask[:] = parscit_feats[ind][0:maxlen,:]
        parscit_padded.append(parscit_mask)
    parscit_arr = np.array([i.tolist() for i in parscit_padded])
    return parscit_arr
    