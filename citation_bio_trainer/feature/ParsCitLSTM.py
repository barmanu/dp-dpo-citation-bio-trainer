import numpy as np
import os, json
import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model
from spacy.attrs import POS, ENT_IOB, ENT_ID
import random
import dask.dataframe as dd
import time
from spacy.tokenizer import Tokenizer


seed_value = 123456
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
from tqdm import tqdm
class ParsCitLSTM:
    def __init__(self, model_config: dict):
        self.__dict__ = model_config
        self.model = load_model(self.model_file)
        self.model._name='parscit_model'
        #self.model.summary()
        self.idx2lab = self.load_json(self.label_dict_file)
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab)
        self.tfh_model = hub.load(self.tfhub_model_dir)
        self.vec_dim = len(self.nlp(".")[0].vector) + 512
        self.dummy_x = np.array([0.] * self.vec_dim)
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
    def predict(self, text: str):
        doc = self.nlp(text)
        _t = [x.text for x in doc]
        _x = np.array([x.vector for x in doc])
        _ems = [np.array(x) for x in self.tfh_model(_t)]
        x = np.concatenate([_ems, _x], axis=-1)
        spacy_nlp = np.array(doc.to_array([POS, ENT_IOB, ENT_ID]))
        t = np.ones_like(spacy_nlp)
        spacy_nlp = np.add(t, spacy_nlp)
        x = np.array([x])
        spacy_nlp = np.array([spacy_nlp])
        _y0, _y1 = self.model.predict(
            x=[
                x,
                self.reshapex(spacy_nlp, 0, 1),
                self.reshapex(spacy_nlp, 1, 2),
                self.reshapex(spacy_nlp, 2, 3)
            ]
        )
        _y0 = _y0[0]
        _y0 = np.argmax(_y0, axis=-1)
        _y1 = _y1[0]
        _y1 = np.argmax(_y1, axis=-1)
        return _t, _y0, _y1
    
    def dask_predict(self, df):
        #start_time = time.time()
        sent_list = df.text.tolist()
        text = " ".join(sent_list)
        if self.nlp.max_length < len(text):
            self.nlp.max_length = 1 + len(text)
        text = text.replace('\n', 'क')
        #print("before nlp fit %0.4f"%(time.time() - start_time))
        doc = self.nlp(text)
        #print("after nlp fit %0.4f"%(time.time() - start_time))
        _t = [x.text for x in doc]
        _x = np.array([x.vector for x in doc])
        _ems = [np.array(x) for x in self.tfh_model(_t)]
        #print("after embedding %0.4f"%(time.time() - start_time))
        x = np.concatenate([_ems, _x], axis=-1)
        spacy_nlp = np.array(doc.to_array([POS, ENT_IOB, ENT_ID]))
        t = np.ones_like(spacy_nlp)
        spacy_nlp = np.add(t, spacy_nlp)
        x = np.array([x])
        spacy_nlp = np.array([spacy_nlp])
        #print("before final model %0.4f"%(time.time() - start_time))
        _y0, _y1 = self.model.predict(
            x=[
                x,
                self.reshapex(spacy_nlp, 0, 1),
                self.reshapex(spacy_nlp, 1, 2),
                self.reshapex(spacy_nlp, 2, 3)
            ]
        )
        #print("after final model %0.4f"%(time.time() - start_time))
        _y0 = _y0[0]
        _y0 = (_y0 == _y0.max(axis=1, keepdims=True)).astype(int)
        split_ind = list(np.cumsum([len(a.split(" ")) for a in sent_list]))
        split_ind.pop()
        feat = np.split(_y0, split_ind)
        parscit_df = pd.DataFrame([], columns=['parscit_feat'])
        parscit_df['parscit_feat'] = feat
        return parscit_df#[i.tolist() for i in feat]
    
    def get_parscit_blocks(self, df, chunk_size=100):
        ls = []
        for start in tqdm(range(0, df.shape[0], chunk_size)):
            df_subset = df.iloc[start:start + chunk_size]
            ls.append(self.dask_predict(df_subset))
        return pd.concat(ls)
        
if __name__ == '__main__':
    path = "/Users/barmanu/Work/dp-dpo-citation-bio-trainer/citation_lstms/data.csv"
    df = pd.read_csv(path)
    arr = df.columns
    tx_arr = []
    lb_arr = []
    for i in arr:
        if "text" in i:
            tx_arr = df[i].tolist()
        if "label" in i:
            lb_arr = df[i].tolist()
    arr_tx = eval(tx_arr[7])
    arr_lb = eval(lb_arr[7])
    text = " ".join(arr_tx)
    print(f"TEXT {text} \n\n")
    c = {
        "model_file": "/nlp/parscit/input_dim:608~hidden_dim:600~output_dim:14~lr:0.01~clip:5.0~beta1:0.9~beta2:0.999~l1:0.0~l2:0.0~drop_rate:0.05~batch:64~epoch:30~crf:False~rnn:True~num_of_rnn:1~s1:19~s2:5~s3:3.model-epoch-29.h5",
        "label_dict_file": "/nlp/parscit/labels.json",
        "tfhub_model_dir": "/nlp/parscit/resource/"}
    model = ParsCitLSTM(model_config=c)
    t, y1, y2 = model.predict(text)
    for token, cat in zip(t, y1):
        print(f"{token} {cat}")