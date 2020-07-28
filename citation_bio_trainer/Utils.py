import json
import os
from sklearn.metrics import *
import pandas as pd
import numpy as np
def load_json(path):
    d = None
    with open(path) as f:
        d = json.load(f)
    f.close()
    return d


def change_nl(x):
    if "\n" in x:
        return "MWLN"
    else:
        return x

    
def calulate_ser_jer(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :param keep_tag:
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    ser = 1.0
    if (tp + fp) > 0.0:
        ser = fp / float(tp + fp)
    jer = 1.0
    if (tp + fn) > 0.0:
        jer = fn / float(tp + fn)
    return ser, jer


def load_from_folder(folderpath, pattern):
    ''' 
    This function takes input folder path and return lists of seqeunces and tags
    param:
    folderpath: path of the folder containing the files
    return:
    sentences: sentences from the file
    sent_tags: tags for each words in sentences
    '''
    
     ## to make sure it splits consistently
    sentences = []
    sent_tags = []
    len_arr = []
    for fpath in os.listdir(folderpath):
        if fpath not in ['data-gen-config.json', 'data_generation_stats.csv'] and ".csv" in fpath:
            fpath = os.path.join(folderpath, fpath)
            df = pd.read_csv(fpath, index_col=0)
            df.fillna("\n", axis=1, inplace=True)
            #df.x = df.x.apply(lambda word: word.strip())
            sentences.append(pattern.join(df.x))
            sent_tags.append(pattern.join(df.y))
    return sentences, sent_tags


def pad_sequences(sentences, sent_tags, maxlen, pattern):
    '''
    This function pads seqences to make them of same length
    '''
    X = [[w for w in s.split(pattern)] for s in sentences]
    y = [[p for p in t.split(pattern)] for t in sent_tags]
    new_X = []
    new_y = []
    for ind in range(len(X)):
        new_seq = []
        new_tag = []
        for i in range(maxlen):
            try:
                new_seq.append(X[ind][i])
                new_tag.append(y[ind][i])
            except:
                new_seq.append("PADword")
                new_tag.append("I-CIT")
        new_X.append(new_seq)
        new_y.append(new_tag)
        
    return new_X, new_y

def load_embedding_matrix(embeddings, nb_words, word_index, embed_dim):  # load embeddings
    embeddings_index = {}
    for word in embeddings.wv.vocab:
        embeddings_index[word] = embeddings[word]
    print('Found %s word vectors.' % len(embeddings_index))

    words_not_found = []
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings[word]
            embedding_matrix[i] = embedding_vector
        except:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


def evaluate(true_labels, pred_labels):
    '''
    This function returns ser, jer and count of mistakes on a dataset
    '''
    ser_jer = [calulate_ser_jer(i,j) for i,j in zip(true_labels, pred_labels)]
    accuracy = [accuracy_score(i,j) for i,j in zip(true_labels, pred_labels)]
    