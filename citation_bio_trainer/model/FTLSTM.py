from tensorflow.keras import callbacks
import numpy as np
import json
import os
from sklearn.metrics import *
import pandas as pd
import numpy as np
import mlflow
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *

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

class Metrics(callbacks.Callback):
    def __init__(self, train_data, validation_data):
        super(Metrics, self).__init__()
        self.validation_data = validation_data
        self.train_data = train_data
        
    def on_train_begin(self, logs={}):
        self.val_ser = []
        self.val_jer = []
        
    def on_epoch_end(self, epoch, logs):
        vx = self.validation_data[0]
        vy = self.validation_data[1]
        vx = np.array(vx)
        pred_y = None
        pred_y = self.model.predict(vx)
        py = np.argmax(pred_y, axis=-1)
        #vy = np.argmax(vy, axis=-1)
        vy = np.array(vy).flatten()
        py = np.array(py).flatten()
        ser, jer = calulate_ser_jer(vy, py)
        print(ser, jer)
        self.val_ser.append(ser)
        self.val_jer.append(jer)
        logs["val_ser"] = ser
        logs["val_jer"] = jer
        print(f"— val_ser: {ser} — val_jer: {jer}")
        return 
    
    
    
def get_model(feat_config, model_config, maxlen):
    input = Input(shape=(maxlen,))
    embed = Embedding(input_dim=feat_config['max_vocab'], input_length=maxlen, output_dim=300, weights=[embedding_matrix], trainable=model_config['trainable'])(input)
    if model_config['lstm']['use']:
        for ind in range(model_config['lstm']['num']):
            if ind ==0:
                lstm = LSTM(model_config['lstm']['units'], return_sequences=True)(embed)
            else:
                lstm = LSTM(model_config['lstm']['units'], return_sequences=True)(lstm)
    
    aux_feats = Input(shape=(maxlen,26))
    conc = Concatenate([lstm, aux_feats])
    td = TimeDistributed(Dense(1, activation=model_config['output_activation']))(lstm)
    model = Model(inputs=[input, aux_feats], outputs=td)
    model.compile(optimizer=model_config['optimizer'],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model