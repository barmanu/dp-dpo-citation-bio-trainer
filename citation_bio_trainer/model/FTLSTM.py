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
import matplotlib.pyplot as plt


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

# class Metrics(callbacks.Callback):
#     def __init__(self, train_data, validation_data):
#         super(Metrics, self).__init__()
#         self.validation_data = validation_data
#         self.train_data = train_data
        
#     def on_train_begin(self, logs={}):
#         self.val_ser = []
#         self.val_jer = []
        
#     def on_epoch_end(self, epoch, logs):
#         vx = self.validation_data[0]
#         vy = self.validation_data[1]
#         vx = np.array(vx)
#         pred_y = None
#         pred_y = self.model.predict(vx)
#         py = np.argmax(pred_y, axis=-1)
#         #vy = np.argmax(vy, axis=-1)
#         vy = np.array(vy).flatten()
#         py = np.array(py).flatten()
#         ser, jer = calulate_ser_jer(vy, py)
#         print(ser, jer)
#         self.val_ser.append(ser)
#         self.val_jer.append(jer)
#         logs["val_ser"] = ser
#         logs["val_jer"] = jer
#         print(f"— val_ser: {ser} — val_jer: {jer}")
#         return 
    
    
    
def get_model(feat_config, model_config, maxlen):
    input = Input(shape=(maxlen,))
    embed = Embedding(input_dim=feat_config['max_vocab'], input_length=maxlen, output_dim=300, weights=[model_config['embedding']['matrix']], trainable=model_config['trainable'])(input)
    if model_config['lstm']['use']:
        for ind in range(model_config['lstm']['num']):
            if ind ==0:
                y = LSTM(model_config['lstm']['units'], return_sequences=True)(embed)
            else:
                y = LSTM(model_config['lstm']['units'], return_sequences=True)(y)
    
    if model_config['aux_feats']['use']:
        aux_feats = Input(shape=(maxlen,model_config['aux_feats']['dim']))
        y = Concatenate()([y, aux_feats])
        
    if model_config['dense']['use']:
        y =  Dense(model_config['dense']['units'], activation=model_config['dense']['activation'])(y)
        
    if model_config['timedistributed']['use']:
        y = TimeDistributed(Dense(1, activation=model_config['output_activation']))(y)
    else:
        y = Dense(1, activation=model_config['output_activation'])(y)
        
    if model_config['aux_feats']['use']:
        model = Model(inputs=[input, aux_feats], outputs=y)
    else:
        model = Model(inputs=input, outputs=y)
        
    model.compile(optimizer=model_config['optimizer'],
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_output(history):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].title.set_text('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epoch')
    axes[0].legend(['train', 'val'], loc='upper left')

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].title.set_text('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epoch')
    axes[1].legend(['train', 'val'], loc='upper left')
    fig.tight_layout()