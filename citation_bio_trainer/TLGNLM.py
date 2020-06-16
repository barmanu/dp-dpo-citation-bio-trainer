import json

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import *


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


def calulate_ser_jer(y_true, y_pred, keep_tag):
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


class BIOLSTM:

    def __init__(self, config):
        """
        :param config_dict:
        """
        self.configs = config

    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(
            hub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                output_shape=[50],
                input_shape=[],
                batch_size=None,
                dtype=tf.string, trainable=True
            )
        )
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.configs["lr"], beta_1=self.configs["beta1"], beta_2=self.configs["beta2"],
                epsilon=self.configs["epsilon"], amsgrad=False,
            ),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self,
                    model: tf.keras.Model,
                    train_data: list,
                    test_data: list,
                    batch: int,
                    epoch: int,
                    verbose: int
                    ):
        """
        :param model:
        :param train_data:
        :param test_data:
        :param batch:
        :param epoch:
        :param verbose:
        :return:
        """

        # train step
        for ep in range(epoch):
            for fpath in train_data:
                df = pd.read_csv(fpath, index_col=0)
                df.fillna("\n", axis=1, inplace=True)
                df["x"] = df.apply(lambda t: change_nl(t.x), axis=1)
                df["y"] = df.apply(lambda x: 1.0 if x.y.startswith("B") else 0.0, axis=1)
                model.fit(
                    x=df.x,
                    y=df.y,
                    batch_size=batch,
                    epochs=1,
                    verbose=verbose,
                )
                aser = 0.0
                ajer = 0.0
                for fpath in test_data:
                    df = pd.read_csv(fpath, index_col=0)
                    df.fillna("\n", axis=1, inplace=True)
                    df["x"] = df.apply(lambda t: change_nl(t.x), axis=1)
                    df["y"] = df.apply(lambda x: 1.0 if x.y.startswith("B") else 0.0, axis=1)
                    d = {
                        "x": df.x.tolist(),
                        "y": df.y.tolist()
                    }
                    _, ser, jer = self.eval_model(model, d)
                    ajer += jer
                    aser += ser
                aser /= float(len(test_data))
                ajer /= float(len(test_data))
                print(f" AVG SER {aser} JER {ajer}")
        return model

    def eval_model(self, model: tf.keras.Model, eval_data: dict):
        """
        :param model:
        :param eval_data:
        :param label_dict:
        :return:
        """
        g = eval_data["y"]
        pred_y = model.predict(x=eval_data["x"])
        p = []
        for ys in pred_y:
            if ys > 0.5:
                p.append(1.0)
            else:
                p.append(0.0)
        rdict = {}
        ser, jer = calulate_ser_jer(g, p, 1)
        rdict["ser"] = ser
        rdict["jer"] = jer
        return rdict, ser, jer


if __name__ == '__main__':
    import os

    files = []
    for fname in os.listdir(
            "/Users/barmanu/Work/dp-dpo-citation-bio-trainer/nlp/exps/output/2020-06-15~07:42:04.444674"):
        if fname.startswith("data-2020"):
            p = os.path.join(
                "/Users/barmanu/Work/dp-dpo-citation-bio-trainer/nlp/exps/output/2020-06-15~07:42:04.444674/", fname)
            files.append(p)

    train_count = 0.8 * len(files)
    train_count = int(min(train_count, len(files) - 1))

    train_data = files[:train_count]
    test_data = files[train_count:]

    config = load_json("../config/model_config.json")

    trainer = BIOLSTM(config)
    model = trainer.get_model()

    model = trainer.train_model(model, train_data, test_data, 64, 10, 2)
