import os

import git
import mlflow.keras
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import *

from citation_bio_trainer.Utils import *


def calulate_ser_jer(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
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

    def __init__(self, config: dict):
        """
        :param config
        {
            "lr.csv": 0.005,
            "clip": 0.0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-07,
            "l1": 0.000,
            "l2": 0.00,
            "drop_rate": 0.00,
            "batch": 64,
            "epoch": 5,
            "embedding_in":" "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
            "embedding_out":50
        }
        """
        self.config = config

    def get_model(self):
        """
        :return: tf.keras.Model
        """
        model = tf.keras.models.Sequential()
        model.add(
            hub.KerasLayer(
                self.config.get("embedding_in", "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"),
                output_shape=[self.config.get("embedding_out", 50)],
                input_shape=[],
                batch_size=None,
                dtype=tf.string, trainable=True
            )
        )
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.get("lr.csv", 0.005),
                beta_1=self.config.get("beta1", 0.0),
                beta_2=self.config.get("beta2", 0.0),
                epsilon=self.config.get("epsilon", 1e-05),
                amsgrad=False,
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
        return model

    def get_flat_model(self):

        model = tf.keras.Sequential()
        model.add()

    def train_model(self,
                    model: tf.keras.Model,
                    train_data: list,
                    test_data: list,
                    batch: int,
                    epoch: int,
                    verbose: int
                    ):
        """
        :param model: tf.keras.Model, model object
        :param train_data: list, train df paths
        :param test_data: list, test df paths
        :param batch: int, batch size
        :param epoch: int, epochs
        :param verbose: int, verbose
        :return: (tf.keras.Model, pd.DataFrame), (trained_model, train_history)
        """
        train_history = []

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
            train_history.append({"epoch": (ep + 1), "SER": aser, "JER": ajer})
            print(f" Epoch {(ep + 1)} SER {aser} JER {ajer}")
        return model, pd.DataFrame(train_history)

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

    path = "../nlp/exps/output/2020-06-15~07:42:04.444674"
    files = []
    for fname in os.listdir(path):
        if fname.startswith("data-2020"):
            p = os.path.join(path, fname)
            files.append(p)

    train_count = 0.8 * len(files)
    train_count = int(min(train_count, len(files) - 1))

    train_data = files[:train_count]
    test_data = files[train_count:]

    config = load_json("../config/model_config.json")

    trainer = BIOLSTM(config)
    model = trainer.get_model()

    model, history_df = trainer.train_model(model, train_data, test_data, 64, 5, 2)
    history_df.to_csv("training.history.csv")

    # init: name, output-dir, timestamp
    repo = git.Repo(search_parent_directories=True)
    exp_name = repo.remote("origin").url
    mlflow.set_tracking_uri("https://mlflow.caps.dev.dp.elsevier.systems")
    mlflow.set_experiment(exp_name)
    for ix, row in history_df.iterrows():
        mlflow.start_run()
        d = dict(load_json(os.path.join(path, "data-gen-config.json")))
        for k, v in d.items():
            if k not in ["data_df", "target_tag"]:
                mlflow.log_param("data-gen-param-" + k, v)
            for k, v in config.items():
                if type(v) == str or type(v) == float or type(v) == int:
                    if "epoch" not in k:
                        mlflow.log_param(k, v)
            mlflow.log_param("train_count", train_count)
            mlflow.log_param("test_count", (len(files) - train_count))
            mlflow.log_param("epoch", row["epoch"])
            mlflow.log_metric("ser", row["SER"])
            mlflow.log_metric("jer", row["JER"])
            mlflow.end_run()
