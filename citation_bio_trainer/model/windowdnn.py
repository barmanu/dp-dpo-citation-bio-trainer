import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import *
from sklearn.preprocessing import *
from tqdm import tqdm


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


class WindowDNN:
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
            "epoch": 10,
            "window":" 3,
            "vocab": 10000,
            "embedding_dim":50,
            "hidden_dim": 100
        }
        """
        self.config = config

    def get_model(self):
        """
        :return: tf.keras.Model
        """

        inputs = []

        embedding_in = {}
        for i in range((self.config.get("window", 3) * 2) + 1):
            _in = tf.keras.layers.Input(shape=(None,), name=str(i))
            inputs.append(_in)
            embedding_in[str(i + 1)] = _in

        embedding = tf.keras.layers.Embedding(
            input_dim=self.config.get("vocab", 10000),
            output_dim=self.config.get("embedding_dim", 100),
            mask_zero=False,
            input_length=None,
            trainable=True,
            name="embedding",
        )

        embedding_out = []
        for k, v in embedding_in.items():
            embedding_out.append(embedding(v))

        x = tf.keras.layers.concatenate(
            inputs=embedding_out, axis=-1, name="cat"
        )
        x = tf.keras.layers.Permute((2, 1))(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=2,
            strides=None,
            padding="valid",
            data_format="channels_last",
        )(x)
        x = tf.keras.layers.Permute((2, 1))(x)
        if self.config.get("drop_rate", 0.0) > 0.0:
            x = tf.keras.layers.Dropout(self.config.get("drop_rate", 0.0))(x)
        x_out = tf.keras.layers.Dense(
            self.config.get("hidden_dim", 10000),
            activation="relu",
            name="hidden",
        )(x)

        out = tf.keras.layers.Dense(1, activation="sigmoid", name="out")(x)

        model = tf.keras.Model(inputs=inputs, outputs=out)
        opt = tf.keras.optimizers.Adam(
            learning_rate=self.config.get("lr.csv", 0.001),
            beta_1=self.config.get("beta1", 0.9),
            beta_2=self.config.get("beta2", 0.999),
            epsilon=self.config.get("lr.csv", 1e-07),
        )
        model.compile(
            optimizer=opt, loss="binary_crossentropy", metrics=["acc"]
        )
        model.summary()
        return model

    def prepare_data(self, file_paths):
        lenc = LabelEncoder()
        words = set()
        words.add("_PADD_")
        max_len = -1
        for path in tqdm(os.listdir(file_paths)):
            path = os.path.join(file_paths, path)
            df = pd.read_csv(path, index_col=0)
            df.fillna("\n", axis=1, inplace=True)
            for w in set(df.x.tolist()):
                words.add(w)
            if len(df) > max_len:
                max_len = len(df)
        words = list(words)
        words.sort()
        lenc.fit(words)
        x = []
        y = []
        for path in tqdm(os.listdir(file_paths)):
            x_all = []
            path = os.path.join(file_paths, path)
            df = pd.read_csv(path, index_col=0)
            df.fillna("\n", axis=1, inplace=True)
            tx = df.x.tolist()
            ty = df.y.tolist()
            ty = [1 if x.startswith("B") else 0 for x in ty]
            rem = max_len - len(tx)
            if rem > 0:
                for i in range(rem):
                    tx.append("_PADD_")
                    ty.append(0)

            window_size = self.config.get("window", 3)
            for i in range(window_size):
                pre = tx.copy()
                for j in range(i + 1):
                    pre.insert(0, "_PADD_")
                pre = pre[: len(tx)]
                x_all.append(lenc.transform(pre))
            x_all.insert(0, lenc.transform(tx))
            for i in range((window_size)):
                post = tx[i + 1 :]
                for i in range(len(tx) - len(post)):
                    post.append("_PADD_")
                x_all.insert(0, lenc.transform(post))
            x.append(x_all)
            y.append(ty)
        x = np.array(x)
        temp_dict = {}
        for i in range(len(x)):
            doc = x[i]
            for j in range(len(doc)):
                if j in temp_dict.keys():
                    v = temp_dict[j]
                    v = np.vstack((v, doc[j]))
                else:
                    v = doc[j]
                temp_dict[j] = v
        x = [temp_dict[k] for k in temp_dict.keys()]
        y = np.reshape(np.array(y), (len(y), len(y[0]), 1))
        self.config["vocab"] = len(words)
        return x, y, words

    @staticmethod
    def train(config, output_dir):
        res = []
        m = WindowDNN(config)
        x, y, words = m.prepare_data(output_dir)
        train_count = int(len(y) * float(m.config.get("train_per", 0.9)))
        train_x = []
        test_x = []
        train_y = y[:train_count, :]
        test_y = y[train_count:, :]
        for i in range(len(x)):
            tempx = x[i]
            train_x.append(tempx[:train_count, :])
            test_x.append(tempx[train_count:, :])
        model = m.get_model()
        for i in range(m.config.get("epoch", 3)):
            model.fit(
                x=train_x,
                y=train_y,
                validation_data=(test_x, test_y),
                epochs=1,
                batch_size=m.config.get("batch", 8),
                verbose=0,
            )
            y_pred = model.predict(test_x)
            y_pred = y_pred.flatten()
            y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
            y_true = test_y.flatten()
            ser, jer = calulate_ser_jer(y_true, y_pred)
            res.append({"epoch": i, "SER": ser, "JER": jer})
            print({"epoch": i, "SER": ser, "JER": jer})
        history = pd.DataFrame(res)
        return model, history
