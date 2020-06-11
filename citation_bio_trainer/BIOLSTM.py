import pandas as pd
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.utils import *
from sklearn.metrics import *


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


class Metrics(callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.val_ser = []
        self.val_jer = []

    def on_epoch_end(self, epoch, logs):
        vx = self.validation_data[0]
        vy = self.validation_data[1]
        vx = np.array(vx)
        vy = np.array(self.validation_data[1])
        pred_y = None
        if "extra" in self.model.name:
            pred_y, _ = self.model.predict(vx)
        else:
            pred_y = self.model.predict(vx)
        py = []
        for ys in pred_y:
            for y in ys:
                if y > 0.5:
                    py.append(1)
                else:
                    py.append(0)
        vy = np.array(vy).flatten()
        py = np.array(py).flatten()
        ser, jer = calulate_ser_jer(vy, py, 1)
        self.val_ser.append(ser)
        self.val_jer.append(jer)
        logs["val_ser"] = ser
        logs["val_jer"] = jer
        print(f"— val_ser: {ser} — val_jer: {jer}")
        return


class BIOLSTM:

    def __init__(self, config):
        """
        :param config_dict:
        """
        self.__dict__ = config
        tmp = []
        for k, v in config.items():
            tmp.append(k[:min(5, len(k))] + "-" + str(v) + ":")
        self.name = "~".join(tmp) + ".model"

    def get_model(self):
        """
        :return: model
        """

        inputs = Input(
            shape=(None, self.input_dim,),
        )

        masked = Masking(mask_value=0., input_shape=(None, self.input_dim,))

        lstm = None

        if self.l2 == 0.0:
            lstm = Bidirectional(
                LSTM(
                    units=self.hidden_dim,
                    return_sequences=True,
                    name="lstm-0"
                ),
                merge_mode="concat",
                name="b_cat_rnn"
            )
        else:
            lstm = Bidirectional(
                LSTM(
                    units=self.hidden_dim,
                    kernel_regularizer=l2(self.l2),
                    bias_regularizer=l2(self.l2),
                    return_sequences=True,
                    name="lstm-0"
                ),
                merge_mode="concat",
                name="b_cat_rnn"
            )

        drop = Dropout(
            self.drop_rate,
            name="drop"
        )

        dense = TimeDistributed(
            Dense(
                1,
                activation='sigmoid',
            ),
            name="output"
        )
        dense_ = None
        if "extra" in self.problem_type:
            dense_ = TimeDistributed(
                Dense(
                    2,
                    activation='softmax',
                ),
                name="extra"
            )

        optim = Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2, amsgrad=False)

        x = masked(inputs)
        if self.rnn:
            x = drop(lstm(x))
            rem = self.num_of_rnn - 1
            if rem > 0:
                for il in range(rem):
                    x = LSTM(
                        units=self.hidden_dim,
                        return_sequences=True,
                        name="lstm-" + str(il + 1)
                    )(x)

        outputs = dense(x)
        if "extra" in self.problem_type:
            extra = dense_(x)
            model = Model(
                inputs=inputs,
                outputs=[outputs, extra],
                name=self.name
            )
            model.compile(
                optimizer=optim,
                loss={
                    "output": "mean_squared_error",
                    "extra": "binary_crossentropy"
                },
                metrics={
                    "output": "binary_accuracy",
                    "extra": "binary_accuracy"
                },
            )
        else:
            model = Model(
                inputs=inputs,
                outputs=outputs,
                name=self.name
            )
            model.compile(
                optimizer=optim,
                loss={
                    "output": "mean_squared_error"
                },
                metrics={
                    "output": "binary_accuracy"
                },
            )

        model.summary()
        return model

    @staticmethod
    def np_array_2_cv_split(x, y, split=5):
        """
        :param x:
        :param x1:
        :param y:
        :param split:
        :return:
        """
        sample_per_split = int(len(x) / float(split))
        cv_dict = {}
        x_splits = []
        y_splits = []
        start = 0
        for i in range(split - 1):
            x_splits.append(x[start:min((start + sample_per_split), len(x)), :])
            y_splits.append(y[start:min((start + sample_per_split), len(y)), :])
            start += sample_per_split

        x_splits.append(x[start:len(x), :])
        y_splits.append(y[start:len(y), :])

        for i in range(split):
            test_x = x_splits[i]
            test_y = y_splits[i]
            train_x = []
            train_y = []
            for j in range(split):
                if i != j:
                    train_x.append(x_splits[j])
                    train_y.append(y_splits[j])
            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)
            cv_dict[(i + 1)] = {
                "train_x": np.array(train_x),
                "train_y": np.array(train_y),
                "test_x": np.array(test_x),
                "test_y": np.array(test_y),
            }
        return cv_dict

    @staticmethod
    def reshapex(seq, i, j):
        """
        :param seq:
        :param i:
        :param j:
        :return:
        """
        x = seq[:, :, i:j]
        x = np.reshape(x, (len(x), len(x[0])))
        return x

    def cross_validate(self, x: np.array, y: np.array, batch: int, epoch: int, label_dict: dict, split=5,
                       verbose=2):
        """
        :param x:
        :param y:
        :param batch:
        :param epoch:
        :param label_dict:
        :param split:
        :param verbose:
        :return:
        """
        cv_dict = BIOLSTM.np_array_2_cv_split(x=x, y=y, split=split)
        cv_models = []
        cls_report = []
        fc = 1
        for k, v in cv_dict.items():
            model = self.get_model()
            hist_df, model, _ = self.train_model(
                model,
                train_data={
                    "x": v["train_x"],
                    "y": v["train_y"],
                },
                test_data={
                    "x": v["test_x"],
                    "y": v["test_y"],
                },
                label_dict=label_dict,
                batch=batch,
                epoch=epoch,
                verbose=verbose
            )
            hist_df["fold"] = fc
            model.name = "fold-" + str(fc) + model.name
            fc = fc + 1
            cls_report.append(hist_df)
            cv_models.append(model)
        return cv_models, cls_report

    def train_model(self, model: Model, train_data: dict, test_data: dict, label_dict: dict, batch: int, epoch: int,
                    verbose: int):
        """
        :param model:
        :param train_data:
        :param test_data:
        :param label_dict:
        :param batch:
        :param epoch:
        :param verbose:
        :return:
        """
        train_history = []
        model_store = []
        valid_metrics = Metrics()

        train_extra_y = to_categorical(y=train_data["y"], num_classes=2)
        test_extra_y = to_categorical(y=test_data["y"], num_classes=2)
        history = None

        if "extra" in self.problem_type:
            history = model.fit(
                x=train_data["x"],
                y=[train_data["y"], train_extra_y],
                validation_data=(test_data["x"], [test_data["y"], test_extra_y]),
                batch_size=batch,
                epochs=epoch,
                verbose=verbose,
                callbacks=[valid_metrics]
            )
        else:
            history = model.fit(
                x=train_data["x"],
                y=train_data["y"],
                validation_data=(test_data["x"], test_data["y"]),
                batch_size=batch,
                epochs=epoch,
                verbose=verbose,
                callbacks=[valid_metrics]
            )
        model_store.append(model)
        train_history = pd.DataFrame(history.history)
        # mlflow.log_metric("SER", rdict["ser"])
        # mlflow.log_metric("JER", rdict["jer"])

        if "extra" in self.problem_type:
            metrics = history.history["val_output_binary_accuracy"]
        else:
            metrics = history.history["val_binary_accuracy"]
        # mlflow.log_metric("Binaray_Accuracy", metrics[len(metrics) - 1])

        return train_history, model, model_store, metrics

    def eval_model(self, model: Model, eval_data: dict):
        """
        :param model:
        :param eval_data:
        :param label_dict:
        :return:
        """
        gold_y = eval_data["y"]
        pred_y = None
        if "extra" in self.problem_type:
            pred_y, _ = model.predict(x=eval_data["x"])
        else:
            pred_y = model.predict(x=eval_data["x"])
        py = []
        for ys in pred_y:
            temp = []
            for yt in ys:
                if yt[0] > 0.5:
                    temp.append(1)
                else:
                    temp.append(0)
                py.append(temp)
        g, p = [], []
        gold_y = gold_y.flatten()
        py = np.array(py).flatten()
        for i in range(len(gold_y)):
            g.append(gold_y[i])
            p.append(py[i])
        rdict = {}
        print(classification_report(g, p))
        ser, jer = calulate_ser_jer(g, p, 1)
        rdict["ser"] = ser
        rdict["jer"] = jer
        return rdict