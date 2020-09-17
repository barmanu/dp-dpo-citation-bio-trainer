import os

import h5py
import numpy as np
import pandas as pd
import spacy
import tensorflow_hub as hub
from tqdm import tqdm


class ETL:
    def __init__(self, config):
        """
        :param config:
        """
        self.__dict__ = config
        self.nlp = spacy.load("en_core_web_sm")
        self.tfhub = hub.load(self.google_vec_url)
        self.vec_dim = 512
        print("### TFhub Loaded ...")

    @staticmethod
    def write_nparray(path, arr):
        """
        :param path:
        :param arr:
        :return:
        """
        with h5py.File(path, "w") as hf:
            hf.create_dataset("d", data=arr)
        hf.close()

    @staticmethod
    def read_nparray(path):
        """
        :param path:
        :return:
        """
        arr = None
        with h5py.File(path, "r") as hf:
            arr = hf["d"][:]
        hf.close()
        return arr

    def data_and_xydict(
        self,
        train_df_file_paths: list,
        test_df_file_paths,
        max_len: int,
        feature_suffix: str,
    ):
        """
        :param df_file_paths:
        :param max_len:
        :return:
        """

        label_dict = {"I-CIT": 0, "B-CIT": 1}

        train_x_path = os.path.join(
            self.output_dir, "train_x_" + feature_suffix + ".h5"
        )
        train_y_path = os.path.join(
            self.output_dir, "train_y_" + feature_suffix + ".h5"
        )
        test_x_path = os.path.join(
            self.output_dir, "test_x_" + feature_suffix + ".h5"
        )
        test_y_path = os.path.join(
            self.output_dir, "test_y_" + feature_suffix + ".h5"
        )

        if (
            os.path.exists(train_x_path)
            and os.path.exists(train_y_path)
            and os.path.exists(test_x_path)
            and os.path.exists(test_y_path)
        ):

            train_x = ETL.read_nparray(train_x_path)
            train_y = ETL.read_nparray(train_y_path)
            test_x = ETL.read_nparray(test_x_path)
            test_y = ETL.read_nparray(test_y_path)

            return (
                {
                    "train": {
                        "x": train_x,
                        "y": train_y,
                    },
                    "test": {
                        "x": test_x,
                        "y": test_y,
                    },
                },
                label_dict,
            )
        else:

            print("\n ### Data Processing Started ...")

            all_labs = ["I-CIT", "B-CIT"]
            all_labs = list(all_labs)
            all_labs = {x: i for i, x in enumerate(all_labs)}
            dummy_x = [0.0] * self.vec_dim
            dummy_x = np.array(dummy_x)
            dummy_y = [0, 0]
            dummy_y = np.array(dummy_y)

            def change_nl(x):
                if "\n" in x:
                    return "MWLN"
                else:
                    return x

            train_x = []
            train_y = []
            count = 0

            for file_path in tqdm(train_df_file_paths[:30]):
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, index_col=0)
                    df.fillna("\n", axis=1, inplace=True)
                    df["x"] = df.apply(lambda t: change_nl(t.x), axis=1)
                    cit_sec = df.x.tolist()
                    cit_lab = df.y.tolist()
                    cit_lab = [
                        [0, 1] if x.startswith("B") else [1, 0]
                        for x in cit_lab  # label creation
                    ]
                    tfhub_x = [np.array(x) for x in self.tfhub(cit_sec)]
                    x = []
                    padd_len = max_len - len(tfhub_x)
                    for i in range(len(tfhub_x)):
                        x.append(tfhub_x[i])
                    for i in range(padd_len):
                        x.append(dummy_x)
                        cit_lab.append(dummy_y)

                    train_x.append(x)
                    train_y.append(cit_lab)
                    count = count + 1
            print("### Writing Training Data  ...")
            train_x = np.array(train_x)
            print("### X-Shape:", train_x.shape)
            train_y = np.array(train_y)
            print("### Y-Shape:", train_y.shape)
            ETL.write_nparray(train_x_path, train_x)
            ETL.write_nparray(train_y_path, train_y)

            test_x = []
            test_y = []
            count = 0

            for file_path in tqdm(test_df_file_paths[:30]):
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path, index_col=0)
                    df.fillna("\n", axis=1, inplace=True)
                    df["x"] = df.apply(lambda t: change_nl(t.x), axis=1)
                    cit_sec = df.x.tolist()
                    cit_lab = df.y.tolist()
                    cit_lab = [
                        [0, 1] if x.startswith("B") else [1, 0]
                        for x in cit_lab  # label creation
                    ]
                    tfhub_x = [np.array(x) for x in self.tfhub(cit_sec)]
                    x = []
                    padd_len = max_len - len(tfhub_x)
                    for i in range(len(tfhub_x)):
                        x.append(tfhub_x[i])
                    for i in range(padd_len):
                        x.append(dummy_x)
                        cit_lab.append(dummy_y)

                    test_x.append(x)
                    test_y.append(cit_lab)
                    count = count + 1

            print("### Writing Test Data  ...")

            test_x = np.array(test_x)
            print("### X-Shape:", test_x.shape)
            test_y = np.array(test_y)
            print("### Y-Shape:", test_y.shape)
            ETL.write_nparray(test_x_path, test_x)
            ETL.write_nparray(test_y_path, test_y)

            return (
                {
                    "train": {"x": train_x, "y": train_y},
                    "test": {"x": test_x, "y": test_y},
                    "labels": all_labs,
                },
                all_labs,
            )
