import argparse
import json
import os
import random

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf

from citation_bio_trainer.etl.ETL import ETL
from citation_bio_trainer.model.BIOLSTM import BIOLSTM


def load_json(path):
    d = None
    with open(path) as f:
        d = json.load(f)
    f.close()
    d = dict(zip(d.values(), d.keys()))
    return d


def get_name_from_dict(d: dict):
    name = ""
    for k, v in d.items():
        k = str(k)[:min(3, len(str(k)) - 1)]
        if type(v) == str:
            v = v.replace("/", "#")
            v = v[:min(len(v) - 1, 3)]
        elif type(v) == int:
            v = str(v)
        elif type(v) == float:
            v = str(v)
        else:
            continue
        name += k + "-" + v + "_"
    return name[0:len(name) - 1]


def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(123)
    tf.random.set_seed(123)
    np.random.seed(123)
    random.seed(123)


def main():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--output-dir",
        dest="output",
        default="./nlp/exps/output",
        help="output dir"
    )

    parser.add_argument(
        "--data-config",
        dest="data_config_path",
        default="./config/data_config.json",
        help="data generation config"
    )
    parser.add_argument(
        "--feature-config",
        dest="feature_config_path",
        default="./config/feature_config.json",
        help="pre-processing/feature engg config"
    )
    parser.add_argument(
        "--model-config",
        dest="model_config_path",
        default="./config/model_config.json",
        help="training model config"
    )

    parser.add_argument(
        "--mlflow-server-url",
        dest="mlflow_server",
        default="https://mlflow.caps.dev.dp.elsevier.systems",
        help="mlflow server path"
    )

    parser.add_argument(
        "--store-at-mlflow-server",
        dest="store_at_server",
        default=True,
        help="to store data and artifacts or not"
    )

    args = parser.parse_args()

    # init: name, output-dir, timestamp

    exp_name = "BI-LSTM"

    # data config loaded
    data_config = None
    with open(args.data_config_path) as jf:
        data_config = json.load(jf)
    jf.close()
    print("### Data Config given:\n")
    for k, v in data_config.items():
        print(f"--{k} {v}")
    print("\n")

    # downloading data if not exits
    input_files = []
    max_len = -1
    args.output = os.path.join(args.output, data_config["local_dir"])
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        os.system("aws s3 cp s3://" + data_config["s3_bucket"] + "/" + data_config[
            "s3_dir"] + "/ " + args.output + " --recursive")
    # loading path and max_len
    for f_name in os.listdir(args.output):
        if f_name.startswith(data_config["training_file_prefix"]):
            df = pd.read_csv(os.path.join(args.output, f_name))
            if max_len < len(df):
                max_len = len(df)
            input_files.append(os.path.join(args.output, f_name))

    # feature config loaded
    feature_config = None
    with open(args.feature_config_path) as jf:
        feature_config = json.load(jf)
    jf.close()

    # model config loaded
    model_config = None
    with open(args.model_config_path) as jf:
        model_config = json.load(jf)
    jf.close()

    etl = ETL(config=feature_config)

    # train/test split recognition from model_config
    train_count = model_config["train_per"] * len(input_files)
    train_count = min(len(input_files) - 1, train_count)

    feature_config["output_dir"] = args.output
    feature_config["train_count"] = train_count
    feature_config["test_count"] = (len(input_files) - train_count)

    feature_suffix = get_name_from_dict(feature_config)

    etl = ETL(config=feature_config)
    data_dict, label_dict = etl.data_and_xydict(df_file_paths=input_files, max_len=max_len,
                                                feature_suffix=feature_suffix)
    lab_dict = dict(zip(label_dict.values(), label_dict.keys()))

    train_data = data_dict["train"]
    test_data = data_dict["test"]

    model_config["input_dim"] = etl.vec_dim
    model_config["output_dim"] = len(lab_dict)

    # logg in mlflow

    biolstm = BIOLSTM(config=model_config)
    model = biolstm.get_model()
    history_df, model, model_store, metrics = biolstm.train_model(
        model=model,
        train_data={
            "x": train_data["x"],
            "y": train_data["y"]
        },
        test_data={
            "x": test_data["x"],
            "y": test_data["y"],
        },
        label_dict=lab_dict,
        batch=model_config["batch"],
        epoch=model_config["epoch"],
        verbose=1
    )

    path = os.path.join(args.output, "training.history.csv")
    history_df.to_csv(path)

    temp = {
        "epoch": [i for i in range(len(metrics[0]))],
        "loss": metrics[0],
        "accuracy": metrics[1],
        "ser": metrics[2],
        "jer": metrics[3]
    }

    if args.store_at_server:
        try:
            mlflow.set_tracking_uri(args.mlflow_server)
        except:
            mlflow.set_tracking_uri(args.output)
        mlflow.set_experiment(exp_name)
        for i in range(len(metrics[0])):
            mlflow.start_run()
            d = dict(load_json(os.path.join(args.output, "data-gen-config.json")))
            for k, v in d.items():
                if "/" not in v and "*" not in v:
                    mlflow.log_param("dsata-gen-" + str(v), k)
            print("\n")
            mlflow.set_tag("features", feature_suffix)
            for k, v in model_config.items():
                if type(v) == str or type(v) == float or type(v) == int:
                    if "epoch" not in k:
                        mlflow.log_param(k, v)
            mlflow.log_param("train_count", train_count)
            mlflow.log_param("test_count", (len(input_files) - train_count))
            mlflow.log_param("epoch", i + 1)
            mlflow.log_metric("loss", metrics[0][i])
            mlflow.log_metric("accuracy", metrics[1][i])
            mlflow.log_metric("ser", metrics[2][i])
            mlflow.log_metric("jer", metrics[3][i])
            mlflow.end_run()

    tdf = pd.DataFrame(temp)
    fig = tdf.plot(x="epoch", y=["loss", "accuracy", "ser", "jer"]).get_figure()
    path = os.path.join(args.output, "training.history.png")
    fig.savefig(path)

    if args.store_at_server:
        mlflow.log_artifact(path)

    with open(os.path.join(args.output, "model-hyper-param-config.json"), 'w') as jf:
        json.dump(model_config, jf)
    jf.close()

    if args.store_at_server:
        mlflow.log_artifact(os.path.join(args.output, "model-hyper-param-config.json"))

    for ik in range(len(model_store)):
        path = os.path.join(args.output, str(model.name) + "-num-" + str(ik) + ".h5")
        model_store[ik].save(path)
        if args.store_at_server:
            mlflow.keras.log_model(model_store[ik], model_store[ik].name)


print("### Done !!!")

if __name__ == '__main__':
    main()
