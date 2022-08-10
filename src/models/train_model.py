import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import lightgbm as lgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, \
    classification_report, log_loss


def read_params(config):
    with open(config) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def accuracy_measures(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1score = f1_score(y_test, predictions)
    target_names = ['0', '1']
    print("Classification report")
    print("---------------------", "\n")
    print(classification_report(y_test, predictions, target_names=target_names), "\n")
    print("Confusion Matrix")
    print("---------------------", "\n")
    print(confusion_matrix(y_test, predictions), "\n")

    print("Accuracy Measures")
    print("---------------------", "\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy, precision, recall, f1score


def convert_obj_to_int(self):
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0, len(object_list_columns)):
        if object_list_dtypes[index] == object:
            self[object_list_columns[index] + new_col_suffix] = self[object_list_columns[index]].map(lambda x: hash(x))
            self.drop([object_list_columns[index]], inplace=True, axis=1)
    return self


def train_and_evaluate(config):
    config = read_params(config)
    train_data_path = config['processed_data_config']['train_data_csv']
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    params = config['lightgbm']

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')

    train.drop('hour', axis=1, inplace=True)
    train.drop('id', axis=1, inplace=True)
    test.drop('hour', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    train_x = train[[i for i in train.columns if i != target]]
    train_y = train['click']

    test_x = test[[i for i in test.columns if i != target]]
    test_y = test['click']

    train_x = convert_obj_to_int(train_x)
    test_x = convert_obj_to_int(test_x)

    msk = np.random.rand(len(train_x)) < 0.8

    lgb_train = lgb.Dataset(train_x[msk], label=train_y[msk],)
    lgb_eval = lgb.Dataset(train_x[~msk], label=train_y[~msk],reference=lgb_train)
    print('Start training...')

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    mlflow.lightgbm.autolog()
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=4000,
                        valid_sets=[lgb_train, lgb_eval],
                        early_stopping_rounds=500)

        y_prob = gbm.predict(test_x)
        print(test_x.columns)
        loss = log_loss(test_y, y_prob)
        y_pred = np.round(y_prob)
        accuracy, precision, recall, f1score = accuracy_measures(test_y, y_pred)

        mlflow.log_param("task", params['task'])
        mlflow.log_param("boosting_type", params['boosting_type'])
        mlflow.log_param("objective", params['objective'])
        mlflow.log_param("metric", params['metric'])
        mlflow.log_param("num_leaves", params['num_leaves'])
        mlflow.log_param("learning_rate", params['learning_rate'])
        mlflow.log_param("feature_fraction", params['feature_fraction'])
        mlflow.log_param("bagging_fraction", params['bagging_fraction'])
        mlflow.log_param("bagging_freq", params['bagging_freq'])
        mlflow.log_param("verbose", params['verbose'])

        mlflow.log_metric("log_loss", loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        signature = infer_signature(train_x, gbm.predict(train_x))

        if tracking_url_type_store != "file":
            mlflow.lightgbm.log_model(
                gbm,
                "model",
                signature=signature,
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.lightgbm.log_model(gbm, "model", signature=signature)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config=parsed_args.config)
