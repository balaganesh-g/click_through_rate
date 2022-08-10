import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from load_data import read_param

types_train = {
    'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(int),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str),
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(int),
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21': np.dtype(int)
}


def split_data(df, test_size, random_state, train_path, test_path):
    """
    This function helps to split the data into train test dataset
    :param df:
    :param test_size:
    :param random_state:
    :param train_path:
    :param test_path:
    :return:
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


def split_and_save_dataset(config):
    """
    This fuction reads the raw data and splits the data into train and test dataset
    and save it into processed data.
    :param config:
    :return:
    """
    config = read_param(config)
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    train_data_path = config["processed_data_config"]["train_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]
    raw_df = pd.read_csv(raw_data_path, dtype=types_train)
    split_data(raw_df, split_ratio, random_state, train_data_path, test_data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    split_and_save_dataset(parsed_args.config)
