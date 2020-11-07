from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils.util import ensure_dir
from base import BaseDataLoader
from tqdm import tqdm

import pandas as pd
import numpy as np
import os


class MyCriteo(BaseDataLoader):
    def __init__(self,
                 data_dir,
                 sparse_norm,
                 cache_path,
                 rebuild_cache,
                 train):
        # define column names
        self.dense_features = ['I' + str(i) for i in range(1, 14)]
        self.sparse_features = ['C' + str(i) for i in range(1, 27)]
        self.feature_names = self.dense_features + self.sparse_features
        self.raw_data = []

        # npy file path
        cache_npy_path = os.path.join(cache_path, 'train_cache.npz' if train else 'eval_cache.npz')

        if rebuild_cache or not os.path.isfile(cache_npy_path):
            # build cache
            data = pd.read_csv(data_dir, names=['label'] + self.feature_names)
            self.feature_size = [len(data.iloc[:, i].unique())
                                 for i in range(len(data.columns))]
            # fill Nan data
            data[self.sparse_features] = data[self.sparse_features].fillna('-1', )
            data[self.dense_features] = data[self.dense_features].fillna(0,)
            # sparse feature encoder: String -> Integer
            for spare_feature in self.sparse_features:
                encoder = LabelEncoder()
                data[spare_feature] = encoder.fit_transform(data[spare_feature])
            # normalize features
            if sparse_norm:
                data[self.feature_names] = MinMaxScaler(
                    feature_range=(0, 1)).fit_transform(data[self.feature_names])
            else:
                data[self.dense_features] = MinMaxScaler(
                    feature_range=(0, 1)).fit_transform(data[self.dense_features])
            feature_data = data[self.feature_names]
            self.length = len(data)
            self.labels = data['label']
            for index in tqdm(range(self.length)):
                self.raw_data.append(feature_data.loc[index].values)
            ensure_dir(cache_path)
            np.savez(cache_npy_path,
                     raw_data=self.raw_data,
                     labels=self.labels)
        else:
            cache_data = np.load(cache_npy_path)
            self.raw_data = cache_data['raw_data']
            self.labels = cache_data['labels']
            self.length = len(self.raw_data)

    def __getitem__(self, index):
        return self.raw_data[index], self.labels[index]

    def __len__(self):
        return self.length
