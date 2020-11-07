from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils.util import ensure_dir
from base import BaseDataLoader

import pandas as pd
import numpy as np
import torch
import time
import os


class MyCriteo(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 sparse_norm,
                 cache_path,
                 rebuild_cache,
                 train):
        print('Processing dataset...')
        start_time = time.time()
        # define column names
        self.dense_features = ['I' + str(i) for i in range(1, 14)]
        self.sparse_features = ['C' + str(i) for i in range(1, 27)]
        self.feature_names = self.dense_features + self.sparse_features

        # npy file path
        cache_raw_npy_path = os.path.join(cache_path, 'train_raw_cache.npy' if train else 'eval_raw_cache.npy')
        cache_label_npy_path = os.path.join(cache_path, 'train_label_cache.npy' if train else 'eval_label_cache.npy')
        cache_feat_size_npy_path = os.path.join(cache_path, 'train_feat_size_cache.npy' if train else 'eval_feat_size_cache.npy')

        if rebuild_cache \
             or not os.path.isfile(cache_raw_npy_path) \
             or not os.path.isfile(cache_label_npy_path) \
             or not os.path.isfile(cache_feat_size_npy_path):
            print('Building cache...')
            cache_start_time = time.time()
            # build cache
            data = pd.read_csv(data_dir, names=['label'] + self.feature_names)
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
            self.feature_size = [len(data.iloc[:, i].unique())
                                 for i in range(len(data.columns))]
            self.length = len(data)
            self.labels = data['label']
            self.raw_data = data[self.feature_names].values
            ensure_dir(cache_path)
            np.save(cache_raw_npy_path, self.raw_data)
            np.save(cache_label_npy_path, self.labels)
            np.save(cache_feat_size_npy_path, self.feature_size)
            print('Building cache finish, time:', time.time() - cache_start_time, 'seconds')
        else:
            self.raw_data = np.load(cache_raw_npy_path)
            self.labels = np.load(cache_label_npy_path)
            self.feature_size = np.load(cache_feat_size_npy_path)
            self.length = len(self.raw_data)
        print('Processing finish, time:', time.time() - start_time, 'seconds')

    def __getitem__(self, index):
        return self.raw_data[index], self.labels[index]

    def __len__(self):
        return self.length
