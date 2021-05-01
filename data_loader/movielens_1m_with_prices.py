from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils.util import ensure_dir
from base import BaseDataLoader

import pandas as pd
import numpy as np
import torch
import time
import os


class MovieLens1MWithPrices(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 cache_path,
                 rebuild_cache,
                 train):
        print('Processing dataset...')
        start_time = time.time()
        # define column names
        self.sparse_features = ["userId", "gender", "age", "occupation", "zip", "movieId"]
        self.dense_features = ["price", "timestamp"]
        self.feature_names = self.dense_features + self.sparse_features

        # npy file path
        cache_raw_npy_path = os.path.join(cache_path, 'train_raw_cache.npy' if train else 'eval_raw_cache.npy')
        cache_label_npy_path = os.path.join(cache_path, 'train_label_cache.npy' if train else 'eval_label_cache.npy')
        cache_price_npy_path = os.path.join(cache_path, 'train_price_cache.npy' if train else 'eval_price_cache.npy')

        if rebuild_cache \
             or not os.path.isfile(cache_raw_npy_path) \
             or not os.path.isfile(cache_label_npy_path) \
             or not os.path.isfile(cache_price_npy_path):
            print('Building cache...')
            cache_start_time = time.time()
            # build cache
            users_df = pd.read_csv(data_dir + 'users.dat', names=["userId", "gender", "age", "occupation", "zip"], sep='::', engine='python')
            prices_df = pd.read_csv(data_dir + 'prices.csv')
            ratings_df = pd.read_csv(data_dir + 'ratings.dat', names=['userId', 'movieId', 'rating', 'timestamp'], sep="::", engine='python')
            del prices_df['title']
            del prices_df['pageUrl']
            # merge tables
            data = pd.merge(pd.merge(ratings_df, users_df), prices_df)
            # drop rating 3
            data.drop(data[data['rating'] == 3].index, inplace=True)
            # nomalize rating
            data.loc[data['rating'] < 3, 'rating'] = 0
            data.loc[data['rating'] > 3, 'rating'] = 1
            self.prices = data['price'].copy()
            print(self.prices)
            # sparse feature encoder: String -> Integer
            for spare_feature in self.sparse_features:
                encoder = LabelEncoder()
                data[spare_feature] = encoder.fit_transform(data[spare_feature])
            # normalize features
            data[self.dense_features] = MinMaxScaler(
                    feature_range=(0, 1)).fit_transform(data[self.dense_features])
            self.prices = data['price']
            self.labels = data['rating']
            self.feature_size = [len(data[i].iloc[:].unique())
                                 for i in self.sparse_features]
            self.length = len(data)
            self.raw_data = data[self.feature_names].values
            print(self.feature_size)
            print(data.head())
            print(self.labels)
            print(self.prices)
            ensure_dir(cache_path)
            np.save(cache_raw_npy_path, self.raw_data)
            np.save(cache_label_npy_path, self.labels)
            np.save(cache_price_npy_path, self.prices)
            print('Building cache finish, time:', time.time() - cache_start_time, 'seconds')
        else:
            self.raw_data = np.load(cache_raw_npy_path)
            self.labels = np.load(cache_label_npy_path)
            self.prices = np.load(cache_price_npy_path)
            self.length = len(self.raw_data)
        # self.prices[self.labels == 1] = 0.1
        # self.prices[self.labels == 0] = 0.9
        print('Processing finish, time:', time.time() - start_time, 'seconds')

    def __getitem__(self, index):
        return self.raw_data[index], self.labels[index], self.prices[index]

    def __len__(self):
        return self.length
