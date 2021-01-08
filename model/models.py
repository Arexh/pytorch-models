import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.rs.wide_deep import WideDeep
from model.rs.dnn import DNN
from model.rs.nfm import NFM
from model.rs.lr import LR


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class LRModel(LR):
    def __init__(self,
                 dense_feat_dim,
                 sparse_feat_dim,
                 feature_size,
                 embedding_size=8,
                 init_std=0.0001,
                 seed=1024):
        super(LRModel, self).__init__(dense_feat_dim,
                                      sparse_feat_dim,
                                      feature_size,
                                      embedding_size,
                                      init_std,
                                      seed)


class DNNModel(DNN):
    def __init__(self,
                 dense_feat_dim,
                 sparse_feat_dim,
                 feature_size,
                 embedding_size,
                 seed=1024,
                 dnn_hidden_units=[400, 400, 400],
                 init_std=0.001,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 dnn_dropout=0):
        super(DNNModel, self).__init__(dense_feat_dim,
                                       sparse_feat_dim,
                                       feature_size,
                                       embedding_size,
                                       init_std,
                                       seed,
                                       dnn_hidden_units=dnn_hidden_units,
                                       dnn_activation=dnn_activation,
                                       dnn_use_bn=dnn_use_bn,
                                       dnn_dropout=dnn_dropout)


class WideDeepModel(WideDeep):
    def __init__(self,
                 dense_feat_dim,
                 sparse_feat_dim,
                 feature_size,
                 embedding_size,
                 seed=1024,
                 dnn_hidden_units=[400, 400, 400],
                 init_std=0.001,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 dnn_dropout=0):
        super(WideDeepModel, self).__init__(dense_feat_dim,
                                            sparse_feat_dim,
                                            feature_size,
                                            embedding_size,
                                            init_std,
                                            seed,
                                            dnn_hidden_units=dnn_hidden_units,
                                            dnn_activation=dnn_activation,
                                            dnn_use_bn=dnn_use_bn,
                                            dnn_dropout=dnn_dropout)


class NFMModel(NFM):
    def __init__(self,
                 dense_feat_dim,
                 sparse_feat_dim,
                 feature_size,
                 embedding_size,
                 seed=1024,
                 dnn_hidden_units=[400, 400, 400],
                 init_std=0.001,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 dnn_dropout=0):
        super(NFMModel, self).__init__(dense_feat_dim,
                                  sparse_feat_dim,
                                  feature_size,
                                  embedding_size,
                                  init_std,
                                  seed,
                                  dnn_hidden_units=dnn_hidden_units,
                                  dnn_activation=dnn_activation,
                                  dnn_use_bn=dnn_use_bn,
                                  dnn_dropout=dnn_dropout)
