import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import DNNLayer, PredictionLayer


class WideDeep(BaseModel):
    def __init__(self,
                 dense_feat_dim,
                 sparse_feat_dim,
                 feature_size,
                 embedding_size,
                 init_std,
                 seed,
                 dnn_hidden_units,
                 dnn_activation,
                 dnn_use_bn,
                 dnn_dropout):
        super().__init__()

        torch.manual_seed(seed)
        self.dense_feat_dim = dense_feat_dim
        self.sparse_feat_dim = sparse_feat_dim

        # define laysers
        self.linear = nn.Linear(dense_feat_dim + sparse_feat_dim * embedding_size, 1)
        self.embedding_dict = nn.ModuleDict(
            {'embedding_' + str(i + 1): nn.Embedding(feature_size[i], embedding_size)
             for i in range(sparse_feat_dim)}
        )
        self.dnn = DNNLayer(dense_feat_dim + sparse_feat_dim * embedding_size,
                            dnn_hidden_units,
                            activation=dnn_activation,
                            dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn,
                            init_std=init_std)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out = PredictionLayer()

        # initialization
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        for p in self.linear.parameters():
            nn.init.normal_(p, mean=0, std=init_std)
        for p in self.dnn_linear.parameters():
            nn.init.normal_(p, mean=0, std=init_std)

    def forward(self, X):
        # compute sparse features embedding
        X_category = X[:, self.dense_feat_dim:]
        X_category_embedding = \
            [self.embedding_dict['embedding_' + str(i + 1)](X_category[:, i].long())
             for i in range(self.sparse_feat_dim)]
        X_category_embedding = torch.cat(X_category_embedding, 1)
        X_ = torch.cat((X[:, 0:self.dense_feat_dim], X_category_embedding), 1)

        # compute dnn
        dnn_output = self.dnn(X_.float())
        dnn_output = self.dnn_linear(dnn_output)

        # compute lr
        lr_output = self.linear(X_.float())

        logit = dnn_output + lr_output
        logit = logit.squeeze()

        return torch.sigmoid(logit)
