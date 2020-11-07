import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import *


class WideDeep(BaseModel):
    def __init__(self,
                 continuous_dim,
                 category_dim,
                 vocabulary_size,
                 dnn_hidden_units=(256, 128),
                 init_std=0.0001,
                 seed=1024,
                 embedding_size=8,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 task='binary',
                 device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        self.continuous_dim = continuous_dim
        self.category_dim = category_dim
        self.linear_model = nn.Linear(continuous_dim + category_dim * embedding_size, 1)

        # self.dnn = DNN(continuous_dim + category_dim * 8, dnn_hidden_units,
        #                activation=dnn_activation,
        #                dropout_rate=dnn_dropout,
        #                use_bn=dnn_use_bn,
        #                init_std=init_std,
        #                device=device)
        # self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.embedding_dict = nn.ModuleDict(
            {'embedding_' + str(i + 1): nn.Embedding(vocabulary_size[i], embedding_size)
             for i in range(category_dim)}
        )
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.out = PredictionLayer()
        self.to(device)

    def forward(self, X):
        X_category = X[:, self.continuous_dim:]
        X_category_embedding = \
            [self.embedding_dict['embedding_' + str(i + 1)](X_category[:, i].long())
             for i in range(self.category_dim)]
        X_category_embedding = torch.cat(X_category_embedding, 1)
        X_ = torch.cat((X[:, 0:self.continuous_dim], X_category_embedding), 1)
        logit = self.linear_model(X_.float())
        # dnn_output = self.dnn(dnn_input)
        # dnn_logit = self.dnn_linear(dnn_output)
        # logit += dnn_logit

        # y_pred = self.out(logit)
        # y_pred = y_pred.squeeze()
        logit = logit.squeeze()
        return torch.sigmoid(logit)
