import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import DNNLayer, PredictionLayer


class BiInteractionPooling(nn.Module):

    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term


class NFM(BaseModel):
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

        self.embedding_dict = nn.ModuleDict(
            {'embedding_' + str(i + 1): nn.Embedding(feature_size[i], embedding_size)
             for i in range(sparse_feat_dim)}
        )
        self.dnn = DNNLayer(dense_feat_dim + embedding_size,
                            dnn_hidden_units,
                            activation=dnn_activation,
                            dropout_rate=dnn_dropout,
                            use_bn=False,
                            init_std=init_std)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.bi_pooling = BiInteractionPooling()
            
        # initialization
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        for p in self.dnn_linear.parameters():
            nn.init.normal_(p, mean=0, std=init_std)

    def forward(self, X):
        # compute sparse features embedding
        X_category = X[:, self.dense_feat_dim:]
        X_continuous = X[:, 0:self.dense_feat_dim]
        X_category_embedding = \
            [self.embedding_dict['embedding_' + str(i + 1)](X_category[:, i].long())
             for i in range(self.sparse_feat_dim)]
        X_category_embedding =  torch.stack(X_category_embedding).permute([1, 0, 2])
        bi_out = self.bi_pooling(X_category_embedding).squeeze()
        X_ = torch.cat((X_continuous, bi_out), 1)

        # compute dnn
        logit = self.dnn(X_.float())
        logit = self.dnn_linear(logit)
        logit = torch.sigmoid(logit)
        logit = logit.squeeze()

        return logit
