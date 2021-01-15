import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from model.layers import DNNLayer, PredictionLayer


# Ref: https://github.com/shenweichen/DeepCTR-Torch/blob/e7d52151ed3c8beafeda941051aecc6294a4a20d/deepctr_torch/layers/interaction.py#L464
class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024,):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
                torch.empty(in_features, in_features))) for i in range(self.layer_num)])
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == 'matrix':
                # W * xi  (bs, in_features, 1)
                dot_ = torch.matmul(self.kernels[i], x_l)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                print("parameterization should be 'vector' or 'matrix'")
                pass
            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l

# Ref: https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/dcn.py


class DCN(BaseModel):
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
                 dnn_dropout,
                 cross_num=2,
                 cross_parameterization='vector'):
        super().__init__()
        torch.manual_seed(seed)
        self.dense_feat_dim = dense_feat_dim
        self.sparse_feat_dim = sparse_feat_dim

        self.embedding_dict = nn.ModuleDict(
            {'embedding_' + str(i + 1): nn.Embedding(feature_size[i], embedding_size)
             for i in range(sparse_feat_dim)}
        )
        self.dnn = DNNLayer(dense_feat_dim + embedding_size * sparse_feat_dim,
                            dnn_hidden_units,
                            activation=dnn_activation,
                            dropout_rate=dnn_dropout,
                            use_bn=False,
                            init_std=init_std)
        self.linear = nn.Linear(dense_feat_dim + embedding_size *
                                sparse_feat_dim + dnn_hidden_units[-1], 1, bias=False)
        self.crossnet = CrossNet(in_features=dense_feat_dim + embedding_size * sparse_feat_dim,
                                 layer_num=cross_num, parameterization=cross_parameterization)

    def forward(self, X):
        # compute sparse features embedding
        X_category = X[:, self.dense_feat_dim:]
        X_continuous = X[:, 0:self.dense_feat_dim]
        X_category_embedding = \
            [self.embedding_dict['embedding_' + str(i + 1)](X_category[:, i].long())
             for i in range(self.sparse_feat_dim)]
        X_category_embedding = torch.cat(X_category_embedding, 1)
        X_ = torch.cat((X[:, 0:self.dense_feat_dim], X_category_embedding), 1)

        # compute deep and cross
        deep_out = self.dnn(X_.float())
        cross_out = self.crossnet(X_.float())
        stack_out = torch.cat((cross_out, deep_out), dim=-1)
        logit = self.linear(stack_out)
        logit = logit.squeeze()

        return torch.sigmoid(logit)
