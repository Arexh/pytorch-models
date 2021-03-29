import torch.nn as nn
import torch

from model.activation import activation_layer


class DNNLayer(nn.Module):
    def __init__(self,
                 inputs_dim,
                 hidden_units,
                 activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 init_std=0.0001,
                 seed=1024):
        super(DNNLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    def __init__(self,
                 task='binary',
                 use_bias=True,
                 **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary, multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        return output


# from https://github.com/shenweichen/DeepCTR-Torch/blob/d18ea26c09ccc16541dd7985d6ba0a8895bc288d/deepctr_torch/layers/interaction.py#L12
class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, X):
        square_of_sum = torch.pow(torch.sum(X, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(X * X, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


# from https://github.com/shenweichen/DeepCTR-Torch/blob/d18ea26c09ccc16541dd7985d6ba0a8895bc288d/deepctr_torch/layers/interaction.py#L462
class CrossNetMix(nn.Module):
    def __init__(self,
                 in_features,
                 low_rank=32,
                 num_experts=4,
                 layer_num=2):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # V: (in_features, low_rank)
        self.V_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, in_features, low_rank))) for i in range(self.layer_num)])
        # C: (low_rank, low_rank)
        self.C_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(num_experts, low_rank, low_rank))) for i in range(self.layer_num)])
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(
            torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
