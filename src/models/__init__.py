from torch import nn
import numpy as np 


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def make_fc_layers(num_layers, in_features, hidden_dim, out_features, activation):
    if activation == "relu":
        ACT = nn.ReLU
    elif activation == "tanh":
        ACT = nn.Tanh
    elif activation == "none":
        ACT = nn.Identity
    else:
        raise NotImplementedError()

    net = []
    for idx in range(num_layers):
        in_dim = in_features if idx == 0 else hidden_dim
        out_dim = out_features if idx == num_layers - 1 else hidden_dim
        net.append(layer_init(nn.Linear(in_dim, out_dim)))
        net.append(ACT())

    return nn.Sequential(*net)
