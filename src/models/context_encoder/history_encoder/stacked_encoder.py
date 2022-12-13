import torch
import torch.nn as nn

from ... import make_fc_layers


class StackedEncoder(nn.Module):
    def __init__(self, args, config):
        super(StackedEncoder, self).__init__()

        self.fc = make_fc_layers(
            config.num_fc_layers,
            (args.obs_dim + args.action_dim)*args.len_history,
            config.hidden_dim,
            config.out_dim,
            config.activation,
        )

    def forward(self, x):
        if "obs_2" not in x:
            h_o, h_a = x["obs"]["history_obs_delta"], x["obs"]["history_act"]
        else:
            h_o, h_a = x["obs_2"]["history_obs_delta"], x["obs_2"]["history_act"]
        x = torch.cat([h_o, h_a], dim=-1)

        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))

        x = self.fc(x)

        return x