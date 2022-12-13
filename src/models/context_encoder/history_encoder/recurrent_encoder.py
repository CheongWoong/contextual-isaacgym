import torch
import torch.nn as nn

from ... import make_fc_layers


class RecurrentEncoder(nn.Module):
    def __init__(self, args, config):
        super(RecurrentEncoder, self).__init__()
        
        self.num_rnn_layers = config.num_rnn_layers

        self.fc = make_fc_layers(
            config.num_fc_layers,
            args.obs_dim + args.action_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.activation,
        )

        self.rnn = nn.LSTM(config.hidden_dim, config.hidden_dim, num_layers=config.num_rnn_layers, batch_first=True)

        self.fc_final = make_fc_layers(
            1,
            config.hidden_dim,
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

        x = self.fc(x)

        batch_size = x.shape[0]
        hidden_dim = x.shape[-1]
        h0 = torch.zeros(self.num_rnn_layers, batch_size, hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_rnn_layers, batch_size, hidden_dim, device=x.device)

        rnn_out, _ = self.rnn(x, (h0, c0))
        x = self.fc_final(rnn_out[:,-1,:])

        return x