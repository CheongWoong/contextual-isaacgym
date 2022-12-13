import numpy as np

import torch 
import torch.nn as nn

from .. import make_fc_layers


class OSI_NLL(nn.Module):
    def __init__(self, args, config, encoder_config, decoder_config):
        super(OSI_NLL, self).__init__()

        self.return_system_prediction = True if getattr(config, "out_dim", 0) < 0 else False

        self.fc_h = make_fc_layers(decoder_config.num_decoder_layers, encoder_config.out_dim, encoder_config.out_dim, encoder_config.out_dim, decoder_config.activation)
        self.fc_mu = make_fc_layers(1, encoder_config.out_dim, encoder_config.out_dim, args.context_dim, "none")
        self.fc_sigma = make_fc_layers(1, encoder_config.out_dim, encoder_config.out_dim, args.context_dim, "none")

    def forward(self, context, x):
        h = self.fc_h(context)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h))

        sys_id_loss = mdn_loss_fn(sigma, mu, x["obs"]["context"])

        loss_dict = {
            "total_loss": sys_id_loss,
            "sys_id_loss": sys_id_loss,
        }
        context_info = {
            "loss": loss_dict,
            "system_prediction_mu": mu,
            "system_prediction_sigma": sigma,
        }

        if self.return_system_prediction:
            return torch.cat([mu, sigma], dim=-1).detach(), context_info
            
        return context.detach(), context_info

oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)
def gaussian_distribution(y, mu, sigma):
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_fn(sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma)
    eps = 1e-07
    result = -torch.log(result + eps)
    return torch.mean(result)