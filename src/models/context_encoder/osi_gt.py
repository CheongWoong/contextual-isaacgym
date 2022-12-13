import torch 
import torch.nn as nn

from .. import make_fc_layers


class OSI_GT(nn.Module):
    def __init__(self, args, config, encoder_config, decoder_config):
        super(OSI_GT, self).__init__()

    def forward(self, context, x):
        return x["obs"]["context"], None