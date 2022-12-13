import torch 
import torch.nn as nn

from .history_encoder import RecurrentEncoder, StackedEncoder, PEARLEncoder

from .osi_gt import OSI_GT
from .osi import OSI
from .osi_nll_loss import OSI_NLL
from .pearl import PEARL
from .cadm import CaDM


class ContextEncoder(nn.Module):
    def __init__(self, args):
        super(ContextEncoder, self).__init__()
        
        self.args = args
        self.config = config = getattr(args.context_encoder, args.context_encoder_type)

        self.use_context = False if args.context_encoder_type == "none" else True

        self.history_encoder, self.decoder = None, None
        if self.use_context:
            if hasattr(config, "history_encoder_type"):
                self.history_encoder_config = getattr(args.history_encoder, config.history_encoder_type)
                self.history_encoder = self.get_history_encoder(config.history_encoder_type)
            if hasattr(config, "decoder_type"):
                self.decoder_config = getattr(args.decoder, config.decoder_type)
                self.decoder = self.get_decoder(config.decoder_type)
        else:
            self.temp = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x):
        context, context_info = None, None
        if self.history_encoder is not None:
            context = self.history_encoder(x)
        if self.decoder is not None:
            context, context_info = self.decoder(context, x)

        return context, context_info

    def get_history_encoder(self, type):
        encoder_map = {
            "stacked": StackedEncoder,
            "rnn": RecurrentEncoder,
            "pearl": PEARLEncoder
        }
        return encoder_map[type](self.args, self.history_encoder_config)

    def get_decoder(self, task):
        decoder_map = {
            "osi_gt": OSI_GT,
            "osi": OSI,
            "osi_nll": OSI_NLL,
            "cadm": CaDM,
            "pearl": PEARL
        }

        return decoder_map[task](self.args, self.config, self.history_encoder_config, self.decoder_config)