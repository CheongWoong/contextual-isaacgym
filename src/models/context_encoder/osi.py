import torch 
import torch.nn as nn

from .. import make_fc_layers


class OSI(nn.Module):
    def __init__(self, args, config, encoder_config, decoder_config):
        super(OSI, self).__init__()

        self.return_system_prediction = True if getattr(config, "out_dim", 0) < 0 else False

        self.fc = make_fc_layers(decoder_config.num_decoder_layers, encoder_config.out_dim, encoder_config.out_dim, encoder_config.out_dim, decoder_config.activation)
        self.fc_final = make_fc_layers(1, encoder_config.out_dim, encoder_config.out_dim, args.context_dim, "none")

        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, context, x):
        system_prediction = self.fc_final(self.fc(context))

        sys_id_loss = torch.sqrt(self.mse_loss(system_prediction, x["obs"]["context"]))

        loss_dict = {
            "total_loss": sys_id_loss,
            "sys_id_loss": sys_id_loss,
        }
        context_info = {
            "loss": loss_dict,
            "system_prediction": system_prediction,
        }

        if self.return_system_prediction:
            return system_prediction.detach(), context_info

        return context.detach(), context_info