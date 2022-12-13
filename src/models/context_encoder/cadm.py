import torch 
import torch.nn as nn

from .. import make_fc_layers


class CaDM(nn.Module):
    def __init__(self, args, config, encoder_config, decoder_config):
        super(CaDM, self).__init__()

        self.fc_forward = make_fc_layers(decoder_config.num_decoder_layers, args.obs_dim + args.action_dim + encoder_config.out_dim, encoder_config.hidden_dim, args.obs_dim, decoder_config.activation)
        self.fc_backward = make_fc_layers(decoder_config.num_decoder_layers, args.obs_dim + args.action_dim + encoder_config.out_dim, encoder_config.hidden_dim, args.obs_dim, decoder_config.activation)
        self.fc_forward_final = make_fc_layers(1, args.obs_dim, args.obs_dim, args.obs_dim, "none")
        self.fc_backward_final = make_fc_layers(1, args.obs_dim, args.obs_dim, args.obs_dim, "none")

        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, context, x):
        if x.get("training", False):
            if "obs_2" in x:
                obs, action, next_obs = x["obs_2"]["obs"], x["action_2"], x["next_obs_2"]
            else:
                obs, action, next_obs = x["obs"]["obs"], x["action"], x["next_obs"]

            forward_input = torch.cat([obs, action, context], dim=-1)
            forward_prediction = self.fc_forward_final(self.fc_forward(forward_input))

            backward_input = torch.cat([next_obs, action, context], dim=-1)
            backward_prediction = self.fc_backward_final(self.fc_backward(backward_input))

            forward_loss = torch.sqrt(self.mse_loss(forward_prediction, next_obs-obs))
            backward_loss = torch.sqrt(self.mse_loss(backward_prediction, obs-next_obs))
            total_loss = forward_loss + backward_loss

            loss_dict = {
                "total_loss": total_loss,
                "forward_loss": forward_loss,
                "backward_loss": backward_loss,
            }
            context_info = {
                "loss": loss_dict,
                "forward_prediction": forward_prediction,
                "backward_prediction": backward_prediction,
            }
        else:
            context_info = None
        
        return context.detach(), context_info