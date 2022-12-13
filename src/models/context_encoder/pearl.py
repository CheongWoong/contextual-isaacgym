import torch 
import torch.nn as nn
import torch.nn.functional as F

from .. import make_fc_layers


class PEARL(nn.Module):
    def __init__(self, args, config, encoder_config, decoder_config):
        super(PEARL, self).__init__()

        self.fc_h = make_fc_layers(decoder_config.num_decoder_layers, encoder_config.out_dim, encoder_config.out_dim, encoder_config.out_dim, decoder_config.activation)
        self.fc_mu = make_fc_layers(1, encoder_config.out_dim, encoder_config.out_dim, encoder_config.out_dim, "none")
        self.fc_sigma = make_fc_layers(1, encoder_config.out_dim, encoder_config.out_dim, encoder_config.out_dim, "none")

    def forward(self, context, x):
        h = self.fc_h(context)
        mus = self.fc_mu(h)
        sigmas = F.softplus(self.fc_sigma(h))

        if len(context.shape) > 2:
            context, mu, sigma = self.infer_posterior(mus, sigmas)
        else:
            context, mu, sigma = self.infer_posterior(mus, sigmas, product=False)

        kld_loss = self.compute_kl_div(mu, sigma)

        loss_dict = {
            "total_loss": kld_loss,
            "kld_loss": kld_loss,
        }
        context_info = {
            "loss": loss_dict,
            "context_mu": mu,
            "context_sigma": sigma,
        }

        return context, context_info

    def infer_posterior(self, mus, sigmas, product=True):
        if product:
            mu, sigma = self.product_of_gaussians(mus, sigmas)
        else:
            mu, sigma = mus, sigmas
        z = self.sample_z(mu, sigma)
        return z, mu, sigma
        
    def product_of_gaussians(self, mus, sigmas_squared):
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
        return mu, sigma_squared

    def sample_z(self, mu, sigma):
        posteriors = torch.distributions.Normal(mu, torch.sqrt(sigma))
        z = posteriors.rsample()
        return z

    def compute_kl_div(self, mu, sigma):
        prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        posteriors = torch.distributions.Normal(mu, torch.sqrt(sigma))
        kl_divs = torch.distributions.kl.kl_divergence(prior, posteriors)
        kl_div_sum = torch.sum(kl_divs)
        return kl_div_sum