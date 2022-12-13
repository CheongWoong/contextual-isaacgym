

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .. import layer_init, make_fc_layers
from ..context_encoder import ContextEncoder


class Critic(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()

        self.fc = make_fc_layers(config.num_layers, in_dim, config.hidden_dim, config.hidden_dim, config.activation)
        self.critic = layer_init(nn.Linear(config.hidden_dim, 1), std=1.0)

    def get_value(self, x, context=None):
        x = x['obs']
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        x = self.fc(x)
        x = self.critic(x)
        return x


class Actor(nn.Module):
    def __init__(self, in_dim, action_dim, config):
        super().__init__()

        self.fc = make_fc_layers(config.num_layers, in_dim, config.hidden_dim, config.hidden_dim, config.activation)
        self.actor_mean = layer_init(nn.Linear(config.hidden_dim, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, x, action=None, context=None):
        x = x['obs']
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        x = self.fc(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class Agent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.context_encoder = ContextEncoder(config)
        context_dim = getattr(self.context_encoder.config, "out_dim", None)
        context_dim = self.context_encoder.history_encoder_config.out_dim if context_dim is None else context_dim
        if context_dim < 0:
            context_dim *= -config.context_dim
        self.critic = Critic(config.obs_dim + context_dim, config.critic)
        self.actor = Actor(config.obs_dim + context_dim, config.action_dim, config.actor)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate, eps=1e-5)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate, eps=1e-5)
        self.context_optimizer = optim.Adam(self.context_encoder.parameters(), lr=config.learning_rate, eps=1e-5)
        self.optimizers = [self.critic_optimizer, self.actor_optimizer, self.context_optimizer]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_context(self, x):
        return self.context_encoder(x)

    def get_value(self, x, context=None):
        return self.critic.get_value(x, context)

    def get_action(self, x, action=None, context=None):
        return self.actor.get_action(x, action, context)

    def get_action_and_value(self, x, action=None, context=None):
        action, log_prob, entropy = self.get_action(x, action, context)
        value = self.get_value(x, context)
        
        return action, log_prob, entropy, value

    def learn(self, rb):
        self.train()
        args = self.config

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            for rollout_data in rb.get(args.minibatch_size):
                x = {"training": True}
                x["obs"], x["action"], x["next_obs"] = rollout_data.observations, rollout_data.actions, rollout_data.next_observations
                if hasattr(rollout_data, "observations_2"):
                    x["obs_2"], x["action_2"], x["next_obs_2"] = rollout_data.observations_2, rollout_data.actions_2, rollout_data.next_observations_2
                context, context_info = self.get_context(x)
                _, newlogprob, entropy, newvalue = self.get_action_and_value(x["obs"], x["action"], context)
                logratio = newlogprob - rollout_data.old_log_prob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = rollout_data.advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - rollout_data.returns) ** 2
                    v_clipped = rollout_data.old_values + torch.clamp(
                        newvalue - rollout_data.old_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - rollout_data.returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - rollout_data.returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if context_info is not None:
                    loss += context_info["loss"]["total_loss"]

                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.context_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                self.critic_optimizer.step()
                self.actor_optimizer.step()
                self.context_optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        writer_dict = {}
        writer_dict["charts/learning_rate"] = self.critic_optimizer.param_groups[0]["lr"]
        writer_dict["losses/value_loss"] = v_loss.item()
        writer_dict["losses/policy_loss"] = pg_loss.item()
        writer_dict["losses/entropy"] = entropy_loss.item()
        writer_dict["losses/old_approx_kl"] = old_approx_kl.item()
        writer_dict["losses/approx_kl"] = approx_kl.item()
        writer_dict["losses/clipfrac"] = np.mean(clipfracs)
        if context_info is not None:
            for key in context_info['loss']:
                writer_dict["losses/" + key] = context_info['loss'][key].item()
        
        return writer_dict
