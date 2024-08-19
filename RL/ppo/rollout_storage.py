import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_dim, recurrent_hidden_state_dim, action_space, device):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_dim = obs_dim
        self.recurrent_hidden_state_dim = recurrent_hidden_state_dim
        self.action_space = action_space
        self.device = device

        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_dim, dtype=torch.float).to(device)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_dim, dtype=torch.float).to(device)
        self.actions = torch.zeros(num_steps, num_processes, *action_space.shape, dtype=torch.long).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, dtype=torch.float).to(device)
        self.masks = torch.ones(num_steps, num_processes, dtype=torch.float).to(device)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, dtype=torch.float).to(device)
        self.returns = torch.zeros(num_steps + 1, num_processes, dtype=torch.float).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, dtype=torch.float).to(device)
        self.advantages = torch.zeros(num_steps, num_processes, dtype=torch.float).to(device)
        self.old_action_log_probs = torch.zeros(num_steps, num_processes, dtype=torch.float).to(device)

    def insert(self, step, obs, recurrent_hidden_states, actions, rewards, masks, value_preds, action_log_probs):
        self.observations[step + 1].copy_(obs)
        self.recurrent_hidden_states[step + 1].copy_(recurrent_hidden_states)
        self.actions[step].copy_(actions)
        self.rewards[step].copy_(rewards)
        self.masks[step].copy_(masks)
        self.value_preds[step].copy_(value_preds)
        self.action_log_probs[step].copy_(action_log_probs)

    def compute_advantages(self, next_value, gamma, gae_lambda):
        self.returns[-1] = next_value
        self.advantages[-1] = 0

        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step] - self.value_preds[step]
            self.advantages[step] = delta + gamma * gae_lambda * self.advantages[step + 1] * self.masks[step]
            self.returns[step] = self.advantages[step] + self.value_preds[step]

    def get(self):
        return self.observations, self.recurrent_hidden_states, self.actions, self.rewards, self.masks, \
               self.value_preds, self.returns, self.action_log_probs, self.advantages
