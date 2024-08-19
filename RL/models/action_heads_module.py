import math
import torch
import torch.nn as nn

from RL.distributions import Categorical, DiagGaussian
from game.enums import ActionTypes

DEFAULT_MLP_SIZE = 64


class ActionHead(nn.Module):
    def __init__(self, in_dim, out_dim, mlp_size=DEFAULT_MLP_SIZE, custom_inputs=None, custom_out_size=None, id=None):
        super(ActionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.custom_inputs = custom_inputs
        self.mlp_size = mlp_size
        self.id = id
        self.custom_out_size = custom_out_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.mlp = init_(nn.Linear(in_dim + (custom_out_size if custom_inputs is not None else 0), mlp_size))
        self.relu = nn.ReLU()
        self.output = init_(nn.Linear(mlp_size, out_dim))
        self.dist = Categorical(mlp_size, out_dim)

    def forward(self, x, masks=None, custom_inputs=None):
        if self.custom_inputs is not None:
            for custom_input_key in self.custom_inputs.keys():
                if custom_input_key not in custom_inputs:
                    raise ValueError(f"Custom input {custom_input_key} is missing")
            x = torch.cat((x, custom_inputs[self.custom_inputs]), dim=-1)
        x = self.relu(self.mlp(x))
        dist = self.dist(x, masks)
        return dist


class RecurrentResourceActionHead(nn.Module):
    def __init__(self, in_dim, out_dim, max_count, mlp_size=DEFAULT_MLP_SIZE, mask_based_on_curr_res=False, id=None):
        super(RecurrentResourceActionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_count = max_count
        self.mask_based_on_curr_res = mask_based_on_curr_res
        self.id = id

        self.mlp = nn.Linear(in_dim, mlp_size)
        self.output = nn.Linear(mlp_size, out_dim)

    def forward(self, x, masks=None, custom_inputs=None):
        x = self.relu(self.mlp(x))
        return self.dist(x, masks)


class MultiActionHeadsGeneralised(nn.Module):
    def __init__(self, action_heads, autoregressive_map, lstm_dim, log_prob_masks=None,
                 type_conditional_action_masks=None):
        super(MultiActionHeadsGeneralised, self).__init__()
        self.action_heads = action_heads
        self.autoregressive_map = autoregressive_map
        self.lstm_dim = lstm_dim
        self.log_prob_masks = log_prob_masks
        self.type_conditional_action_masks = type_conditional_action_masks

    def forward(self, main_input, masks=None, custom_inputs=None, deterministic=False, condition_on_action_type=None,
                log_specific_head_probs=False, actions=None):
        if log_specific_head_probs:
            log_output = {}

        for i, action_head in enumerate(self.action_heads):
            masks_ = masks[i] if masks else None
            custom_inputs_ = custom_inputs if custom_inputs else None

            if condition_on_action_type is not None:
                masks_ = masks_[torch.arange(masks_.size(0)), condition_on_action_type]

            if self.type_conditional_action_masks[i]:
                masks_ = masks_.masked_fill(self.type_conditional_action_masks[i], 0)

            dist = action_head(main_input, masks=masks_, custom_inputs=custom_inputs_)

            if actions is not None:
                log_probs = dist.log_probs(actions[i])
            else:
                log_probs = dist.log_probs(dist.mode())

            if log_specific_head_probs:
                log_output[action_head.id] = dist.log_probs(dist.mode())

        return dist.mode(), log_probs, dist.entropy(), log_output
