import numpy as np
import torch

def calculate_cumulative_reward(search_results, discount_factor=0.99):
    """
    Calculates the cumulative discounted reward from a sequence of search results.
    """
    cumulative_reward = 0.0
    for i, (_, _, reward) in enumerate(search_results):
        cumulative_reward += (discount_factor ** i) * reward
    return cumulative_reward

def restore_environment_state(env, state):
    """
    Restores the environment to a previous state.
    """
    env.restore_state(state)

def save_environment_state(env):
    """
    Saves the current environment state.
    """
    return env.clone_state()

def apply_policy(policy, state):
    """
    Applies the policy to a given state to obtain action probabilities.
    """
    with torch.no_grad():
        action_probabilities = policy(state)
    return action_probabilities
