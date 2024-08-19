import numpy as np
import torch

class SearchManager:
    def __init__(self, env, policy, search_depth=10, exploration_coefficient=1.0):
        self.env = env
        self.policy = policy
        self.search_depth = search_depth
        self.exploration_coefficient = exploration_coefficient

    def run_search(self, initial_state):
        """
        Runs a forward search from the initial state.
        """
        state = initial_state
        search_results = []

        for depth in range(self.search_depth):
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            search_results.append((state, action, reward))

            if done:
                break

            state = next_state

        return search_results

    def select_action(self, state):
        """
        Selects the action using a combination of policy and exploration.
        """
        action_probabilities = self.policy(state)
        action = self.apply_exploration(action_probabilities)
        return action

    def apply_exploration(self, action_probabilities):
        """
        Applies exploration to the action selection process.
        """
        probabilities = action_probabilities + self.exploration_coefficient * np.random.dirichlet(
            np.ones_like(action_probabilities))
        action = np.argmax(probabilities)
        return action

    def evaluate_results(self, search_results):
        """
        Evaluates the results of the search and returns the best action.
        """
        cumulative_rewards = np.zeros(len(search_results))
        for i in range(len(search_results)):
            _, _, reward = search_results[i]
            cumulative_rewards[i] = reward

        best_index = np.argmax(cumulative_rewards)
        best_action = search_results[best_index][1]

        return best_action
