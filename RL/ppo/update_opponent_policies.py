import numpy as np

def update_opponent_policies(earlier_policies, rollout_manager, args):
    p = get_prob_dist(num_policies=len(earlier_policies))

    for i in range(len(rollout_manager.processes)):
        policy_dicts = np.random.choice(earlier_policies, 3, p=p)

        rollout_manager.update_policy(policy_dicts[0], process_id=i, policy_id=1)
        rollout_manager.update_policy(policy_dicts[1], process_id=i, policy_id=2)
        rollout_manager.update_policy(policy_dicts[2], process_id=i, policy_id=3)

def get_prob_dist(num_policies, linear_num=800, linear_prob=0.5):
    p = ((1-linear_prob) / num_policies) * np.ones((num_policies,))

    num_aux = min(linear_num, num_policies)
    h = (2 * linear_prob) / (num_aux + 1)
    grad = h / num_aux
    aux_p = np.zeros((num_aux,))
    for i in range(num_aux):
        aux_p[i] += i * grad

    p[-num_aux:] += aux_p

    p = p / np.sum(p)

    return p
