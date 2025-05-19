import numpy as np
import torch

def explained_variance(y_pred, y_true):
    """
    Compute the explained variance in a given prediction.

    The explained variance is computed as one minus the ratio of the variance of
    the residuals to the variance of the original data. This gives a score of
    how well the prediction explains the data.

    Parameters
    ----------
    y_pred : array-like
        Array of values predicted by a model.
    y_true : array-like
        Array of ground truth values to compare to `y_pred`.

    Returns
    -------
    explained_variance : float
        The fraction of the variance in `y_true` explained by `y_pred`.
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return float('nan')
    return 1 - np.var(y_true - y_pred) / var_y


def set_seed(seed):
    """
    Set the seed for random number generators in Python, NumPy, and PyTorch.

    This function helps ensure reproducibility of experiments by setting the
    seed for the random number generators used by the random module, NumPy,
    and PyTorch.

    Args:
        seed (int): The seed value to be used by the random number generators.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda, device):
    """
    Compute the Generalized Advantage Estimation (GAE).

    GAE is used to reduce variance in policy gradient methods by providing a
    balance between bias and variance through the `gae_lambda` parameter.

    Args:
        rewards (torch.Tensor): Tensor of rewards received at each time step.
        values (torch.Tensor): Tensor of value function estimates at each time step.
        dones (torch.Tensor): Tensor indicating episode termination at each time step (1 if done, 0 otherwise).
        next_value (float): The value function estimate for the next state.
        gamma (float): Discount factor for rewards.
        gae_lambda (float): GAE parameter that controls the bias-variance trade-off.
        device (torch.device): Device to perform computation on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing:
            - advantages (torch.Tensor): Estimated advantage for each time step.
            - returns (torch.Tensor): Estimated returns for each time step.
    """
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            nextnonterminal = 1.0 - dones[-1]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns
