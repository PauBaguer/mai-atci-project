import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize neural network layers with orthogonal weights and constant bias.
    Orthogonal initialization helps with stable training.

    Args:
        layer (nn.Module): The layer to initialize.
        std (float): Standard deviation for weight initialization.
        bias_const (float): Constant value to initialize bias.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)  # Orthogonal weight init with std scaling
    torch.nn.init.constant_(layer.bias, bias_const)  # Constant bias init
    return layer


class CategoricalAgent(nn.Module):
    """
    Policy and value network agent for environments with discrete action spaces.
    Uses categorical distribution over actions.
    """

    def __init__(self, envs):
        super().__init__()
        # Flatten observation space shape to input vector size
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        # Number of discrete actions
        n_actions = envs.single_action_space.n

        # Critic network: estimates state value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # Output layer (value scalar)
        )

        # Actor network: outputs logits for categorical distribution over actions
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),  # Output layer (logits per action)
        )

    def get_value(self, x):
        """
        Compute state value for observation x.

        Args:
            x (Tensor): Batch of observations.

        Returns:
            Tensor: Estimated state values.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Compute action distribution, sample action (if not provided),
        log probability and entropy of the action, and state value.

        Args:
            x (Tensor): Batch of observations.
            action (Tensor or None): Optional actions to evaluate.

        Returns:
            action (Tensor): Selected or provided actions.
            log_prob (Tensor): Log probability of the actions.
            entropy (Tensor): Entropy of the action distribution.
            value (Tensor): State value estimates.
        """
        logits = self.actor(x)  # Actor outputs logits for each action
        probs = Categorical(logits=logits)  # Categorical distribution over actions

        if action is None:
            action = probs.sample()  # Sample action if none provided

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class NormalAgent(nn.Module):
    """
    Policy and value network agent for environments with continuous action spaces.
    Uses Gaussian (Normal) distribution with learned mean and stddev.
    """

    def __init__(self, envs):
        super().__init__()
        # Flatten observation space shape to input vector size
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        # Flatten action space shape (continuous vector)
        act_shape = np.prod(envs.single_action_space.shape)

        # Critic network: estimates state value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # Output layer (value scalar)
        )

        # Actor mean network: outputs mean of Gaussian for each action dimension
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),  # Input layer
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),  # Hidden layer
            nn.Tanh(),
            layer_init(nn.Linear(64, act_shape), std=0.01),  # Output layer (mean vector)
        )
        # Log standard deviation parameter (learnable), shared across batch
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape))

    def get_value(self, x):
        """
        Compute state value for observation x.

        Args:
            x (Tensor): Batch of observations.

        Returns:
            Tensor: Estimated state values.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Compute Gaussian action distribution (mean and std), sample action if none provided,
        compute log probability and entropy of actions, and state value.

        Args:
            x (Tensor): Batch of observations.
            action (Tensor or None): Optional actions to evaluate.

        Returns:
            action (Tensor): Selected or provided actions.
            log_prob (Tensor): Log probability of the actions (summed over action dims).
            entropy (Tensor): Entropy of the action distribution (summed over dims).
            value (Tensor): State value estimates.
        """
        action_mean = self.actor_mean(x)  # Mean vector of Gaussian
        action_logstd = self.actor_logstd.expand_as(action_mean)  # Log std dev expanded to batch size
        action_std = torch.exp(action_logstd)  # Convert log std to std
        probs = Normal(action_mean, action_std)  # Normal distribution for continuous actions

        if action is None:
            action = probs.sample()  # Sample action if none provided

        # Log prob and entropy summed over action dimensions for each batch element
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
