import gymnasium as gym
import numpy as np


def make_env_discrete(gym_id, seed, idx, capture_video, run_name):
    """
    Factory function to create a discrete action space environment with common wrappers.

    Args:
        gym_id (str): Gym environment ID.
        seed (int): Random seed for reproducibility.
        idx (int): Index of the environment (used to condition video capture).
        capture_video (bool): Whether to record video of agent performance.
        run_name (str): Unique identifier for the current experiment run.

    Returns:
        thunk (function): A function that creates and returns the wrapped environment instance.
    """

    def thunk():
        # Create the gym environment with RGB array rendering mode
        env = gym.make(gym_id, render_mode="rgb_array")

        # Wrap to record episode statistics like returns and lengths
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Record video only for the first environment (idx == 0) and if capture_video enabled
        if capture_video and idx == 0:
            # Record video every 25 episodes in 'videos/{run_name}' folder at 50 fps
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                fps=50,
                episode_trigger=lambda episode_id: episode_id % 25 == 0
            )

        # Seed the action and observation spaces for reproducibility
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk


def make_env_continuous(gym_id, seed, idx, capture_video, run_name):
    """
    Factory function to create a continuous action space environment with normalization and clipping wrappers.

    Args:
        gym_id (str): Gym environment ID.
        seed (int): Random seed for reproducibility.
        idx (int): Index of the environment (used to condition video capture).
        capture_video (bool): Whether to record video of agent performance.
        run_name (str): Unique identifier for the current experiment run.

    Returns:
        thunk (function): A function that creates and returns the wrapped environment instance.
    """

    def thunk():
        # Create the gym environment with RGB array rendering mode
        env = gym.make(gym_id, render_mode="rgb_array")

        # Wrap to record episode statistics like returns and lengths
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Record video only for the first environment (idx == 0) and if capture_video enabled
        if capture_video and idx == 0:
            # Record video every 25 episodes in 'videos/{run_name}' folder at 50 fps
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                fps=50,
                episode_trigger=lambda episode_id: episode_id % 25 == 0
            )

        # Clip actions to ensure they are within valid bounds
        env = gym.wrappers.ClipAction(env)

        # Normalize observations to zero mean and unit variance (running estimate)
        env = gym.wrappers.NormalizeObservation(env)

        # Clip normalized observations to the range [-10, 10] to avoid extreme values
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)

        # Normalize rewards to zero mean and unit variance (running estimate)
        env = gym.wrappers.NormalizeReward(env)

        # Clip normalized rewards to the range [-10, 10] to avoid large spikes
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        # Seed the action and observation spaces for reproducibility
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk
