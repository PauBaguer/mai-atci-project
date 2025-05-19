from config import parse_args
from env_utils import make_env_discrete
from agent import CategoricalAgent
from train import train
from logger import Logger
from utils import set_seed
import torch
import gymnasium as gym
import os
import time

def main():
    # Parse command line arguments with 'discrete' variant defaults
    args = parse_args("discrete")

    # Create a unique run name with environment, experiment name, seed, and current timestamp
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Set random seeds for reproducibility (Python, numpy, torch)
    set_seed(args.seed)

    # Set PyTorch CuDNN backend determinism for reproducible GPU behavior
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create vectorized environments with discrete action space
    # Each env uses a thunk returned by make_env_discrete, seeded differently
    envs = gym.vector.SyncVectorEnv(
        [make_env_discrete(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Confirm environment action space is discrete (required for this agent)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Discrete action space required"

    # Convert any Dict info observations into list form for easier processing
    envs = gym.wrappers.vector.DictInfoToList(envs)

    # Initialize the categorical policy/value network agent
    agent = CategoricalAgent(envs).to(device)

    # Create Adam optimizer with specified learning rate and epsilon for numerical stability
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize logger for Tensorboard and optionally wandb tracking
    logger = Logger(run_name, args)

    # Run the PPO training loop with the discrete variant flag
    train(agent, envs, optimizer, args, logger, device, variant="discrete")

    # If video capture and wandb tracking enabled, upload saved videos to wandb
    if args.capture_video and args.track:
        import wandb
        for i, v in enumerate(os.listdir(f"videos/{run_name}")):
            filename = v
            # Extract episode number from filename like "rl-video-episode-25.mp4"
            episode = int(v.split("-")[-1].replace(".mp4", ""))
            wandb.log({f"media/video-episode-{episode}": wandb.Video(f"videos/{run_name}/{filename}")})
            print("adding video", filename, episode)

    # Close logger and finalize wandb run if used
    logger.close()


if __name__ == "__main__":
    main()
