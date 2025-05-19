from config import parse_args
from env_utils import make_env_continuous
from agent import NormalAgent
from train import train
from logger import Logger
from utils import set_seed
import torch
import gymnasium as gym
import os
import time

def main():
    # Parse command line arguments using continuous variant defaults
    args = parse_args("continuous")

    # Generate a unique run name combining environment, experiment name, seed, and timestamp
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Set random seeds for reproducibility across python, numpy, torch
    set_seed(args.seed)

    # Ensure deterministic behavior for CuDNN backend if requested
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Select device: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a vectorized environment with continuous action space
    # Each env instance is initialized by a thunk from make_env_continuous with its own seed
    envs = gym.vector.SyncVectorEnv(
        [make_env_continuous(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    # Verify the environment has continuous (Box) action space required by this agent
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous (Box) action space required"

    # Wrap to convert dict observations into list form for easier processing if needed
    envs = gym.wrappers.vector.DictInfoToList(envs)

    # Instantiate the NormalAgent (Gaussian policy) and move to the selected device
    agent = NormalAgent(envs).to(device)

    # Create the Adam optimizer with learning rate and epsilon for numerical stability
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize logger to record training metrics and optionally support wandb
    logger = Logger(run_name, args)

    # Execute the PPO training loop with variant flag set to "continuous"
    train(agent, envs, optimizer, args, logger, device, variant="continuous")

    # If video capture and wandb tracking are enabled, upload recorded videos to wandb
    if args.capture_video and args.track:
        import wandb
        for i, v in enumerate(os.listdir(f"videos/{run_name}")):
            filename = v
            # Parse episode number from video filename (e.g., "rl-video-episode-25.mp4")
            episode = int(v.split("-")[-1].replace(".mp4", ""))
            wandb.log({f"media/video-episode-{episode}": wandb.Video(f"videos/{run_name}/{filename}")})
            print("adding video", filename, episode)

    # Finalize logging and close resources
    logger.close()


if __name__ == "__main__":
    main()
