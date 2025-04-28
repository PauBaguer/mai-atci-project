import argparse
import os
import time
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import gym

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='Experiment name')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='Id of the Gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000,
                        help='Total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False` is used to have the same results')

    #wanb
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default='ppo',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity-name', type=str, default=None,
                        help="The entity (team) of wandb's project")


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        )
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env(gym_id):
        def _make_env():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda t: t% 100 == 0)
            return env
        return _make_env

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])

    observation = envs.reset()
    try:
        for _ in range(2000):
            action = envs.action_space.sample()
            observation, reward, terminated, truncated, info = envs.step(action)

            if 'final_info' in info.keys():
                for item in info['final_info']:
                    if type(item) == dict and 'episode' in item.keys():
                        print(f"global_step={item['episode']['l']}, episodic_return={item['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", item['episode']['r'], item['episode']['l'])
                        # writer.add_scalar("charts/episodic_length", item['episode']['l'], item['episode']['l'])
                        break

    finally:
        envs.close()
