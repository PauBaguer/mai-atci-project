import argparse
import os
from distutils.util import strtobool

def parse_args(variant="discrete"):
    parser = argparse.ArgumentParser(description="PPO training configuration")

    parser.add_argument(
        '--variant',
        choices=["discrete", "continuous"],
        default=variant,
        help="Choose PPO variant: 'discrete' for categorical actions (e.g., CartPole), "
             "'continuous' for Gaussian actions (e.g., Walker2d). Default: %(default)s"
    )

    parser.add_argument(
        '--exp-name',
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="Name of the experiment. Used for logging and saving results. Default: %(default)s"
    )

    # Variant-specific defaults
    if variant == "discrete":
        parser.add_argument(
            '--gym-id',
            type=str,
            default="CartPole-v1",
            help="Gym environment ID to use. Default: %(default)s"
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=2.5e-4,
            help="Learning rate for optimizer. Default: %(default)s"
        )
        parser.add_argument(
            '--num-envs',
            type=int,
            default=8,
            help="Number of parallel environments for experience collection. Default: %(default)s"
        )
        parser.add_argument(
            '--num-steps',
            type=int,
            default=128,
            help="Number of steps per environment per update. Default: %(default)s"
        )
        parser.add_argument(
            '--num-minibatches',
            type=int,
            default=4,
            help="Number of minibatches to split each update batch into. Default: %(default)s"
        )
        parser.add_argument(
            '--update-epochs',
            type=int,
            default=4,
            help="Number of epochs to update policy per batch. Default: %(default)s"
        )
        parser.add_argument(
            '--ent-coef',
            type=float,
            default=0.01,
            help="Entropy coefficient for policy loss (encourages exploration). Default: %(default)s"
        )
        parser.add_argument(
            '--total-timesteps',
            type=int,
            default=500_000,
            help="Total number of environment timesteps for training. Default: %(default)s"
        )
    else:  # continuous
        parser.add_argument(
            '--gym-id',
            type=str,
            default="Walker2d-v5",
            help="Gym environment ID to use. Default: %(default)s"
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=3e-4,
            help="Learning rate for optimizer. Default: %(default)s"
        )
        parser.add_argument(
            '--num-envs',
            type=int,
            default=1,
            help="Number of parallel environments for experience collection. Default: %(default)s"
        )
        parser.add_argument(
            '--num-steps',
            type=int,
            default=2048,
            help="Number of steps per environment per update. Default: %(default)s"
        )
        parser.add_argument(
            '--num-minibatches',
            type=int,
            default=32,
            help="Number of minibatches to split each update batch into. Default: %(default)s"
        )
        parser.add_argument(
            '--update-epochs',
            type=int,
            default=10,
            help="Number of epochs to update policy per batch. Default: %(default)s"
        )
        parser.add_argument(
            '--ent-coef',
            type=float,
            default=0.0,
            help="Entropy coefficient for policy loss (encourages exploration). Default: %(default)s"
        )
        parser.add_argument(
            '--total-timesteps',
            type=int,
            default=1_000_000,
            help="Total number of environment timesteps for training. Default: %(default)s"
        )

    # Shared arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: %(default)s"
    )
    parser.add_argument(
        '--torch-deterministic',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="If true, sets PyTorch to deterministic mode for reproducibility. Default: %(default)s"
    )
    parser.add_argument(
        '--track',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help="Enable experiment tracking with Weights and Biases (wandb). Default: %(default)s"
    )
    parser.add_argument(
        '--wandb-project-name',
        type=str,
        default='ppo',
        help="Weights and Biases project name for tracking. Default: %(default)s"
    )
    parser.add_argument(
        '--wandb-entity-name',
        type=str,
        default=None,
        help="Weights and Biases entity/team name. Default: %(default)s"
    )
    parser.add_argument(
        '--capture-video',
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs='?',
        const=True,
        help="Capture videos of agent performance in the 'videos' folder. Default: %(default)s"
    )
    parser.add_argument(
        '--anneal-lr',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Anneal learning rate linearly from initial value to zero over training. Default: %(default)s"
    )
    parser.add_argument(
        '--gae',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Use Generalized Advantage Estimation for advantage calculation. Default: %(default)s"
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help="Discount factor for rewards. Default: %(default)s"
    )
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help="Lambda parameter for trade-off in GAE bias vs variance. Default: %(default)s"
    )
    parser.add_argument(
        '--norm-adv',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Normalize advantages to mean 0 and std 1 during training. Default: %(default)s"
    )
    parser.add_argument(
        '--clip-coef',
        type=float,
        default=0.2,
        help="Clipping coefficient for PPO surrogate objective. Default: %(default)s"
    )
    parser.add_argument(
        '--clip-vloss',
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs='?',
        const=True,
        help="Use value function clipping to stabilize training. Default: %(default)s"
    )
    parser.add_argument(
        '--vf-coef',
        type=float,
        default=0.5,
        help="Value function loss coefficient. Default: %(default)s"
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help="Maximum gradient norm for gradient clipping. Default: %(default)s"
    )
    parser.add_argument(
        '--target-kl',
        type=float,
        default=0.015,
        help="Early stopping threshold for approximate KL divergence. Default: %(default)s"
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
