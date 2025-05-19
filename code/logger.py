from torch.utils.tensorboard import SummaryWriter
import wandb

class Logger:
    def __init__(self, run_name, args):
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.wandb = None
        self.track = args.track

        if self.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity_name,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
            self.wandb = wandb

        # Log hyperparameters as a markdown table
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
        )

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        if self.wandb is not None:
            self.wandb.log({tag: value}, step=step)

    def close(self):
        self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
