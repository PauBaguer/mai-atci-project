import torch
import numpy as np

from utils import compute_gae
from utils import explained_variance

def train(agent, envs, optimizer, args, logger, device, variant="discrete"):
    """
    Main training loop for PPO.

    Args:
        agent: The policy/value network.
        envs: Vectorized environments.
        optimizer: Optimizer for updating the agent.
        args: Configuration arguments.
        logger: Logging utility for metrics.
        device: Torch device (CPU or CUDA).
        variant: 'discrete' or 'continuous' action space variant.
    """

    # Initialize tensors to store rollouts (observations, actions, logprobs, rewards, dones, values)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    import time
    global_step = 0
    start_time = time.time()

    # Reset envs and get initial observations
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Main training loop: iterate over number of updates
    for update in range(1, num_updates + 1):

        # Anneal learning rate linearly

        frac = 1.0 - (update - 1.0) / num_updates  # from 1 to 0 over training
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]['lr'] = lrnow

        # Collect rollout data over num_steps steps per environment
        for step in range(args.num_steps):
            global_step += args.num_envs  # increment by batch size
            obs[step] = next_obs  # store observation
            dones[step] = next_done  # store done flags

            with torch.no_grad():
                # Query policy for action, log prob, and value estimate
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action  # store action taken
            logprobs[step] = logprob  # store log prob of action

            # Step envs with actions and collect reward/info
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)  # store rewards
            next_obs = torch.Tensor(next_obs).to(device)  # update next obs
            next_done = torch.Tensor(terminated | truncated).to(device)  # update done flags

            # Log episodic returns if episode finished
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    logger.log_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    logger.log_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # Compute value estimates for next_obs to bootstrap returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda, device)



        # Flatten batch tensors for training update
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # PPO policy and value update epochs
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # shuffle batch indices for minibatch sampling
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Evaluate current policy on minibatch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds] if variant == "continuous" else b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Approximate KL divergence for early stopping
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                # Normalize advantages if enabled
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PPO clipped surrogate policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss clipped
                newvalue = newvalue.view(-1)

                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()


                entropy_loss = entropy.mean()

                # Total loss: policy loss - entropy bonus + value loss weighted
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backprop and gradient step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stop if KL divergence exceeds target
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Calculate explained variance to measure value function fit quality
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        explained_var = explained_variance(y_pred, y_true)

        # Log metrics to TensorBoard and wandb
        logger.log_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.log_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.log_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.log_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.log_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.log_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.log_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.log_scalar("losses/explained_variance", explained_var, global_step)

        elapsed_time = time.time() - start_time
        sps = int(global_step / elapsed_time)  # Steps per second
        print(f"SPS: {sps}")
        logger.log_scalar("charts/SPS", sps, global_step)

    envs.close()
