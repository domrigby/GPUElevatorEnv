import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# Import the GPUVectorElevatorEnv from your module
from elevator_env import GPUVectorElevatorEnv

class PolicyNet(nn.Module):
    def __init__(self, n_elevators, n_floors, hidden_size=256):
        super().__init__()
        # Observations: pos_norm, load_norm, waiting_norm, lambdas_norm, cumulative_wait_norm
        obs_size = n_elevators * 2 + n_floors * 2 + 1
        self.fc = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_size, n_elevators * 3))

        self.critic = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.LeakyReLU(), nn.Linear(hidden_size, 1))

    def forward(self, obs):
        # Concatenate normalized observations
        batch = obs['elevator_pos_norm'].shape[0]
        cum_norm = obs['cumulative_wait_norm'].view(batch, 1)
        x = torch.cat([
            obs['elevator_pos_norm'],          # (batch, n_elevators)
            obs['elevator_load_norm'],         # (batch, n_elevators)
            obs['waiting_norm'],               # (batch, n_floors)
            obs['lambdas_norm'],               # (batch, n_floors)
            cum_norm,                          # (batch, 1)
        ], dim=1)
        h = self.fc(x)
        logits = self.actor(h).view(h.size(0), -1, 3)  # (batch, n_elevators, 3)
        value = self.critic(h).squeeze(-1)              # (batch,)
        return logits, value

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.masks = []

    def clear(self):
        for attr in vars(self): setattr(self, attr, [])

if __name__=="__main__":
    # Hyperparameters
    epochs = 100000
    batch_size = 2048
    n_steps = 64
    gamma = 0.99
    gae_lambda = 0.95
    ppo_eps = 0.2
    lr = 1e-4  # lowered learning rate to stabilize updates
    max_grad_norm = 0.5  # tighter clipping threshold

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/elevator_ppo")

    device = torch.device('cuda')

    # Env setup
    num_envs = batch_size
    n_elevators = 10
    n_floors = 10
    capacity = 20
    lambdas = torch.full((num_envs, n_floors), 0.5, device=device)

    env = GPUVectorElevatorEnv(num_envs, n_elevators, n_floors, capacity, lambdas, max_lambda=1., device=device)
    policy = PolicyNet(n_elevators, n_floors).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    # LR scheduler to reduce LR on plateau of policy loss
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    buffer = RolloutBuffer()
    obs = env.reset()

    global_step = 0
    for epoch in range(epochs):
        buffer.clear()
        epoch_reward = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_kl = 0.0
        epoch_grad_norm = 0.0

        obs = env.reset()

        # Collect rollout
        for step in range(n_steps):
            with torch.no_grad():
                logits, value = policy(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=1)
            next_obs, reward, done, _ = env.step(action)
            buffer.obs.append(obs)
            buffer.actions.append(action)
            buffer.logprobs.append(logprob.detach())
            buffer.values.append(value.detach())
            buffer.rewards.append(reward)
            buffer.masks.append((1 - done.float()))
            obs = next_obs
            epoch_reward += reward.sum().item()
            global_step += reward.numel()

            # Step-level logging
            writer.add_scalar('Reward/step', reward.mean().item(), global_step)

        # Log average waiting at end of rollout
        mean_wait = obs['waiting_norm'].float().mean().item()
        writer.add_scalar('Waiting/mean', mean_wait, epoch)

        # Compute returns and advantages
        rewards = torch.stack(buffer.rewards, 0)
        values = torch.stack(buffer.values, 0)
        masks = torch.stack(buffer.masks, 0)
        logprobs = torch.stack(buffer.logprobs, 0)
        with torch.no_grad():
            _, next_value = policy(obs)
        next_value = next_value.detach()

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        running_return = next_value
        running_adv = torch.zeros(batch_size, device=device)

        for t in reversed(range(n_steps)):
            running_return = rewards[t] + gamma * running_return * masks[t]
            if t == n_steps - 1:
                td_error = rewards[t] + gamma * next_value * masks[t] - values[t]
            else:
                td_error = rewards[t] + gamma * values[t+1] * masks[t] - values[t]
            running_adv = td_error + gamma * gae_lambda * running_adv * masks[t]
            returns[t] = running_return
            advantages[t] = running_adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten dims helper
        def flatten(x):
            return x.view(-1, *x.shape[2:]) if x.ndim > 2 else x.view(-1)

        obs_flat = {k: flatten(torch.stack([o[k] for o in buffer.obs], 0)).to(device) for k in buffer.obs[0]}
        actions_flat = flatten(torch.stack(buffer.actions, 0))
        old_logprobs_flat = flatten(logprobs)
        returns_flat = flatten(returns)
        adv_flat = flatten(advantages)

        # PPO update
        for _ in range(4):
            logits, values_pred = policy(obs_flat)
            dist_new = Categorical(logits=logits)
            new_logprob = dist_new.log_prob(actions_flat).sum(dim=1)
            ratio = torch.exp(new_logprob - old_logprobs_flat)

            # KL divergence estimate
            kl = (old_logprobs_flat - new_logprob).mean()

            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1-ppo_eps, 1+ppo_eps) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns_flat - values_pred).pow(2).mean()
            entropy_loss = dist_new.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy += entropy_loss.item()
            epoch_kl += kl.item()
            epoch_grad_norm += grad_norm

        # Adjust learning rate based on policy loss plateau
        # scheduler.step(epoch_policy_loss / 4)

        # Log scalars to TensorBoard
        writer.add_scalar('Reward/epoch', epoch_reward / batch_size, epoch)
        writer.add_scalar('Loss/policy', epoch_policy_loss / 4, epoch)
        writer.add_scalar('Loss/value', epoch_value_loss / 4, epoch)
        writer.add_scalar('Policy/entropy', epoch_entropy / 4, epoch)
        writer.add_scalar('Policy/KL', epoch_kl / 4, epoch)
        writer.add_scalar('Grad/Norm', epoch_grad_norm / 4, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if epoch % 100 == 0:
            model_path = "outputs/elevator_ppo_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': None, #scheduler.state_dict(),
            }, model_path)
            print(f"Saved model checkpoint to {model_path}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Reward {epoch_reward/batch_size:.2f}, "
                  f"Policy Loss {epoch_policy_loss/4:.4f}, Value Loss {epoch_value_loss/4:.4f}, "
                  f"Entropy {epoch_entropy/4:.4f}, KL {epoch_kl/4:.4f}, GradNorm {epoch_grad_norm/4:.4f}, "
                  f"LR {optimizer.param_groups[0]['lr']:.6f}")

    # Close the writer when done
    writer.close()
