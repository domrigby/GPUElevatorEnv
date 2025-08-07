import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Import your GPUVectorElevatorEnv and PolicyNet
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from elevator_env import GPUVectorElevatorEnv
from train_elevator_env import PolicyNet
from torch.distributions import Categorical

# Env setup
num_envs = 1
n_elevators = 10
n_floors = 10
capacity = 20
steps = 100
lambdas = torch.full((num_envs, n_floors), 0.2, device=torch_device)

env = GPUVectorElevatorEnv(num_envs, n_elevators, n_floors, capacity, lambdas, max_lambda=1., device=torch_device)
# Use a random policy or load a trained model
policy = PolicyNet(n_elevators, n_floors).to(torch_device)
policy.load_state_dict(torch.load("outputs/elevator_ppo_model.pth")['model_state_dict'])
policy.eval()

# Collect states over time
positions = []  # list of shape (step, n_elevators)
waiting = []    # list of shape (step, n_floors)
obs = env.reset()
total_reward = 0
with torch.no_grad():
    for t in range(steps):
        # Random actions: 0=stop,1=up,2=down (for demo)
        # actions = torch.randint(0, 3, (1, n_elevators), device=torch_device)
        # Or use policy:
        logits, value = policy(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()

        obs, reward, done, info = env.step(actions)
        # move to CPU for plotting
        positions.append((n_floors - 1)* obs['elevator_pos_norm'].cpu().numpy().flatten())
        waiting.append(20 * obs['waiting_norm'].cpu().numpy().flatten())
        total_reward += reward
        print(total_reward, list(actions.cpu()), 9 * obs['elevator_pos_norm'].cpu().numpy().flatten(), obs['waiting_norm'].cpu().numpy().flatten())

# Prepare figure
fig, (ax_elev, ax_wait) = plt.subplots(1, 2, figsize=(8, 4))

# Plot initial elevator positions using first frame
def get_initial_scatter_data():
    xs = list(range(n_elevators))
    ys = positions[0].tolist()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
                 'black', 'white', 'yellow', 'cyan', 'magenta', 'lime', 'olive',
                 'teal', 'navy', 'maroon', 'gold', 'turquoise'][:n_elevators]
    return xs, ys, colors

init_xs, init_ys, init_colors = get_initial_scatter_data()
elev_scatter = ax_elev.scatter(init_xs, init_ys, s=200, c=init_colors)
ax_elev.set_ylim(-0.5, n_floors - 0.5)
ax_elev.set_xlim(-0.5, n_elevators - 0.5)
ax_elev.set_xticks(range(n_elevators))
ax_elev.set_yticks(range(n_floors))
ax_elev.set_title('Elevator Positions')

# Waiting bar chart
bars = ax_wait.bar(range(n_floors), [0]*n_floors)
ax_wait.set_ylim(0, 10)
ax_wait.set_xlabel('Floor')
ax_wait.set_ylabel('Waiting Count')
ax_wait.set_title('Passengers Waiting')

# Animation update function
def update(frame):
    pos = positions[frame]
    wait = waiting[frame]
    # Update elevator scatter
    xs = list(range(n_elevators))
    ys = pos.tolist()
    # Each point as [x, y]
    offsets = list(zip(xs, ys))
    elev_scatter.set_offsets(offsets)
    # Update bar heights
    for bar, h in zip(bars, wait):
        bar.set_height(h)
    fig.suptitle(f'Time step: {frame}')
    return [elev_scatter] + list(bars)

ani = FuncAnimation(
    fig, update, frames=steps, blit=True, interval=200, repeat=False
)

plt.tight_layout()
plt.show()
