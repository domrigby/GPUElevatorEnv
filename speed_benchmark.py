import time
import torch
import pandas as pd
import plotly.express as px
import subprocess
import shutil

# Import your environment and policy definitions
from elevator_env import GPUVectorElevatorEnv
from train_elevator_env import PolicyNet

# Configuration
batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
timesteps_per_batch = 1000  # number of environment steps per batch
n_elevators = 4
n_floors = 10
capacity = 20
lambdas_val = 0.5

# Devices
device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda') if torch.cuda.is_available() else None

def init(batch_size, device):
    env = GPUVectorElevatorEnv(
        num_envs=batch_size,
        n_elevators=n_elevators,
        n_floors=n_floors,
        capacity=capacity,
        lambdas=torch.full((batch_size, n_floors), lambdas_val, device=device),
        device=device
    )
    policy = PolicyNet(n_elevators, n_floors).to(device)
    policy.eval()
    obs = env.reset()
    return env, policy, obs

records = []
for bs in batch_sizes:
    # Benchmark CPU
    env_cpu, policy_cpu, obs_cpu = init(bs, device_cpu)
    actions_cpu = torch.randint(0, 3, (bs, n_elevators), device=device_cpu)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(timesteps_per_batch):
            obs_cpu, reward_cpu, done_cpu, _ = env_cpu.step(actions_cpu)
            _ = policy_cpu(obs_cpu)
    elapsed_cpu = time.time() - start_time
    records.append({'batch_size': bs, 'time_sec': elapsed_cpu, 'device': 'cpu'})

    # Benchmark GPU
    if device_gpu:
        env_gpu, policy_gpu, obs_gpu = init(bs, device_gpu)
        actions_gpu = torch.randint(0, 3, (bs, n_elevators), device=device_gpu)
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                o, r, d, _ = env_gpu.step(actions_gpu)
                _ = policy_gpu(o)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(timesteps_per_batch):
                obs_gpu, reward_gpu, done_gpu, _ = env_gpu.step(actions_gpu)
                _ = policy_gpu(obs_gpu)
        torch.cuda.synchronize()
        elapsed_gpu = time.time() - start_time
        records.append({'batch_size': bs, 'time_sec': elapsed_gpu, 'device': 'gpu'})

# Create DataFrame
df = pd.DataFrame(records)

# Plot: Experience generated over time = batch_size * timesteps_per_batch / elapsed_time
# Calculate samples_per_second
df['samples_per_sec'] = df['batch_size'] * timesteps_per_batch / df['time_sec']

fig = px.line(
    df,
    x='batch_size',
    y='samples_per_sec',
    color='device',
    title='Samples per Second vs Batch Size',
    markers=True,
    labels={'samples_per_sec': 'Samples/sec', 'batch_size': 'Batch Size'}
)

# Save and open
output_filename = 'benchmark_samples_per_sec.html'
fig.write_html(output_filename, include_plotlyjs='cdn')
try:
    if shutil.which('firefox'):
        subprocess.run(['firefox', output_filename], check=False)
    else:
        subprocess.run(['xdg-open', output_filename], check=False)
except:
    pass
print(f"Plot saved to {output_filename}")
