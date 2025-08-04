import torch

class GPUVectorElevatorEnv:
    """
    Batched elevator environment running purely on GPU, with normalized observations.

    Observations (normalized to [0,1]):
      - elevator_pos_norm: position / (n_floors-1)
      - elevator_load_norm: load / capacity
      - waiting_norm: waiting_count / max_queue (assumed capacity)
      - lambdas_norm: lambda / max_lambda (provided)
      - cumulative_wait_norm: cumulative_wait / (max_wait_horizon * capacity)
    """

    def __init__(
        self,
        num_envs: int,
        n_elevators: int,
        n_floors: int,
        capacity: int,
        lambdas: torch.Tensor,
        max_lambda: float,
        max_wait_horizon: int = 1000,
        device: torch.device = torch.device('cuda'),
    ):
        assert lambdas.shape == (num_envs, n_floors), \
            "lambdas must be of shape (num_envs, n_floors)"

        self.num_envs = num_envs
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.capacity = capacity
        self.max_lambda = max_lambda
        self.max_wait_horizon = max_wait_horizon
        self.device = device

        # Context: Poisson rates per floor per env
        self.lambdas = lambdas.to(self.device)

        # Normalization constants
        self.pos_den = float(max(1, n_floors - 1))
        self.load_den = float(capacity)
        self.wait_den = float(capacity)
        self.lambda_den = float(max_lambda)
        # cumulative_wait max = max_wait_horizon * n_floors * max_queue per floor
        self.cumwait_den = float(max_wait_horizon * n_floors * capacity)

        # Initialize state tensors
        self.elevator_pos = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.elevator_load = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.waiting = torch.zeros((num_envs, n_floors), dtype=torch.long, device=self.device)
        self.cumulative_wait = torch.zeros((num_envs,), dtype=torch.float, device=self.device)

    def reset(self):
        """Reset all envs to initial state."""
        self.elevator_pos.zero_()
        self.elevator_load.zero_()
        self.waiting.zero_()
        self.cumulative_wait.zero_()
        return self._get_obs()

    def _get_obs(self):
        """Return normalized observation dict."""
        # Raw tensors
        pos = self.elevator_pos.float()
        load = self.elevator_load.float()
        wait = self.waiting.float()
        lam = self.lambdas.float()
        cum = self.cumulative_wait.float()

        # Normalization
        obs = {
            'elevator_pos_norm': pos / self.pos_den,
            'elevator_load_norm': load / self.load_den,
            'waiting_norm': wait / self.wait_den,
            'lambdas_norm': lam / self.lambda_den,
            'cumulative_wait_norm': cum / self.cumwait_den,
        }
        return obs

    def step(self, actions: torch.Tensor):
        """
        Step the environment by one time-step.

        actions: LongTensor of shape (num_envs, n_elevators) in {0,1,2}
        returns: obs, reward, done, info
        """
        # Move elevators
        move = torch.zeros_like(self.elevator_pos)
        move[actions == 1] = 1
        move[actions == 2] = -1
        self.elevator_pos = torch.clamp(self.elevator_pos + move, 0, self.n_floors - 1)

        # Handle stops
        stop_mask = (actions == 0)
        if stop_mask.any():
            self.elevator_load[stop_mask] = 0
            batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.n_elevators)
            floor_idx = self.elevator_pos
            wait_floor = self.waiting[batch_idx, floor_idx]
            can_board = torch.minimum(wait_floor, torch.full_like(wait_floor, self.capacity))
            self.elevator_load = torch.where(stop_mask, can_board, self.elevator_load)
            boarded = can_board.sum(dim=1)
            self.waiting = torch.clamp(self.waiting - (boarded.unsqueeze(1) // self.n_elevators), min=0)

        # Arrivals
        arrivals = torch.poisson(self.lambdas)
        self.waiting += arrivals.to(torch.long)

        # Cumulative wait and reward
        step_wait = self.waiting.sum(dim=1).float()
        self.cumulative_wait += step_wait
        reward = -step_wait / 1_000 # instantaneous negative wait per step

        obs = self._get_obs()
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        return obs, reward, done, info
