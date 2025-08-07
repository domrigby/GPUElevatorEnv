import torch

class GPUVectorElevatorEnv:
    """
    Batched elevator environment running purely on GPU, with normalized observations.

    Observations (normalized to [0,1]):
      - elevator_pos_norm: position / (n_floors-1)
      - elevator_load_norm: load / capacity
      - waiting_norm: waiting_count / max_queue (assumed capacity)
      - lambdas_norm: lambda / max_lambda (provided)
      - cumulative_wait_norm: cumulative_wait / (max_wait_horizon * n_floors * capacity)

    Actions (per elevator):
      0 = stop (load/unload if applicable)
      1 = move up
      2 = move down
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
        allow_ground_arrivals: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        assert lambdas.shape == (num_envs, n_floors), \
            "lambdas must be of shape (num_envs, n_floors)"

        self.num_envs = num_envs
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.capacity = int(capacity)
        self.max_lambda = float(max_lambda)
        self.max_wait_horizon = int(max_wait_horizon)
        self.allow_ground_arrivals = bool(allow_ground_arrivals)
        self.device = device

        # Poisson rates per floor per env
        self.lambdas = lambdas.to(self.device)

        # Normalization constants
        self.pos_den = float(max(1, n_floors - 1))
        self.load_den = float(capacity)
        self.wait_den = float(capacity)
        self.lambda_den = float(max_lambda)
        self.cumwait_den = float(max_wait_horizon * n_floors * capacity)

        # State
        self.elevator_pos = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.elevator_load = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.waiting = torch.zeros((num_envs, n_floors), dtype=torch.long, device=self.device)
        self.cumulative_wait = torch.zeros((num_envs,), dtype=torch.float, device=self.device)
        self.delivered = torch.zeros((num_envs,), dtype=torch.long, device=self.device)  # total drop-offs
        self.t = torch.zeros((), dtype=torch.long, device=self.device)  # global step counter

        # If ground-floor arrivals not allowed, zero them in the rate tensor (doesn't mutate caller's tensor)
        if not self.allow_ground_arrivals:
            self.lambdas = self.lambdas.clone()
            self.lambdas[:, 0] = 0.0

    def reset(self):
        """Reset all envs to initial state."""
        self.elevator_pos.zero_()
        self.elevator_load.zero_()
        self.waiting.zero_()
        self.cumulative_wait.zero_()
        self.delivered.zero_()
        self.t.zero_()
        return self._get_obs()

    def _get_obs(self):
        """Return normalized observation dict."""
        obs = {
            'elevator_pos_norm': self.elevator_pos.float() / self.pos_den,
            'elevator_load_norm': self.elevator_load.float() / self.load_den,
            'waiting_norm': self.waiting.float() / self.wait_den,
            'lambdas_norm': self.lambdas.float() / self.lambda_den,
            'cumulative_wait_norm': self.cumulative_wait.float() / self.cumwait_den,
        }
        return obs

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """
        Step the environment by one time-step.

        actions: LongTensor of shape (num_envs, n_elevators) with values in {0,1,2}
        returns: obs, reward, done, info
        """
        actions = actions.to(self.device).long()
        assert actions.shape == (self.num_envs, self.n_elevators)
        if torch.any((actions < 0) | (actions > 2)):
            raise ValueError("actions must be in {0,1,2}")

        # === Move elevators ===
        move = torch.zeros_like(self.elevator_pos)
        move[actions == 1] = 1
        move[actions == 2] = -1
        self.elevator_pos = torch.clamp(self.elevator_pos + move, 0, self.n_floors - 1)

        # === Handle stops (unload at ground, then board) ===
        stop_mask = (actions == 0)
        any_stop = bool(stop_mask.any())

        boarded_this_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        unloaded_this_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        if any_stop:
            pos = self.elevator_pos  # (E, K)
            load = self.elevator_load  # (E, K)
            E, K, F = self.num_envs, self.n_elevators, self.n_floors

            # Unload only at ground on stop
            at_ground = (pos == 0)
            unload_mask = stop_mask & at_ground
            unload_counts = torch.where(unload_mask, load, torch.zeros_like(load))
            # Update loads and delivered counters
            if unload_counts.any():
                unloaded_per_env = unload_counts.sum(dim=1)
                unloaded_this_step += unloaded_per_env
                self.delivered += unloaded_per_env
                load = load - unload_counts  # zero where unload_mask else unchanged

            # Board passengers (all non-ground floors)
            remaining_cap = (self.capacity - load).clamp(min=0)  # (E, K)

            # Gather floor waiting where each elevator is (0 at ground)
            batch_idx = torch.arange(E, device=self.device).unsqueeze(1).expand(E, K)
            floor_idx = pos
            wait_at_floor = self.waiting[batch_idx, floor_idx]  # (E, K)
            wait_at_floor = torch.where(pos == 0, torch.zeros_like(wait_at_floor), wait_at_floor)

            # Provisional per-elevator board requests limited by remaining cap and queue
            prov_board = torch.minimum(remaining_cap, wait_at_floor)
            prov_board = torch.where(stop_mask, prov_board, torch.zeros_like(prov_board))

            # Apportion fairly when multiple elevators stop on same (env, floor)
            # Flatten (env, floor) to a single index for scatter-add
            flat_floor = (batch_idx * F + floor_idx).view(-1)  # (E*K,)
            prov_board_f = prov_board.float().view(-1)  # scatter_add needs float
            stop_flat_mask = stop_mask.view(-1)

            totals_attempted = torch.zeros(E * F, dtype=torch.float, device=self.device)
            totals_attempted.scatter_add_(0, flat_floor[stop_flat_mask], prov_board_f[stop_flat_mask])

            wait_flat = self.waiting.view(-1).float()
            actual_total_boarded_flat = torch.minimum(wait_flat, totals_attempted)  # per (env,floor)

            # Compute ratio per (env,floor) and map back to each elevator
            eps = 1e-8
            ratio_flat = actual_total_boarded_flat / (totals_attempted + eps)
            ratio_per_elev = ratio_flat[flat_floor].view(E, K)

            # Actual per-elevator boarded (integer)
            actual_board = torch.floor(prov_board.float() * ratio_per_elev).to(torch.long)

            # Update loads
            load = (load + actual_board).clamp(max=self.capacity)
            self.elevator_load = load

            # Subtract boarded from waiting per floor (using actual totals recomputed from per-elevator)
            actual_board_flat = actual_board.float().view(-1)
            boarded_per_floor_flat = torch.zeros(E * F, dtype=torch.float, device=self.device)
            boarded_per_floor_flat.scatter_add_(0, flat_floor[stop_flat_mask], actual_board_flat[stop_flat_mask])

            # Apply to waiting
            waiting_flat = self.waiting.view(-1).float()
            waiting_flat = torch.clamp(waiting_flat - boarded_per_floor_flat, min=0.0)
            self.waiting = waiting_flat.view(E, F).to(torch.long)

            # Stats
            boarded_this_step += actual_board.sum(dim=1)

        # === Arrivals ===
        arrivals = torch.poisson(self.lambdas)  # (E, F), float
        if not self.allow_ground_arrivals:
            arrivals = arrivals.clone()
            arrivals[:, 0] = 0.0
        self.waiting += arrivals.to(torch.long)

        # === Cumulative wait and reward ===
        step_wait = self.waiting.sum(dim=1).float()
        self.cumulative_wait += step_wait
        reward = -step_wait / 10_000.0  # instantaneous negative wait per step

        # === Time / done ===
        self.t += 1
        done = torch.full((self.num_envs,), self.t.item() >= self.max_wait_horizon, dtype=torch.bool, device=self.device)

        obs = self._get_obs()
        info = {
            "boarded": boarded_this_step,
            "unloaded": unloaded_this_step,
            "delivered_total": self.delivered.clone(),
            "time_step": int(self.t.item()),
        }
        return obs, reward, done, info
