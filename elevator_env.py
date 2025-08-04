import torch

class GPUVectorElevatorEnv:
    """
    Batched elevator environment running purely on GPU.

    Each batch represents an independent instance of the elevator game.

    Observations:
      - elevator positions (batch x n_elevators)
      - elevator loads (batch x n_elevators)
      - waiting counts (batch x n_floors)
      - context lambdas (batch x n_floors)

    Actions:
      - 0: stop (open/close doors)
      - 1: move up
      - 2: move down

    Rewards:
      - Negative total waiting passengers across all floors (to minimize wait)

    All tensors are managed on a single device (GPU) for vectorized execution.
    """

    def __init__(
        self,
        num_envs: int,
        n_elevators: int,
        n_floors: int,
        capacity: int,
        lambdas: torch.Tensor,
        device: torch.device = torch.device('cuda'),
    ):
        assert lambdas.shape == (num_envs, n_floors), \
            "lambdas must be of shape (num_envs, n_floors)"

        self.num_envs = num_envs
        self.n_elevators = n_elevators
        self.n_floors = n_floors
        self.capacity = capacity
        self.device = device

        # Context: Poisson rates per floor per env
        self.lambdas = lambdas.to(self.device)

        # Initialize state tensors
        self.elevator_pos = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.elevator_load = torch.zeros((num_envs, n_elevators), dtype=torch.long, device=self.device)
        self.waiting = torch.zeros((num_envs, n_floors), dtype=torch.long, device=self.device)

    def reset(self):
        """Reset all envs to initial state."""
        self.elevator_pos.zero_()
        self.elevator_load.zero_()
        self.waiting.zero_()
        return self._get_obs()

    def _get_obs(self):
        """Return observation dict of current state."""
        return {
            'elevator_pos': self.elevator_pos.clone(),
            'elevator_load': self.elevator_load.clone(),
            'waiting': self.waiting.clone(),
            'lambdas': self.lambdas.clone(),
        }

    @torch.inference_mode
    def step(self, actions: torch.Tensor):
        """
        Step the environment by one time-step.

        actions: LongTensor of shape (num_envs, n_elevators) in {0,1,2}
        returns: obs, reward, done, info
        """
        # Move elevators
        # Up: +1 floor, Down: -1 floor, Stop: 0
        move = torch.zeros_like(self.elevator_pos)
        move[actions == 1] = 1
        move[actions == 2] = -1

        # Update positions and clamp within [0, n_floors-1]
        self.elevator_pos = torch.clamp(self.elevator_pos + move, 0, self.n_floors - 1)

        # At stops (action==0), handle boarding/alighting
        stop_mask = (actions == 0)
        if stop_mask.any():
            # Unload all passengers whose destination is this floor (simplest assumption)
            # For demonstration, assume onboard passengers always leave when stopping
            unloading = stop_mask * self.elevator_load
            # drop loads to zero
            self.elevator_load[stop_mask] = 0

            # Boarding: fill with waiting passengers on that floor up to capacity
            # gather waiting for each env and elevator
            # waiting: (num_envs, n_floors)
            # elevator floors: (num_envs, n_elevators)
            batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.n_elevators)
            floor_idx = self.elevator_pos
            wait_at_floor = self.waiting[batch_idx, floor_idx]
            can_board = torch.minimum(
                wait_at_floor,
                torch.full_like(wait_at_floor, self.capacity)
            )
            # Board as much as capacity allows
            self.elevator_load = torch.where(
                stop_mask,
                can_board,
                self.elevator_load
            )
            # Decrease waiting by boarded amount
            total_boarded = can_board.sum(dim=1)
            # Simplest: subtract sum across elevators from waiting (distribute equally)
            self.waiting = torch.clamp(
                self.waiting - total_boarded.unsqueeze(1) // self.n_elevators,
                min=0
            )

        # Generate new arrivals via Poisson
        new_arrivals = torch.poisson(self.lambdas)
        self.waiting += new_arrivals.to(torch.long)

        # Reward: negative sum of waiting passengers
        reward = -self.waiting.sum(dim=1).float()

        obs = self._get_obs()
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        return obs, reward, done, info
