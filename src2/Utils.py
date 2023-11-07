import torch


def random_shuffle(states_batch: torch.Tensor,
                   actions_batch: torch.Tensor,
                   rewards_batch: torch.Tensor,
                   next_states_batch: torch.Tensor,
                   advantages_batch: torch.Tensor,
                   seed: int = 0) -> tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    torch.manual_seed(seed)
    batch_size = states_batch.shape[0]
    rng_indices = torch.randperm(n=batch_size)
    return (states_batch[rng_indices],
            actions_batch[rng_indices],
            rewards_batch[rng_indices],
            next_states_batch[rng_indices],
            advantages_batch[rng_indices])
