import torch


def random_shuffle(states_batch: list[torch.Tensor],
                   actions_batch: list[torch.Tensor],
                   rewards_batch: list[torch.Tensor],
                   next_states_batch: list[torch.Tensor],
                   advantages_batch: list[torch.Tensor],
                   seed: int = 0) -> tuple[list[torch.Tensor],
                                           list[torch.Tensor],
                                           list[torch.Tensor],
                                           list[torch.Tensor],
                                           list[torch.Tensor]]:
    torch.manual_seed(seed)
    batch_size = len(states_batch)
    rng_indices = torch.randperm(n=batch_size)
    return ([states_batch[idx] for idx in rng_indices],
            [actions_batch[idx] for idx in rng_indices],
            [rewards_batch[idx] for idx in rng_indices],
            [next_states_batch[idx] for idx in rng_indices],
            [advantages_batch[idx] for idx in rng_indices])
