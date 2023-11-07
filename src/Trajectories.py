import torch
import gymnasium as gym
import numpy as np

from src import PolicyNetwork
from src import MultiHeadNetwork


class Trajectories:
    def __init__(self,
                 batch_size: int,
                 max_game_length: int,
                 env: gym.Env,
                 policy_network: PolicyNetwork,
                 multihead_network: MultiHeadNetwork,
                 seed: int,
                 architecture: str = 'Individual Networks'):
        if not isinstance(batch_size, int):
            raise TypeError(f'batch_size must be an integer, not {type(batch_size)}.')
        if batch_size < 1:
            raise ValueError(f'batch_size must be a positive integer >= 1, not {batch_size}.')

        self.batch_size = batch_size
        self.max_game_length = max_game_length
        self.env = env
        self.architecture = architecture
        self.policy_network = policy_network
        self.multihead_network = multihead_network
        self.seed = seed
        torch.manual_seed(self.seed)

    def get_trajectory(self, seed: int):
        # Retrieving initial state
        state, info = self.env.reset(seed=seed)

        # For storing trajectory
        states, actions, rewards, next_states = [], [], [], []

        done = False
        while not done and len(rewards) < self.max_game_length:
            # Retrieve current action prob. distribution
            if self.architecture == 'Individual Networks':
                action_probs = self.policy_network(torch.tensor(state))
            elif self.architecture == 'Multi Head Network':
                _, action_probs = self.multihead_network(torch.tensor(state))
            else:
                raise ValueError(f'Architecture must be one of: {["Individual Networks", "Multi Head Network"]}.')

            # Sample action from distribution
            action = torch.multinomial(input=action_probs, num_samples=1)

            # interact with env.
            next_state, reward, done, _, _ = self.env.step(action.item())

            # Store trajectory data
            states.append(state.tolist())
            actions.append(action.item())
            rewards.append(reward)
            next_states.append(next_state.tolist())

            # Update current state
            state = next_state

        return states, actions, rewards, next_states

    def get_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        States_batch has dims:      (batch size , game length , state space size)
        next_states_batch has dims: (batch size , game length , state space size)
        actions_batch has dims:     (batch size,  game length)
        rewards_batch has dims:     (batch size,  game length)
        """

        states_batch, actions_batch, rewards_batch, next_states_batch = [], [], [], []
        for datapoint in range(self.batch_size):
            states, actions, rewards, next_states = self.get_trajectory(seed=self.seed + datapoint)
            states_batch.append(torch.tensor(states))
            actions_batch.append(torch.tensor(actions))
            rewards_batch.append(torch.tensor(rewards))
            next_states_batch.append(torch.tensor(next_states))

        return states_batch, actions_batch, rewards_batch, next_states_batch
