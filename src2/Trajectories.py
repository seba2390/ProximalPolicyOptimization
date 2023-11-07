import torch
import gym
import numpy as np

from src.PolicyNetwork import PolicyNetwork


class Trajectories:
    def __init__(self,
                 batch_size: int,
                 env: gym.Env,
                 policy_network: PolicyNetwork):
        if not isinstance(batch_size,int):
            raise TypeError(f'batch_size must be an integer, not {type(batch_size)}.')
        if batch_size < 1:
            raise ValueError(f'batch_size must be a positive integer >= 1, not {batch_size}.')

        self.batch_size = batch_size
        self.env = env
        self.policy_network = policy_network

    def get_trajectory(self):
        # Retrieving initial state
        state, info = self.env.reset()

        # For storing trajectory
        states, actions, rewards, next_states = [], [], [], []

        done = False
        while not done:
            # Retrieve current action prob. distribution
            action_probs = self.policy_network(torch.tensor(state))

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

    def get_batch(self):
        states_batch, actions_batch, rewards_batch, next_states_batch = [], [], [], []
        shortest_game = np.inf
        for datapoint in range(self.batch_size):
            states, actions, rewards, next_states = self.get_trajectory()
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            if len(states) < shortest_game:
                shortest_game = len(states)

        # Truncating length of each game to length of shortest
        states_batch = torch.tensor([state[:shortest_game] for state in states_batch])
        actions_batch = torch.tensor([action[:shortest_game] for action in actions_batch])
        rewards_batch = torch.tensor([reward[:shortest_game] for reward in rewards_batch])
        next_states_batch = torch.tensor([next_state[:shortest_game] for next_state in next_states_batch])

        return states_batch, actions_batch, rewards_batch, next_states_batch
