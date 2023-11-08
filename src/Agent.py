import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np

from src.ValueNetwork import ValueNetwork
from src.PolicyNetwork import PolicyNetwork
from src.Trajectories import Trajectories
from src.MultiHeadNetwork import MultiHeadNetwork
from src.Utils import random_shuffle
from src.OptimizerParameters import AdamOptimizerParameters


class PPOAgent:
    def __init__(self,
                 env: gym.Env,
                 state_space_size: int,
                 action_space_size: int,
                 batch_size: int,
                 gamma: float,
                 lmbda: float,
                 epsilon: float,
                 smooting_const: float,
                 shuffle_batches: bool,
                 normalize_advantages: bool = True,
                 architecture: str = 'Individual Networks',
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 device: str = 'cpu'):

        self.env = env

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.smoothing_constant = smooting_const
        self.normalize_advantages = normalize_advantages
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches

        self.architecture = architecture
        self.dtype = dtype
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)

        __defined_architectures__ = ['Individual Networks', 'Multi Head Network']
        if self.architecture not in __defined_architectures__:
            raise ValueError(f'Architecture must be one of: {__defined_architectures__}.')

        if self.architecture == 'Individual Networks':

            self.value_net = ValueNetwork(state_space_size=self.state_space_size,
                                          dtype=self.dtype,
                                          seed=self.seed,
                                          device=self.device)

            self.policy_net = PolicyNetwork(state_space_size=self.state_space_size,
                                            action_space_size=self.action_space_size,
                                            dtype=self.dtype,
                                            seed=self.seed,
                                            device=self.device)

            self.policy_net_OLD = PolicyNetwork(state_space_size=self.state_space_size,
                                                action_space_size=self.action_space_size,
                                                dtype=self.dtype,
                                                seed=self.seed,
                                                device=self.device)
            # Initialize to same weights as policy net
            self.policy_net_OLD.load_state_dict(self.policy_net.state_dict())

        elif architecture == 'Multi Head Network':
            self.multihead_net = MultiHeadNetwork(state_space_size=self.state_space_size,
                                                  action_space_size=self.action_space_size,
                                                  dtype=self.dtype,
                                                  seed=self.seed,
                                                  device=self.device)
            self.multihead_net_OLD = MultiHeadNetwork(state_space_size=self.state_space_size,
                                                      action_space_size=self.action_space_size,
                                                      dtype=self.dtype,
                                                      seed=self.seed,
                                                      device=self.device)
            # Initialize to same weights as multihead net
            self.multihead_net_OLD.load_state_dict(self.multihead_net.state_dict())

    def get_normalized_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        normalized_advantages = (advantages - advantages.mean()) / (torch.std(advantages) + self.smoothing_constant)
        return normalized_advantages

    def compute_GAE(self, deltas: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(deltas)
        advantage = 0.0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.lmbda * advantage
            advantages[t] = advantage

        if self.normalize_advantages:
            return self.get_normalized_advantages(advantages=advantages)
        return advantages

    def compute_TD_residual(self, rewards: torch.tensor, next_values: torch.tensor,
                            values: torch.tensor) -> torch.tensor:

        return rewards + self.gamma * next_values - values

    def get_policy_loss(self, states: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor):

        if self.architecture == "Individual Networks":
            # Compute the probability of the action taken under the old policy
            action_probs_old = self.policy_net_OLD(states)
            pi_old = torch.gather(input=action_probs_old, dim=1, index=actions.unsqueeze(1))

            # Compute the probability of the action taken under the current policy
            action_probs_new = self.policy_net(states)
            pi_new = torch.gather(input=action_probs_new, dim=1, index=actions.unsqueeze(1))

            # Compute the ratio r(θ)
            r = (pi_new / pi_old).flatten()

            # Compute the clipped surrogate objective
            surrogate_obj = r * advantages

            clipped_obj = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Compute the policy loss
            policy_loss = -torch.min(surrogate_obj, clipped_obj).mean()
            return policy_loss

        elif self.architecture == "Multi Head Network":
            # Compute the probability of the action taken under the old policy
            _, action_probs_old = self.multihead_net_OLD(states)
            pi_old = torch.gather(input=action_probs_old, dim=1, index=actions.unsqueeze(1))

            # Compute the probability of the action taken under the current policy
            _, action_probs_new = self.multihead_net(states)
            pi_new = torch.gather(input=action_probs_new, dim=1, index=actions.unsqueeze(1))

            # Compute the ratio r(θ)
            r = (pi_new / pi_old).flatten()

            # Compute the clipped surrogate objective
            surrogate_obj = r * advantages

            clipped_obj = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # Compute the policy loss
            policy_loss = -torch.min(surrogate_obj, clipped_obj).mean()
            return policy_loss

    def get_value_loss(self, states: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor):

        if self.architecture == "Individual Networks":
            # Compute target value (for last step - set to reward)
            target_values = rewards + self.gamma * self.value_net(next_states).flatten().detach()
            target_values = torch.cat((target_values[:-1], torch.tensor([rewards[-1].item()])))

            # Compute estimated value
            estimated_values = self.value_net(states).flatten()

            # Compute the value loss
            value_loss = torch.nn.functional.mse_loss(estimated_values, target_values)
            return value_loss

        elif self.architecture == "Multi Head Network":
            # Compute target value (for last step - set to reward)
            values, _ = self.multihead_net(next_states)
            target_values = rewards + self.gamma * values.flatten().detach()
            target_values = torch.cat((target_values[:-1], torch.tensor([rewards[-1].item()])))

            # Compute estimated value
            estimated_values, _ = self.multihead_net(states)

            # Compute the value loss
            value_loss = torch.nn.functional.mse_loss(estimated_values.flatten(), target_values)
            return value_loss

    def train(self,
              episodes: int,
              policy_optimizer_params: AdamOptimizerParameters = None,
              value_optimizer_params: AdamOptimizerParameters = None,
              multihead_optimizer_params: AdamOptimizerParameters = None,
              num_policy_epochs: int = None,
              num_value_epochs: int = None,
              num_multihead_epochs: int = None):

        avg_accumulated_reward = []

        if self.architecture == 'Individual Networks':

            avg_value_net_loss, avg_policy_net_loss = [], []

            if num_policy_epochs is None or num_value_epochs is None or policy_optimizer_params is None or value_optimizer_params is None:
                raise ValueError("If architecture is Individual Networks, all the following params must be provided"
                                 " [num_policy_epochs, num_value_epochs, policy_optimizer_params, value_optimizer_params].")

            # Define the optimizer for the policy network
            policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(),
                                                lr=policy_optimizer_params.lr,
                                                betas=policy_optimizer_params.betas,
                                                weight_decay=policy_optimizer_params.weight_decay)

            # Define the optimizer for the value network
            value_optimizer = torch.optim.Adam(params=self.value_net.parameters(),
                                               lr=value_optimizer_params.lr,
                                               betas=value_optimizer_params.betas,
                                               weight_decay=value_optimizer_params.weight_decay)

            for episode in tqdm(range(episodes)):

                # Retrieving batch of trajectories
                trajectories = Trajectories(batch_size=self.batch_size,
                                            env=self.env,
                                            policy_network=self.policy_net,
                                            multihead_network=None,
                                            architecture=self.architecture,
                                            seed=episode * self.batch_size + 1)

                # States_batch has dims:      (batch size , game length , state space size)
                # next_states_batch has dims: (batch size , game length , state space size)
                # actions_batch has dims:     (batch size , game length)
                # rewards_batch has dims:     (batch size , game length)
                states_batch, actions_batch, rewards_batch, next_states_batch = trajectories.get_batch()
                avg_accumulated_reward.append(np.mean([torch.sum(rewards).detach().item() for rewards in rewards_batch]))

                # Computing advantages
                advantages_batch = []
                for trajectory in range(self.batch_size):
                    states = states_batch[trajectory]
                    rewards = rewards_batch[trajectory]
                    next_states = next_states_batch[trajectory]

                    values = self.value_net(states).flatten()
                    # The value is 0 at the end of the trajectory (hence the concatenation of 0 at the end)
                    next_values = torch.cat((self.value_net(next_states)[:-1].flatten(), torch.tensor([0])))
                    deltas = self.compute_TD_residual(rewards=rewards, next_values=next_values, values=values)
                    advantages_batch.append(self.compute_GAE(deltas=deltas))

                # Store the old policy parameters (before update)
                self.policy_net_OLD.load_state_dict(self.policy_net.state_dict())

                # Policy Network Update:
                __current_policy_loss__ = []
                for policy_epoch in range(num_policy_epochs):
                    for trajectory in range(self.batch_size):
                        states = states_batch[trajectory].detach()
                        actions = actions_batch[trajectory].detach()
                        advantages = advantages_batch[trajectory].detach()

                        # Compute the policy loss
                        policy_loss = self.get_policy_loss(states=states, actions=actions, advantages=advantages)

                        # Update policy parameters using the optimizer
                        policy_optimizer.zero_grad()
                        policy_loss.backward()
                        policy_optimizer.step()
                        __current_policy_loss__.append(policy_loss.detach().item())

                    # Shuffle batch
                    if self.shuffle_batches:
                        states_batch, actions_batch, rewards_batch, next_states_batch, advantages_batch = random_shuffle(
                            states_batch=states_batch,
                            actions_batch=actions_batch,
                            rewards_batch=rewards_batch,
                            next_states_batch=next_states_batch,
                            advantages_batch=advantages_batch,
                            seed=self.seed+episode*num_multihead_epochs+policy_epoch+1)
                avg_policy_net_loss.append(np.mean(__current_policy_loss__))

                # Value Network Update
                __current_value_loss__ = []
                for value_epoch in range(num_value_epochs):
                    for trajectory in range(self.batch_size):
                        states = states_batch[trajectory]
                        rewards = rewards_batch[trajectory]
                        next_states = next_states_batch[trajectory]

                        # Compute the value loss
                        value_loss = self.get_value_loss(states=states, next_states=next_states, rewards=rewards)

                        # Update value network parameters using the optimizer
                        value_optimizer.zero_grad()
                        value_loss.backward()
                        value_optimizer.step()
                        __current_value_loss__.append(value_loss.detach().item())

                    # Shuffle batch
                    if self.shuffle_batches:
                        states_batch, actions_batch, rewards_batch, next_states_batch, advantages_batch = random_shuffle(
                            states_batch=states_batch,
                            actions_batch=actions_batch,
                            rewards_batch=rewards_batch,
                            next_states_batch=next_states_batch,
                            advantages_batch=advantages_batch,
                            seed=self.seed+episode*num_multihead_epochs+value_epoch+1)
                avg_value_net_loss.append(np.mean(__current_value_loss__))

            return avg_accumulated_reward, avg_value_net_loss, avg_policy_net_loss

        elif self.architecture == 'Multi Head Network':
            if num_multihead_epochs is None or multihead_optimizer_params is None:
                raise ValueError("If architecture is Multi Head Network, all the following params must be provided"
                                 " [num_multihead_epochs, multihead_optimizer_params].")

            avg_multihead_net_loss = []

            # Define the optimizer for the multi head network
            optimizer = torch.optim.Adam(params=self.multihead_net.parameters(),
                                         lr=multihead_optimizer_params.lr,
                                         betas=multihead_optimizer_params.betas,
                                         weight_decay=multihead_optimizer_params.weight_decay)

            for episode in tqdm(range(episodes)):
                # Retrieving batch of trajectories
                trajectories = Trajectories(batch_size=self.batch_size,
                                            env=self.env,
                                            policy_network=None,
                                            multihead_network=self.multihead_net,
                                            architecture=self.architecture,
                                            seed=episode * self.batch_size + 1)

                # States_batch has dims:      (batch size , game length , state space size)
                # next_states_batch has dims: (batch size , game length , state space size)
                # actions_batch has dims:     (batch size , game length)
                # rewards_batch has dims:     (batch size , game length)
                states_batch, actions_batch, rewards_batch, next_states_batch = trajectories.get_batch()
                avg_accumulated_reward.append(np.mean([torch.sum(rewards).detach().item() for rewards in rewards_batch]))
                self.multihead_net_OLD.load_state_dict(self.multihead_net.state_dict())

                # Computing advantages
                advantages_batch = []
                for trajectory in range(self.batch_size):
                    states = states_batch[trajectory]
                    rewards = rewards_batch[trajectory]
                    next_states = next_states_batch[trajectory]

                    values, _ = self.multihead_net(states)
                    next_values, _ = self.multihead_net(next_states)
                    # The value is 0 at the end of the trajectory (hence the concatenation of 0 at the end)
                    next_values = torch.cat((next_values[:-1].flatten(), torch.tensor([0])))
                    deltas = self.compute_TD_residual(rewards=rewards, next_values=next_values, values=values.flatten())
                    advantages_batch.append(self.compute_GAE(deltas=deltas))

                __current_multihead_loss__ = []
                for multihead_epoch in range(num_multihead_epochs):
                    for trajectory in range(self.batch_size):
                        # First updating multihead w. respect to policy head
                        states = states_batch[trajectory].detach()
                        actions = actions_batch[trajectory].detach()
                        advantages = advantages_batch[trajectory].detach()

                        # Compute the policy loss
                        policy_loss = self.get_policy_loss(states=states, actions=actions, advantages=advantages)

                        # Then updating multihead w. respect to value head
                        states = states_batch[trajectory]
                        rewards = rewards_batch[trajectory]
                        next_states = next_states_batch[trajectory]

                        # Compute the value loss
                        value_loss = self.get_value_loss(states=states, next_states=next_states, rewards=rewards)

                        # Update multihead network parameters using the optimizer
                        total_loss = policy_loss + value_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        __current_multihead_loss__.append(total_loss.detach().item())

                    # Shuffle batch
                    if self.shuffle_batches:
                        states_batch, actions_batch, rewards_batch, next_states_batch, advantages_batch = random_shuffle(
                            states_batch=states_batch,
                            actions_batch=actions_batch,
                            rewards_batch=rewards_batch,
                            next_states_batch=next_states_batch,
                            advantages_batch=advantages_batch,
                            seed=self.seed+episode*num_multihead_epochs+multihead_epoch+1)
                avg_multihead_net_loss.append(np.mean(__current_multihead_loss__))

            return avg_accumulated_reward, avg_multihead_net_loss

    def play(self):

        # Reset environment and get the initial state
        state, info = self.env.reset(seed=self.seed)

        done = False
        game_length = 0
        total_reward = 0
        while not done:

            # Convert state to tensor for policy network
            state_tensor = torch.tensor(state, dtype=self.dtype).unsqueeze(0)

            # Get action probabilities from the policy network
            if self.architecture == "Multi Head Network":
                _, action_probs = self.multihead_net_OLD(state_tensor)
            else:
                action_probs = self.policy_net(state_tensor)

            # Select the action with the highest probability
            action = torch.argmax(action_probs, dim=1)

            # Take the action in the environment
            state, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated

            game_length += 1
            total_reward += reward

        # Close the rendering window
        self.env.close()

        statement = f"# --- Survived for: {game_length} episodes, and earned a total reward of: {total_reward} --- #"
        print("#" * len(statement))
        print(statement)
        print("#" * len(statement))
