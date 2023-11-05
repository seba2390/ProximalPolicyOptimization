import gym
import torch
from tqdm import tqdm

from src.ValueNetwork import ValueNetwork
from src.PolicyNetwork import PolicyNetwork


class PPOAgent:
    def __init__(self,
                 env: gym.Env,
                 state_space_size: int,
                 action_space_size: int,
                 gamma: float,
                 lmbda: float,
                 epsilon: float,
                 smooting_const: float,
                 normalize_advantages: bool = True,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu'):

        self.env = env

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.smoothing_constant = smooting_const
        self.normalize_advantages = normalize_advantages

        self.dtype = dtype
        self.device = device

        self.value_net = ValueNetwork(state_space_size=self.state_space_size,
                                      dtype=self.dtype,
                                      device=self.device)

        self.policy_net = PolicyNetwork(state_space_size=self.state_space_size,
                                        action_space_size=self.action_space_size,
                                        dtype=self.dtype,
                                        device=self.device)

        self.policy_net_OLD = PolicyNetwork(state_space_size=self.state_space_size,
                                            action_space_size=self.action_space_size,
                                            dtype=self.dtype,
                                            device=self.device)
        # Initialize to same weights as policy net
        self.policy_net_OLD.load_state_dict(self.policy_net.state_dict())

    def get_normalized_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Normalize the advantages by subtracting the mean and dividing by the standard deviation.

        The formula for normalization is given by:

            .. math:: A_{\\text{normalized}} = \\frac{A - A_{\\text{mean}}}{A_{\\text{std}} + \\text{smoothing const.}}

        Parameters:

        - advantages (torch.Tensor): A tensor of advantages to be normalized.
        - smoothing_constant (float, optional): A small value added for numerical stability. Default is 1e-10.

        Returns:

        - torch.Tensor: A tensor of normalized advantages.
        """

        normalized_advantages = (advantages - advantages.mean()) / (torch.std(advantages) + self.smoothing_constant)
        return normalized_advantages

    def compute_GAE(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE) for a given sequence of deltas and optionally normalize the advantages.

        GAE provides a bias-variance tradeoff for estimating the advantage function. It computes
        the advantage by taking a weighted average of n-step advantage estimators:

            .. math:: A_t = \\delta_t + (\\gamma \\lambda) \\delta_{t+1} + (\\gamma \\lambda)^2 \\delta_{t+2} + \\dots

        Where:

        - :math:`\\delta_t` is the temporal difference (TD) residual at time t.
        - :math:`\\gamma` is the discount factor.
        - :math:`\\lambda` is a hyperparameter that determines the weighting of future TD residuals.

        If normalization is requested, the advantages are normalized using:

            .. math:: A_{\\text{normalized}} = \\frac{A - \\overline{A}}{\\text{std}(A) + \\text{smoothing\_constant}}

        Parameters:
        - deltas (torch.Tensor): A sequence of TD residuals.
        - gamma (float): Discount factor, typically in the range [0, 1].
        - lambda_ (float): GAE hyperparameter, typically in the range [0, 1].
        - normalize_advantages (bool, optional): If set to True, the advantages are normalized. Default is True.
        - smoothing_constant (float, optional): A small value added for numerical stability during normalization. Default is 1e-10.

        Returns:
        - torch.Tensor: A tensor of computed (and possibly normalized) GAE advantages for each delta.
        """

        advantages = torch.zeros_like(deltas)
        advantage = 0.0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.lmbda * advantage
            advantages[t] = advantage

        if self.normalize_advantages:
            return self.get_normalized_advantages(advantages=advantages)
        return advantages

    def compute_TD_residual(self, reward_t: float, next_value_t: float, value_t: float) -> float:
        """
        Compute the Temporal Difference (TD) residual for a given time step.

        The TD residual (  :math:`\delta_t` ) is given by:

            .. math:: \\delta_t = r_t + \\gamma \cdot V(s_{t+1}) - V(s_t)

        where:

        - :math:`r_t` is the reward at time :math:`t` .
        - :math:`\\gamma` is the discount factor, typically in the range [0, 1] .
        - :math:`V(s_t)` is the estimated value of state :math:`s_t` .
        - :math:`V(s_{t+1})` is the estimated value of state :math:`s_{t+1}` .

        Parameters:

        - reward_t (float): Reward at time \( t \).
        - gamma (float): Discount factor, typically in the range [0, 1].
        - next_value_t (float): Estimated value of state at time \( t+1 \).
        - value_t (float): Estimated value of state at time \( t \).

        Returns:

        - float: The computed TD residual for the given time step.
        """

        return reward_t + self.gamma * next_value_t - value_t

    def get_policy_loss(self, state: torch.Tensor, action: int, advantage: float):
        """
        Compute the Proximal Policy Optimization (PPO) clipped objective loss for a given state, action, and advantage.

        The PPO-Clip loss is defined as:

        .. math::
            L^{\\text{CLIP}}(\\theta) = -\\mathbb{E}[\\min(r(\\theta) \\cdot A_t, \\text{clip}(r(\\theta), 1 - \\epsilon, 1 + \\epsilon) \\cdot A_t)]

        Where:

        - :math:`r(\\theta)` is the ratio of the probability of taking an action under the current policy to the probability under the old policy.
        - :math:`A_t` is the advantage at time :math:`t`.
        - :math:`\\epsilon` is a hyperparameter to clip the ratio.

        Parameters:

        - state (torch.Tensor): The state for which the policy loss is to be computed.
        - action (int): The action taken by the agent.
        - advantage (float): The computed advantage for the given state-action pair.
        - epsilon (float): The hyperparameter for the PPO clipping.

        Returns:

        - float: The computed PPO-Clip loss for the given inputs.
        """
        # Compute the PPO-Clip loss

        # Compute the probability of the action taken under the old policy
        action_probs_old = self.policy_net_OLD(state)
        pi_old = action_probs_old[action]

        # Compute the probability of the action taken under the current policy
        action_probs_new = self.policy_net(state)
        pi_new = action_probs_new[action]

        # Compute the ratio r(θ)
        r = pi_new / pi_old

        # Compute the clipped surrogate objective
        surrogate_obj = r * advantage
        clipped_obj = torch.clamp(r, 1 - self.epsilon, 1 + self.epsilon) * advantage

        # Compute the PPO-Clip loss
        loss = -torch.min(surrogate_obj, clipped_obj).mean()
        return loss

    def get_value_loss(self, state: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, is_last_step: bool):
        """
        Compute the value loss for a given state using Temporal Difference (TD) learning.

        The value loss is calculated using the squared difference between the estimated value of the current state
        and a target value. The target value is computed as:

            .. math:: \\text{target value} = r_t + \\gamma \cdot V(s_{t+1})

        Where:

        - :math:`r_t` is the reward at time :math:`t`.
        - :math:`\\gamma` is the discount factor, typically in the range [0, 1].
        - :math:`V(s_{t+1})` is the estimated value of state :math:`s_{t+1}`.

        Parameters:

        - state (torch.Tensor): The current state.
        - next_state (torch.Tensor): The next state.
        - reward (torch.Tensor): The reward at the current time step.
        - gamma (float): Discount factor, typically in the range [0, 1].
        - is_last_step (bool): Boolean indicating whether the current step is the last in the episode.

        Returns:
        - torch.Tensor: The computed value loss for the given state.
        """

        # Compute target value
        if is_last_step:  # If it's the last step in the episode
            target_value = reward
        else:
            # We detach the value estimate of the next state to prevent it from being
            # updated during the gradient descent of the current state's value.
            # This is done to treat the next state's value estimate as a constant target.
            target_value = reward + self.gamma * self.value_net(next_state).detach()

        # Compute estimated value
        value_estimate = self.value_net(state)

        # Compute the value loss
        value_loss = torch.nn.functional.mse_loss(value_estimate, target_value)

        return value_loss

    def train(self, episodes: int, policy_lr: float, value_lr: float, num_policy_epochs: int, num_value_epochs: int):

        # Define the optimizer for the policy network
        policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Define the optimizer for the value network
        value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)

        accumulated_reward = []

        for episode in tqdm(range(episodes)):
            # Retrieving initial state
            state, info = self.env.reset()

            # For storing trajectory
            states, actions, rewards, next_states = [], [], [], []

            done = False
            while not done:
                # Retrieve current action prob. distribution
                action_probs = self.policy_net(torch.tensor(state))

                # Sample action from distribution
                action = torch.multinomial(input=action_probs, num_samples=1)

                # interact with env.
                next_state, reward, done, _, _ = self.env.step(action.item())

                # Store trajectory data
                states.append(state.tolist())
                actions.append([action.item()])
                rewards.append([reward])
                next_states.append(next_state.tolist())

                # Update current state
                state = next_state

            # Making trajectory tensors to prepare for forward-pass in torch NN.
            states, actions, rewards, next_states = torch.tensor(states), torch.tensor(actions), torch.tensor(
                rewards), torch.tensor(next_states)

            # Saving game length in variable
            game_length = states.shape[0]

            # Iterate backwards through the trajectory to compute deltas and advantages
            deltas = torch.zeros(size=(game_length,))
            for t in range(game_length):
                # Retrieve data for current time step
                state_t, next_state_t, reward_t = states[t], next_states[t], rewards[t]

                # Compute value estimates
                value_t = self.value_net(state_t)
                if t == game_length - 1:  # If it's the last step in the episode
                    next_value_t = torch.tensor([[0.0]])  # The value is 0 at the end of the episode
                else:
                    next_value_t = self.value_net(next_state_t)

                # Compute the TD residual (delta)
                deltas[t] = self.compute_TD_residual(reward_t=reward_t, next_value_t=next_value_t, value_t=value_t)

            advantages = self.compute_GAE(deltas=deltas)

            # Store the old policy parameters (before update)
            self.policy_net_OLD.load_state_dict(self.policy_net.state_dict())

            # For a fixed number of policy update epochs:
            for policy_epoch in range(num_policy_epochs):

                # TODO: Shuffle your data if needed (e.g., if you use mini-batches)

                for t in range(game_length):
                    # Retrieve t'th step of trajectory
                    state_t, action_t, advantage_t = states[t].detach(), actions[t].detach(), advantages[t].detach()

                    # Compute the policy loss
                    policy_loss = self.get_policy_loss(state=state_t, action=action_t, advantage=advantage_t)

                    # Update policy parameters using the optimizer
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

            # Step 4: Value Network Update
            for value_epoch in range(num_value_epochs):

                # TODO: Shuffle your data if needed (e.g., if you use mini-batches)

                value_losses = []  # To store value losses for debugging/analysis

                is_last_step = False
                for t in range(game_length):
                    # Retrieve t'th step of trajectory
                    state_t, next_state_t, reward_t = states[t], next_states[t], rewards[t]
                    if t == game_length - 1:
                        is_last_step = True
                    # Compute value loss
                    value_loss = self.get_value_loss(state=state_t,
                                                     next_state=next_state_t,
                                                     reward=reward_t,
                                                     is_last_step=is_last_step)

                    # Update value network parameters using the optimizer
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()
            accumulated_reward.append(float(torch.sum(rewards).detach().numpy()))
        return accumulated_reward

    def play(self, render=True):
        # Reset environment and get the initial state
        state, info = self.env.reset()

        done = False
        game_length = 0

        while not done:
            # Render the game if render is True
            if render:
                self.env.render()

            # Convert state to tensor for policy network
            state_tensor = torch.tensor(state, dtype=self.dtype).unsqueeze(0)

            # Get action probabilities from the policy network
            action_probs = self.policy_net(state_tensor)

            # Select the action with the highest probability
            action = torch.argmax(action_probs, dim=1)

            # Take the action in the environment
            state, reward, done, _, _ = self.env.step(action.item())

            game_length += 1

        # Close the rendering window
        self.env.close()
        print("#" * 36)
        print(f"# --- Survived for: {game_length} episodes --- #")
        print("#" * 36)
