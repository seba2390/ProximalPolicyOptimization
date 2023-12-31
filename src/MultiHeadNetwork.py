import torch


class PolicyNetworkHead(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 action_space_size: int,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 device: str = 'cpu') -> None:
        super(PolicyNetworkHead, self).__init__()

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.dtype = dtype
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)

        self.lin_layer_1 = torch.nn.Linear(in_features=self.state_space_size,
                                           out_features=self.action_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_1 = torch.nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.activtion_1(self.lin_layer_1(state))
        return out


class ValueNetworkHead(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 device: str = 'cpu') -> None:
        super(ValueNetworkHead, self).__init__()

        self.state_space_size = state_space_size
        self.dtype = dtype
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)

        self.lin_layer_1 = torch.nn.Linear(in_features=state_space_size,
                                           out_features=1,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.lin_layer_1(state)
        return out


class MultiHeadNetwork(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 action_space_size: int,
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0,
                 device: str = 'cpu') -> None:
        super(MultiHeadNetwork, self).__init__()

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.dtype = dtype
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)

        # Backbone layers
        self.lin_layer_1 = torch.nn.Linear(in_features=self.state_space_size,
                                           out_features=2 * self.state_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_1 = torch.nn.ReLU()

        self.lin_layer_2 = torch.nn.Linear(in_features=self.lin_layer_1.out_features,
                                           out_features=self.state_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_2 = torch.nn.ReLU()

        # Heads
        self.value_network = ValueNetworkHead(state_space_size=self.state_space_size,
                                              dtype=self.dtype,
                                              seed=self.seed,
                                              device=self.device)
        self.policy_network = PolicyNetworkHead(state_space_size=self.state_space_size,
                                                action_space_size=self.action_space_size,
                                                dtype=self.dtype,
                                                seed=self.seed,
                                                device=self.device)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.activtion_1(self.lin_layer_1(state))
        out = self.activtion_2(self.lin_layer_2(out))
        return self.value_network.forward(out), self.policy_network.forward(out)
