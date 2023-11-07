import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 action_space_size: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu') -> None:
        super(PolicyNetwork, self).__init__()

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.dtype = dtype
        self.device = device

        self.lin_layer_1 = torch.nn.Linear(in_features=self.state_space_size,
                                           out_features=2 * self.state_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_1 = torch.nn.ReLU()

        self.lin_layer_2 = torch.nn.Linear(in_features=self.lin_layer_1.out_features,
                                           out_features=self.action_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_2 = torch.nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.activtion_1(self.lin_layer_1(state))
        out = self.activtion_2(self.lin_layer_2(out))
        return out