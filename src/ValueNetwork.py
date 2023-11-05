import torch


class ValueNetwork(torch.nn.Module):
    def __init__(self,
                 state_space_size: int,
                 dtype: torch.dtype = torch.float32,
                 device: str = 'cpu') -> None:
        super(ValueNetwork, self).__init__()

        self.state_space_size = state_space_size
        self.dtype = dtype
        self.device = device

        self.lin_layer_1 = torch.nn.Linear(in_features=self.state_space_size,
                                           out_features=self.state_space_size,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)
        self.activtion_1 = torch.nn.ReLU()

        self.lin_layer_2 = torch.nn.Linear(in_features=self.lin_layer_1.out_features,
                                           out_features=1,
                                           bias=True,
                                           dtype=self.dtype,
                                           device=self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.activtion_1(self.lin_layer_1(state))
        out = self.lin_layer_2(out)
        return out
