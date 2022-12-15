import torch

class LocationParams(torch.nn.Module):
    def __init__(self, n_channels, size) -> None:
        super().__init__()

        self.params = torch.nn.Parameter(torch.zeros(n_channels, size, size))

    def forward(self, cond):
        batch_size = cond.shape[0]
        cond = torch.cat([cond, self.params.broadcast_to((batch_size, *self.params.shape))], dim=1)
        return cond
