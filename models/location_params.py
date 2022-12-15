import torch

class LocationParams(torch.nn.Module):
    def __init__(self, n_channels, size) -> None:
        super().__init__()

        if n_channels > 0:
            self.params = torch.nn.Parameter(torch.zeros(n_channels, size, size))
        else:
            self.params = torch.nn.Parameter()

    def forward(self, cond):
        batch_size = cond.shape[0]
        cond = torch.cat([cond, self.params.broadcast_to((batch_size, *self.params.shape))], dim=1)
        return cond
