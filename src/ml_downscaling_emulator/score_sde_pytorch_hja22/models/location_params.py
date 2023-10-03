import torch

class LocationParams(torch.nn.Module):
    def __init__(self, n_channels, size, size_y=None) -> None:
        """
        If shape of data is square, only use `size` as a parameter.
        If data is not square, use `size` and `size_y`.
        """
        super().__init__()

        if size_y is None:
            size_y = size

        self.params = torch.nn.Parameter(torch.zeros(n_channels, size, size_y))

    def forward(self, cond):
        batch_size = cond.shape[0]
        cond = torch.cat([cond, self.params.broadcast_to((batch_size, *self.params.shape))], dim=1)
        return cond
