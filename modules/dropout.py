import torch
from torch import nn

from .common import ScalarVector


class VectorDropout(nn.Module):
    
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy = nn.Parameter(torch.ones(1))

    def forward(self, x):
        '''
        Args:
            x:  (*, vec_dim)
        '''
        device = x.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class SVDropout(nn.Module):

    def __init__(self, drop_rate):
        super().__init__()
        self.s_dropout = nn.Dropout(drop_rate)
        self.v_dropout = nn.Dropout(drop_rate)
        self.dummy = nn.Parameter(torch.ones(1))

    @property
    def device(self):
        """Get device from dummy parameter"""
        return self.dummy.device

    def forward(self, x: ScalarVector) -> ScalarVector:
        return ScalarVector(
            s = self.s_dropout(x.s),
            v = self.v_dropout(x.v),
        )


