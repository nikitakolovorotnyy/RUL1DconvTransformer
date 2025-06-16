import torch
import torch.nn as nn
from Layers import TransformerEncoder, LearnableEmbedding


class Model(nn.Module):
    """
    A model for RUL tasks.
    Args:
        k_size - kernel size.
        dim - hidden layer number of features.
        r - sketched space dimension.
        p - kernel's power.
        length - time window length.
    """
    def __init__(self,
                 k_size: int,
                 dim: int,
                 r: int,
                 length: int,
                 num_blocks: int
                 ):
        super().__init__()
        self.embedding = nn.Conv1d(24, dim, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2)
        self.learnable_encode = LearnableEmbedding(dim, length)
        self.transformers = nn.ModuleList([
            TransformerEncoder(dim=dim, k_size=k_size, num_heads=8, r=r, p=4, func_type='sketch', gamma=0.1)
            for _ in range(num_blocks)
        ])
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, 1)
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.learnable_encode(x)
        for enc in self.transformers:
            x = enc(x)
        return self.regression(x).squeeze(1)
