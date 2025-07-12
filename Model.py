import torch
import torch.nn as nn
from Layers import TransformerEncoderLinear, LearnableEmbedding, SinePosEncoding


class Model(nn.Module):
    def __init__(self,
                 k_size: int,
                 dim: int,
                 r: int,
                 length: int,
                 num_blocks: int,
                 func_type: str
                 ):
        super().__init__()
        self.embedding = nn.Conv1d(14, dim, kernel_size=k_size, stride=1, padding=(k_size - 1) // 2)

        self.learnable_encode = LearnableEmbedding(dim, length)
        self.posenc = SinePosEncoding(dim, length)
        self.transformers = nn.ModuleList([
            TransformerEncoderLinear(dim=dim, num_heads=8, r=r, p=4, func_type=func_type, gamma=0.1)
            for _ in range(num_blocks)
        ])
        self.regression = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                        nn.Flatten(),
                                        nn.Linear(dim, dim * 2),
                                        nn.GELU(),
                                        nn.Linear(dim * 2, 1))

    '''
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.learnable_encode(x)
        for enc in self.transformers:
            x = enc(x)
        return self.regression(x).squeeze(1)
'''

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.learnable_encode(x)
        x = x.permute(0, 2, 1)

        for enc in self.transformers:
            x = enc(x)

        x = x.permute(0, 2, 1)
        out = self.regression(x)
        return out.squeeze(1)
