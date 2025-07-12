import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinePosEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Args:
        dim - hidden layer number of features.
        length - time window length.
    """
    def __init__(self,
                 dim: int,
                 length: int):
        super().__init__()
        pos = torch.arange(0, length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(1, dim, length)
        pe[:, 0::2, :] = torch.sin(pos * div_term).transpose(0, 1).unsqueeze(0)
        pe[:, 1::2, :] = torch.cos(pos * div_term).transpose(0, 1).unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :, :x.size(2)]


class LearnableEmbedding(nn.Module):
    """
    Learnable positional encoding.
    Args:
        dim - hidden layer number of features.
        length - time window length.
    """
    def __init__(self,
                 dim: int,
                 length: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, dim, length))
        nn.init.normal_(self.pe, 0.0, 0.02)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :, :x.size(2)]


class Sketching(nn.Module):
    """
    Matrix approximation through a sketching technique. Inspired with Polysketchformer and TensorSketch methods.
    Args:
        dim - hidden layer number of features.
        r - sketched space dimension.
        p - kernel's power.
    """
    def __init__(self,
                 dim: int,
                 r: int,
                 p: int):
        super().__init__()
        assert p >= 2 and p % 2 == 0, "Only p = 1, 2 and 4 are supported for now"
        self.p = p
        self.r = r
        self.G1 = nn.Parameter(torch.randn(dim, r))
        self.G2 = nn.Parameter(torch.randn(dim, r))

        nn.init.normal_(self.G1, mean=0.0, std=1.0 / r ** 0.5)
        nn.init.normal_(self.G2, mean=0.0, std=1.0 / r ** 0.5)

    def sketchwithnegativity(self, x: torch.Tensor, p: int):
        if p == 1:
            return x
        else:
            M1 = self.sketchwithnegativity(x, p // 2)
            M2 = self.sketchwithnegativity(x, p // 2)
            P1 = torch.matmul(M1, self.G1)
            P2 = torch.matmul(M2, self.G2)
            return (1.0 / self.r ** 0.5) * (P1 * P2)

    def sketch(self, x: torch.Tensor):
        if self.p == 2:
            P1 = torch.matmul(x, self.G1)
            P2 = torch.matmul(x, self.G2)
            return (1.0 / self.r ** 0.5) * (P1 * P2)
        else:
            half = self.sketchwithnegativity(x, self.p // 2)
            return half * half

    def forward(self, x: torch.Tensor):
        return self.sketch(x)


class Conv1DMHSA(nn.Module):
    """
    Multi-head self-attention with convolutions.
    Args:
        k_size - kernel size.
        dim - hidden layer number of features.
        num_heads - number of heads.
        r - sketched space dimension.
        p - kernel's power.
        func_type - function type for attention weights.
        gamma - a value for a learnable parameter initialization.
    """
    def __init__(self,
                 k_size: int,
                 dim: int,
                 num_heads: int,
                 r: int,
                 p: int,
                 func_type: str,
                 gamma: float):
        super().__init__()
        assert func_type in ['softmax', 'sketch'], "Unsupported function type"
        self.dim = dim
        self.func_type = func_type
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        padding = (k_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

        self.q = nn.Conv1d(dim, dim, k_size, padding=0, bias=True)
        self.k = nn.Conv1d(dim, dim, k_size, padding=0, bias=True)
        self.v = nn.Conv1d(dim, dim, 1, padding=0, bias=False)

        self.projection = nn.Linear(dim, dim)

        self.gamma_q = nn.Parameter(torch.ones(1) * gamma)
        self.beta_q = nn.Parameter(torch.zeros(1))
        self.gamma_k = nn.Parameter(torch.ones(1) * gamma)
        self.beta_k = nn.Parameter(torch.zeros(1))

        self.q_sketch = Sketching(self.head_dim, r, p)
        self.k_sketch = Sketching(self.head_dim, r, p)

    def forward(self, x: torch.Tensor):
        batch, dim, length = x.shape
        x_pad = self.padding_opertor(x)
        q, k, v = self.q(x_pad), self.k(x_pad), self.v(x)

        def shape(tensor):
            return tensor.view(batch, self.num_heads, self.head_dim, length).transpose(2, 3)

        q, k, v = shape(q), shape(k), shape(v)

        if self.func_type == 'softmax':
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            x = torch.matmul(attn, v)

        elif self.func_type == 'sketch':
            q = self.gamma_q * q + self.beta_q
            k = self.gamma_k * k + self.beta_k
            phi_q = self.q_sketch(q)
            phi_k = self.k_sketch(k)
            scores = torch.matmul(phi_q, phi_k.transpose(-2, -1))
            attn = scores / ((scores.sum(dim=-1, keepdim=True)) + 1e-6)
            x = torch.matmul(attn, v)

        x = x.transpose(2, 3).contiguous().view(batch, self.dim, length).transpose(1, 2)
        return self.projection(x)


class TransformerEncoder(nn.Module):
    """
    An encoder-transformer block with MHSA and FFN.
    Args:
        dim - hidden layer number of features.
        k_size - kernel size.
        num_heads - number of heads.
        r - sketched space dimension.
        p - kernel's power.
        func_type - function type for attention weights.
        gamma - a value for a learnable parameter initialization.
    """
    def __init__(self,
                 dim: int,
                 k_size: int,
                 num_heads: int,
                 r: int,
                 p: int,
                 func_type: str,
                 gamma: float):
        super().__init__()
        self.Conv1DMHSA = Conv1DMHSA(k_size, dim, num_heads, r, p, func_type, gamma)
        self.FFN = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # x = (B, D, L)
        y = self.Conv1DMHSA(x)  # (B, D, L)
        x = x.transpose(1, 2) + y  # (B, L, D)
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (B, D, L)
        y = self.FFN(x.transpose(1, 2))  # (B, L, D)
        x = x.transpose(1, 2) + y  # (B, L, D)
        x = self.norm2(x)
        return x.transpose(1, 2)  # (B, D, L)


class MHSA(nn.Module):
    """
    Multi-head self-attention.
    Args:
        k_size - kernel size.
        dim - hidden layer number of features.
        num_heads - number of heads.
        r - sketched space dimension.
        p - kernel's power.
        func_type - function type for attention weights.
        gamma - a value for a learnable parameter initialization.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 r: int,
                 p: int,
                 func_type: str,
                 gamma: float):
        super().__init__()
        assert func_type in ['softmax', 'sketch'], "Unsupported function type"
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.out_dim = dim
        self.func_type = func_type

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.projection = nn.Linear(dim, dim)

        self.gamma_q = nn.Parameter(torch.ones(1) * gamma)
        self.beta_q = nn.Parameter(torch.zeros(1))
        self.gamma_k = nn.Parameter(torch.ones(1) * gamma)
        self.beta_k = nn.Parameter(torch.zeros(1))

        self.q_sketch = Sketching(self.head_dim, r, p)
        self.k_sketch = Sketching(self.head_dim, r, p)

    def forward(self, x: torch.Tensor):
        batch, patch, dim = x.size()
        q = self.q(x).view(batch, patch, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(batch, patch, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(batch, patch, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.func_type == 'softmax':
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=-1)
            x = torch.matmul(attn, v)

        elif self.func_type == 'sketch':
            q = self.gamma_q * q + self.beta_q
            k = self.gamma_k * k + self.beta_k
            phi_q = self.q_sketch(q)
            phi_k = self.k_sketch(k)
            scores = torch.matmul(phi_q, phi_k.transpose(-2, -1))
            attn = scores / ((scores.sum(dim=-1, keepdim=True)) + 1e-6)
            x = torch.matmul(attn, v)

        x = x.permute(0, 2, 1, 3).contiguous().view(batch, patch, dim)
        return self.projection(x)


class TransformerEncoderLinear(nn.Module):
    """
    An encoder-transformer block with MHSA and FFN.
    Args:
        dim - hidden layer number of features.
        k_size - kernel size.
        num_heads - number of heads.
        r - sketched space dimension.
        p - kernel's power.
        func_type - function type for attention weights.
        gamma - a value for a learnable parameter initialization.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 r: int,
                 p: int,
                 func_type: str,
                 gamma: float):
        super().__init__()
        self.MHSA = MHSA(dim, num_heads, r, p, func_type, gamma)
        self.FFN = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # x = (B, P, D)
        y = self.MHSA(x)
        x = self.norm1(x + y)
        y = self.FFN(x)
        x = self.norm2(x + y)
        return x

