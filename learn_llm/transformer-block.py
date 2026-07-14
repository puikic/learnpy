import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm（LLaMA 使用）：相比 LayerNorm 去掉了"减均值"这一步，
    只用均方根（Root Mean Square）做缩放，计算更省、效果相当。
        y = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))  # 可学习缩放，无 bias

    def forward(self, x):
        # 沿最后一维（特征维）算均方根；用 float 提升数值稳定性再转回原 dtype
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络（LLaMA 使用）：用门控机制替代传统 FFN 的 ReLU。
        FFN(x) = W_down( SiLU(W_gate(x)) * W_up(x) )
    有两条上升支路：gate 支路过 SiLU 激活当"门"，up 支路是"内容"，逐元素相乘。
    """
    def __init__(self, hidden_dim, intermediate_dim=None):
        super().__init__()
        # LLaMA 惯例：中间维约 8/3 * hidden，这里简化为 4 倍再取整
        if intermediate_dim is None:
            intermediate_dim = int(hidden_dim * 8 / 3)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SiLU(x) = x * sigmoid(x)，即 F.silu
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """标准多头因果自注意力（decoder-only 用），带下三角 mask 防止看未来。"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.att_drop = nn.Dropout(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 因果 mask：下三角为 True 保留，上三角（未来）置 -inf
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attention_score = attention_score.masked_fill(~causal_mask, float("-inf"))

        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        out = torch.matmul(attention_weight, V)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block（LLaMA 风格）：
        x = x + Attn(RMSNorm(x))     # 先归一化再进子层，输出加回残差
        x = x + FFN(RMSNorm(x))
    Pre-Norm（norm 在残差内部）比 Post-Norm 更易训练、深层更稳定。
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))  # 残差连接 1：注意力子层
        x = x + self.ffn(self.ffn_norm(x))    # 残差连接 2：前馈子层
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, hidden_dim, num_heads = 2, 5, 32, 4

    x = torch.rand(batch_size, seq_len, hidden_dim)

    # 单独测各组件形状
    print("RMSNorm  :", RMSNorm(hidden_dim)(x).shape)
    print("SwiGLU   :", SwiGLU(hidden_dim)(x).shape)
    print("CausalAtt:", CausalSelfAttention(hidden_dim, num_heads)(x).shape)

    # 完整 block；堆叠多层就是一个 decoder-only 主干
    block = TransformerBlock(hidden_dim, num_heads)
    out = block(x)
    print("Block out:", out.shape)  # 期望 (2, 5, 32)
