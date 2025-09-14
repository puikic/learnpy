import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attention_value = torch.matmul(
            Q, K.transpose(-1, -2)
        )
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim),
            -1
        )
        output = torch.matmul(
            attention_weight, V
        )
        return output

X = torch.rand(4,3,5)
n = SelfAttention(5)
print(n(X))
