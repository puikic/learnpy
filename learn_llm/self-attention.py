import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.att_drop = nn.Dropout(0.1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        
        attention_weight = softmax(attention_score, -1)
        attention_weight = self.att_drop(attention_weight)
        output = torch.matmul(
            attention_weight, V
        )
        return output

# X = torch.rand(4,3,5)
# n = SelfAttention(5)
# print(n(X))

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.att_drop = nn.Dropout(0.1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()   
        Q = self.q_proj(x) # (batch, seq_len, hidden_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape -> (batch, seq_len, num_heads, head_dim)
        # transpose -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        #(batch, num_heads, seq_len, seq_len) 
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        
        attention_weight = torch.softmax(attention_score, -1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, V)  # (batch, num_heads, seq_len, head_dim)
        # concat 多头 -> (batch, seq_len, hidden_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attention_output)
        return output

X = torch.rand(4,3,15)
n = MultiHeadAttention(15, 3)
print(n(X))

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.att_drop = nn.Dropout(0.1)
        self.nums_kv_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def repeat_kv(self, x, nums):
        batch_size, num_kv_heads, seq_len, head_dim = x.size() # or x.shape
        if nums == 1:
            return x
        x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, nums, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * nums, seq_len, head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()   
        Q = self.q_proj(x) # (batch, seq_len, hidden_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        K = self.repeat_kv(K, self.nums_kv_groups)
        V = self.repeat_kv(V, self.nums_kv_groups)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)   
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))

        attention_weight = torch.softmax(attention_score, -1)
        attention_weight = self.att_drop(attention_weight)
        attention_output = torch.matmul(attention_weight, V)  # (batch, num_heads, seq_len, head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attention_output)  
        return output