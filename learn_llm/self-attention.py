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
        
        attention_weight = torch.softmax(attention_score, -1)
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


class GroupedQueryAttentionV1(nn.Module):
    def __init__(self, hidden_dim, head_num, kv_head_num):
        super().__init__()
        assert head_num % kv_head_num == 0
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.head_dim = hidden_dim // head_num
        self.group_size = head_num // kv_head_num
        self.att_drop = nn.Dropout(0.1)
        # Q 按全部 head 投影；K/V 只按 kv_head_num 投影，参数量相比 MHA 减少
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, kv_head_num * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, kv_head_num * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        K = self.k_proj(x)  # (batch, seq_len, kv_head_num * head_dim)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.kv_head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.kv_head_num, self.head_dim).transpose(1, 2)

        # 用 repeat_interleave 在头维度复制 K/V: [kv0, kv1] -> [kv0, kv0, kv1, kv1]
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))

        attention_weight = torch.softmax(attention_score, -1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, V)  # (batch, head_num, seq_len, head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attention_output)
        return output


X = torch.rand(4,3,16)
n = GroupedQueryAttentionV1(16, 4, 2)
print(n(X))


class GroupedQueryAttentionV2(nn.Module):
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
        x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, nums, seq_len, head_dim) # 不能改变顺序
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

        context = torch.matmul(attention_weight, V)  # (batch, head_num, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(context)
        return output


X = torch.rand(4,3,16)
n = GroupedQueryAttentionV2(16, 4, 2)
print(n(X))

class MultiHeadAttentionWithKVCache(nn.Module):
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

    def forward(self, x, mask=None, use_kv=False, kv_cache=None):
        # x 的形状: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.size()   
        Q = self.q_proj(x) 
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape -> (batch, seq_len, num_heads, head_dim)
        # transpose -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # ==========================================
        # 🌟 核心修改 1：KV Cache 记忆拼接逻辑
        # ==========================================
        if use_kv:
            if kv_cache is not None:
                past_K, past_V = kv_cache
                # 沿着代表“字数/序列长度”的维度（dim=2）将历史记忆与当前的新词特征进行拼接
                K = torch.cat([past_K, K], dim=2)
                V = torch.cat([past_V, V], dim=2)
            
            # 更新缓存：把拼接好的完整 K 和 V 打包，准备返回给下一步使用
            kv_cache = (K, V)

        # 注意：如果启用了 KV Cache，此时 K 的 seq_len 长度已经膨胀，包含了历史所有的字！
        # attention_score 形状 -> (batch, num_heads, q_seq_len, k_seq_len) 
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 注意：在自回归的 Decode 阶段通常不需要传 mask，因为只有一个词，且只看历史
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        
        attention_weight = torch.softmax(attention_score, -1)
        attention_weight = self.att_drop(attention_weight)

        # V 的长度也已经膨胀，与 attention_weight 完美对应相乘
        attention_output = torch.matmul(attention_weight, V)  # (batch, num_heads, q_seq_len, head_dim)
        
        # concat 多头 -> (batch, seq_len, hidden_dim)
        # 这里的 seq_len 取自最初 x.size() (即 Q 的长度)。它不受 KV 缓存膨胀的影响，逻辑极其安全！
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attention_output)
        
        # ==========================================
        # 🌟 核心修改 2：根据开关决定是否返回更新后的 Cache
        # ==========================================
        if use_kv:
            return output, kv_cache
            
        return output

class GroupedQueryAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads          # Query 的头数 (例如: 32)
        self.num_kv_heads = num_kv_heads    # K 和 V 的头数 (例如: 8)
        
        # 确保 Q 的头数能被 KV 的头数整除
        assert self.num_heads % self.num_kv_heads == 0 
        self.num_groups = self.num_heads // self.num_kv_heads # 每个组里有几个 Q 头共享一个 KV (例如: 4)
        
        self.head_dim = hidden_dim // num_heads
        self.att_drop = nn.Dropout(0.1)
        
        # ==========================================
        # 🌟 核心修改 1：K 和 V 的投影矩阵变小了！
        # ==========================================
        self.q_proj = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        # 注意这里：输出维度不再是 hidden_dim，而是缩减为了 num_kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim)
        
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None, use_kv=False, kv_cache=None):
        batch_size, seq_len, _ = x.size()   
        
        Q = self.q_proj(x) # (batch, seq_len, num_heads * head_dim)
        K = self.k_proj(x) # (batch, seq_len, num_kv_heads * head_dim)
        V = self.v_proj(x) # (batch, seq_len, num_kv_heads * head_dim)

        # reshape & transpose
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 注意：K 和 V 在第 1 维度 (Heads) 上的大小是 num_kv_heads，而不是 num_heads
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ==========================================
        # 🌟 核心修改 2：显存极度压缩的 KV Cache 拼接
        # ==========================================
        if use_kv:
            if kv_cache is not None:
                past_K, past_V = kv_cache
                # 拼接时，我们依然顺着 Seq_len (dim=2) 拼接。
                # 此时显存里存的 K 和 V，头数只有 num_kv_heads，体积大大缩小！
                K = torch.cat([past_K, K], dim=2)
                V = torch.cat([past_V, V], dim=2)
            
            # 更新缓存：把浓缩版的 K 和 V 存入缓存
            kv_cache = (K, V)

        # ==========================================
        # 🌟 核心修改 3：算分前的“就地膨胀” (repeat_interleave)
        # ==========================================
        # 因为 Q 有 32 个头，K 和 V 只有 8 个头，直接做矩阵乘法会报错 (维度不匹配)
        # 所以必须把 K 和 V 的头复制扩充回 32 个，才能和 Q 匹配。
        # 这一步发生在把 K/V 存入 Cache 之后！
        K_expanded = torch.repeat_interleave(K, repeats=self.num_groups, dim=1)
        V_expanded = torch.repeat_interleave(V, repeats=self.num_groups, dim=1)

        # 现在的 K_expanded 和 V_expanded 形状变成了 (batch, num_heads, seq_len, head_dim)
        # 接下来就是和传统 MHA 一模一样的标准计算流程了
        
        attention_score = torch.matmul(Q, K_expanded.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))
        
        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, V_expanded)  
        
        # concat 多头 -> (batch, seq_len, hidden_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attention_output)
        
        if use_kv:
            return output, kv_cache
            
        return output