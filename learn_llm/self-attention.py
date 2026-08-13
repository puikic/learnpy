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

# ============================================================
# Multi-head Latent Attention (MLA)  —— DeepSeek-V2 的杀手锏
# ------------------------------------------------------------
# 一句话原理：不再直接缓存又大又占显存的多头 K/V，而是先把输入“压缩”成一个
# 低维隐向量 c_kv（latent），只把这个小向量塞进 KV Cache；真正算注意力前
# 再把它“解压”还原成多头 K/V。这样 KV Cache 的体积从 O(num_heads*head_dim)
# 骤降到 O(kv_lora_rank)，长上下文推理时显存立省一大截。
#
# 有个坑：RoPE 是跟“位置”强绑定的，没法被吸收进升维矩阵里。所以 MLA 把
# Q/K 各自劈成两半——一半是不带位置信息的 nope 部分（走 latent 压缩通道），
# 另一半是专门承载位置信息的 rope 部分（走一条“解耦”的旁路）。
# ============================================================

def rotate_half(x):
    # 把最后一维前后两半做 (-后半, 前半) 的旋转，配合 cos/sin 实现复数域旋转
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def precompute_rope(seq_len, dim, base=10000.0):
    # 预先把每个位置、每个频率对应的 cos/sin 算好，用的时候直接查表
    assert dim % 2 == 0, "RoPE 的维度必须是偶数"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    pos = torch.arange(seq_len).float()                                 # (seq_len,)
    freqs = torch.outer(pos, inv_freq)                                  # (seq_len, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                             # (seq_len, dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    # x: (batch, num_heads, seq_len, rope_dim); cos/sin: (seq_len, rope_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, rope_dim) 便于广播
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads,
                 q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                 max_seq_len=1024, rope_base=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank            # Query 压缩后的维度
        self.kv_lora_rank = kv_lora_rank          # K/V 联合压缩后的维度（就是被缓存的 latent）
        self.qk_nope_head_dim = qk_nope_head_dim  # 每个头里“不带位置”的那半
        self.qk_rope_head_dim = qk_rope_head_dim  # 每个头里“带 RoPE”的那半
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 拼起来才是完整的 q/k 头维度
        self.v_head_dim = v_head_dim
        self.att_drop = nn.Dropout(0.1)

        # ---------- Query 通道：先降维压缩，再升维展开成多头 ----------
        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_down_norm = nn.RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        # ---------- K/V 通道：联合压缩成 latent，再升维还原多头 K(nope) 与 V ----------
        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank, bias=False)
        self.kv_down_norm = nn.RMSNorm(kv_lora_rank)
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        # ---------- 解耦 RoPE 的 key：直接从原始 hidden 投影，且所有头共享一份 ----------
        self.k_rope_proj = nn.Linear(hidden_dim, qk_rope_head_dim, bias=False)

        # 注意：输出投影的输入维度是 num_heads * v_head_dim，未必等于 hidden_dim（MLA 维度解耦）
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_dim, bias=False)

        cos, sin = precompute_rope(max_seq_len, qk_rope_head_dim, base=rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # ---------- Query：压缩 -> 升维 -> 拆成 nope / rope 两半 ----------
        q = self.q_up_proj(self.q_down_norm(self.q_down_proj(x)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---------- K/V：压缩成 latent c_kv，升维还原 k_nope 与 value ----------
        c_kv = self.kv_down_norm(self.kv_down_proj(x))                 # (b, s, kv_lora_rank)
        kv = self.kv_up_proj(c_kv)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ---------- 解耦 RoPE：q_rope 每头各一份；k_rope 所有头共享一份 ----------
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        q_rope = apply_rope(q_rope, cos, sin)                         # (b, num_heads, s, rope)
        k_rope = self.k_rope_proj(x).view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_rope = apply_rope(k_rope, cos, sin)                         # (b, 1, s, rope)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)            # 广播到所有头

        # ---------- 拼回完整 Q / K，然后走标准注意力 ----------
        query = torch.cat([q_nope, q_rope], dim=-1)                   # (b, num_heads, s, qk_head_dim)
        key = torch.cat([k_nope, k_rope], dim=-1)                     # (b, num_heads, s, qk_head_dim)

        attention_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.qk_head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, value)      # (b, num_heads, s, v_head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attention_output)
        return output


X = torch.rand(4, 3, 32)
n = MultiHeadLatentAttention(
    hidden_dim=32, num_heads=4,
    q_lora_rank=16, kv_lora_rank=8,
    qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
)
print(n(X).shape)   # torch.Size([4, 3, 32])


class MultiHeadLatentAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads,
                 q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                 max_seq_len=1024, rope_base=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.att_drop = nn.Dropout(0.1)

        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_down_norm = nn.RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank, bias=False)
        self.kv_down_norm = nn.RMSNorm(kv_lora_rank)
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        self.k_rope_proj = nn.Linear(hidden_dim, qk_rope_head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_dim, bias=False)

        cos, sin = precompute_rope(max_seq_len, qk_rope_head_dim, base=rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, mask=None, use_kv=False, kv_cache=None):
        batch_size, seq_len, _ = x.size()

        # decode 阶段要知道“现在写到第几个字了”，好给 RoPE 喂对绝对位置
        past_len = kv_cache[0].size(1) if (use_kv and kv_cache is not None) else 0

        # ---------- Query ----------
        q = self.q_up_proj(self.q_down_norm(self.q_down_proj(x)))
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---------- K/V：只算“压缩版” c_kv 和共享 k_rope（这俩才是要进缓存的东西） ----------
        c_kv = self.kv_down_proj(x)                                   # (b, s, kv_lora_rank) 缓存原始压缩量
        k_rope = self.k_rope_proj(x).view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # RoPE 用“绝对位置”这一段 cos/sin：新来的 token 从 past_len 开始编号
        cos = self.cos[past_len: past_len + seq_len]
        sin = self.sin[past_len: past_len + seq_len]
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)                         # (b, 1, s, rope) —— 缓存“已转好角度”的 k_rope

        # ==========================================
        # 🌟 MLA 的精髓：KV Cache 里只存 latent(c_kv) 和 一份共享的 k_rope
        #    对比 MHA 缓存完整多头 K/V，这里的体积小了整整 num_heads 倍级别
        # ==========================================
        if use_kv:
            if kv_cache is not None:
                past_c_kv, past_k_rope = kv_cache
                c_kv = torch.cat([past_c_kv, c_kv], dim=1)            # c_kv 的 seq 维在 dim=1
                k_rope = torch.cat([past_k_rope, k_rope], dim=2)      # k_rope 是 (b,1,s,rope)，seq 维在 dim=2
            kv_cache = (c_kv, k_rope)

        # ---------- 用的时候才“解压”：在完整历史的 c_kv 上升维还原多头 K(nope)/V ----------
        kv_seq_len = c_kv.size(1)
        kv = self.kv_up_proj(self.kv_down_norm(c_kv))
        kv = kv.view(batch_size, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope_full = k_rope.expand(-1, self.num_heads, -1, -1)       # 共享的 k_rope 广播到所有头

        query = torch.cat([q_nope, q_rope], dim=-1)                   # (b, H, s, qk_head_dim)
        key = torch.cat([k_nope, k_rope_full], dim=-1)                # (b, H, kv_seq_len, qk_head_dim)

        attention_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.qk_head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, value)      # (b, H, s, v_head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attention_output)

        # 生产级实现还会把 W_UK 吸收进 W_Q、W_UV 吸收进 W_O（矩阵吸收），
        # 连升维都省了、直接拿 latent 算分，这里为了好懂保留了显式解压。
        if use_kv:
            return output, kv_cache
        return output


# 验证：KV Cache 逐 token 解码，应当和“一次性 + 因果 mask”的结果完全一致
torch.manual_seed(0)
mla_kv = MultiHeadLatentAttentionWithKVCache(
    hidden_dim=32, num_heads=4,
    q_lora_rank=16, kv_lora_rank=8,
    qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
)
mla_kv.eval()  # 关掉 dropout，两条路径才可比

X = torch.rand(1, 5, 32)
causal_mask = torch.tril(torch.ones(5, 5))          # 下三角：只能看自己和前面的字
full_out = mla_kv(X, mask=causal_mask)              # 方式一：一次性 prefill

kv_cache = None                                     # 方式二：一个字一个字地 decode
step_outs = []
for t in range(5):
    out_t, kv_cache = mla_kv(X[:, t:t + 1, :], use_kv=True, kv_cache=kv_cache)
    step_outs.append(out_t)
step_out = torch.cat(step_outs, dim=1)

print("逐token解码 vs 一次性因果forward 是否一致:", torch.allclose(full_out, step_out, atol=1e-5))
print("缓存的 latent c_kv 形状:", kv_cache[0].shape, " 共享 k_rope 形状:", kv_cache[1].shape)

# ============================================================
# 简化版 MLA（不考虑 RoPE）—— 只保留「低秩压缩 KV Cache」这条主线
# ------------------------------------------------------------
# 去掉 RoPE 后，就不用再把 Q/K 劈成 nope/rope 两半了：
#   K/V 可以整体压进一个 latent 向量 c_kv，用的时候再升维解压。
# 这一版专门用来看清 MLA 最核心的那句话：
#   「KV Cache 里只存又小又浓缩的 c_kv，多头 K/V 是算分前临时解压出来的」。
# 代价：没有位置编码，纯粹拿来理解压缩/解压机制，不能直接上生产。
# ============================================================

class SimpleMLA(nn.Module):
    def __init__(self, hidden_dim, num_heads, q_lora_rank, kv_lora_rank):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_lora_rank = q_lora_rank      # Query 压缩后的维度
        self.kv_lora_rank = kv_lora_rank    # K/V 联合压缩后的维度（就是被缓存的 latent）
        self.att_drop = nn.Dropout(0.1)

        # ---------- Query 通道：降维压缩 -> 升维展开成多头 ----------
        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_down_norm = nn.RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)

        # ---------- K/V 通道：联合压缩成 latent -> 升维还原成多头 K 和 V ----------
        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank, bias=False)
        self.kv_down_norm = nn.RMSNorm(kv_lora_rank)
        # 一次升维同时吐出 K 和 V，所以输出维度是 2 * hidden_dim
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim * 2, bias=False)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # ---------- Query：压缩 -> 升维 -> 拆多头 ----------
        Q = self.q_up_proj(self.q_down_norm(self.q_down_proj(x)))
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- K/V：压缩成 latent c_kv -> 升维 -> 劈成 K 和 V ----------
        c_kv = self.kv_down_norm(self.kv_down_proj(x))              # (batch, seq_len, kv_lora_rank)
        kv = self.kv_up_proj(c_kv)                                 # (batch, seq_len, 2*hidden_dim)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.head_dim * 2).transpose(1, 2)
        K, V = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        # ---------- 标准注意力（和最上面的 SelfAttention 一模一样） ----------
        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, V)       # (batch, num_heads, seq_len, head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attention_output)
        return output


X = torch.rand(4, 3, 32)
n = SimpleMLA(hidden_dim=32, num_heads=4, q_lora_rank=16, kv_lora_rank=8)
print(n(X).shape)   # torch.Size([4, 3, 32])


class SimpleMLAWithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads, q_lora_rank, kv_lora_rank):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.att_drop = nn.Dropout(0.1)

        self.q_down_proj = nn.Linear(hidden_dim, q_lora_rank, bias=False)
        self.q_down_norm = nn.RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(q_lora_rank, num_heads * self.head_dim, bias=False)

        self.kv_down_proj = nn.Linear(hidden_dim, kv_lora_rank, bias=False)
        self.kv_down_norm = nn.RMSNorm(kv_lora_rank)
        self.kv_up_proj = nn.Linear(kv_lora_rank, num_heads * self.head_dim * 2, bias=False)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, mask=None, use_kv=False, kv_cache=None):
        batch_size, seq_len, _ = x.size()

        # ---------- Query ----------
        Q = self.q_up_proj(self.q_down_norm(self.q_down_proj(x)))
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- K/V：先只算出「压缩版」c_kv，它才是要进缓存的东西 ----------
        c_kv = self.kv_down_proj(x)                                 # (batch, seq_len, kv_lora_rank)

        # ==========================================
        # 🌟 MLA 精髓：KV Cache 里只存又小又浓缩的 latent c_kv
        #    对比 MHA 缓存完整多头 K/V，体积小了整整 (num_heads*head_dim / kv_lora_rank) 倍
        # ==========================================
        if use_kv:
            if kv_cache is not None:
                past_c_kv = kv_cache
                c_kv = torch.cat([past_c_kv, c_kv], dim=1)         # c_kv 的 seq 维在 dim=1
            kv_cache = c_kv

        # ---------- 用的时候才「解压」：在完整历史的 c_kv 上升维还原多头 K/V ----------
        kv_seq_len = c_kv.size(1)
        kv = self.kv_up_proj(self.kv_down_norm(c_kv))
        kv = kv.view(batch_size, kv_seq_len, self.num_heads, self.head_dim * 2).transpose(1, 2)
        K, V = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, float('-inf'))
        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_drop(attention_weight)

        attention_output = torch.matmul(attention_weight, V)       # (batch, num_heads, seq_len, head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attention_output)

        if use_kv:
            return output, kv_cache
        return output


# 验证：KV Cache 逐 token 解码，应当和「一次性 + 因果 mask」的结果完全一致
torch.manual_seed(0)
mla = SimpleMLAWithKVCache(hidden_dim=32, num_heads=4, q_lora_rank=16, kv_lora_rank=8)
mla.eval()  # 关掉 dropout，两条路径才可比

X = torch.rand(1, 5, 32)
causal_mask = torch.tril(torch.ones(5, 5))          # 下三角：只能看自己和前面的字
full_out = mla(X, mask=causal_mask)                 # 方式一：一次性 prefill

kv_cache = None                                     # 方式二：一个字一个字地 decode
step_outs = []
for t in range(5):
    out_t, kv_cache = mla(X[:, t:t + 1, :], use_kv=True, kv_cache=kv_cache)
    step_outs.append(out_t)
step_out = torch.cat(step_outs, dim=1)

print("逐token解码 vs 一次性因果forward 是否一致:", torch.allclose(full_out, step_out, atol=1e-5))
print("缓存的 latent c_kv 形状:", kv_cache.shape, "（每个字只存", mla.kv_lora_rank, "维）")
