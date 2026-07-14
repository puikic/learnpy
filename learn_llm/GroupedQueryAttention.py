"""GQA / MQA 学习脚本

本文件由 GroupedQueryAttention.ipynb 转写而来，改成和 self-attention.py 一样的风格：
- 每个张量操作都标注形状
- 用“直觉优先”的中文注释解释在干什么
- 用 🌟 标出关键逻辑
- 每一块都能单独看懂

分成两部分：
  Part A（需要联网下载 gpt2）：用 transformers 观察真实模型的注意力 & KV Cache 加速效果
  Part B（纯 torch，可直接跑）：手写 MHA / MQA 的“最朴素循环版”，理解多头的本质
"""

import time
import torch
import torch.nn as nn


# ============================================================
# Part A-1：用 GPT-2 观察自回归生成时，每一步的注意力分数
# ============================================================
# output_attentions=True 会让模型额外吐出每一层、每个头的注意力权重，
# 我们借此“看见”模型在生成下一个词时到底在关注前面哪些词。
def watch_attention_during_generation(num_new_tokens=3):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # 🌟 关键：必须在 config 里打开 output_attentions，否则 outputs.attentions 是 None
    config = AutoConfig.from_pretrained("gpt2")
    config.output_attentions = True

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with torch.no_grad():
        # 把起始句子编码成 token id
        inputs = gpt2_tokenizer("Hope is a", return_tensors="pt", add_special_tokens=False)

        # 手写一个“贪心生成”循环：一次生成一个新 token
        for i in range(num_new_tokens):
            print("-" * 50)
            print(f"Iteration {i + 1}")
            print(f"当前序列: '{gpt2_tokenizer.decode(inputs.input_ids[0])}'")

            outputs = gpt2(inputs.input_ids)          # 前向：输入 (1, cur_len)
            logits = outputs.logits                   # (1, cur_len, vocab)

            # 只看最后一个位置的预测分布，取分数最高的 token 作为下一个词
            next_token = torch.argmax(logits[:, -1, :])   # 标量

            # 把新 token 拼回输入末尾，序列长度 +1，进入下一轮
            inputs.input_ids = torch.cat(
                [inputs.input_ids, next_token.reshape([1, 1])], dim=-1
            )
            print("生成的 token:", gpt2_tokenizer.decode(next_token))

            # 取第 0 层、第 0 个头的注意力矩阵看看
            # outputs.attentions[层] -> (batch, num_heads, cur_len, cur_len)
            first_layer_attentions = outputs.attentions[0][0]   # (num_heads, cur_len, cur_len)
            print("第 1 个头的注意力分数矩阵:")
            print(first_layer_attentions[0])                    # (cur_len, cur_len)

        print("-" * 50)
        print(f"最终序列: '{gpt2_tokenizer.decode(inputs.input_ids[0])}'")


# ============================================================
# Part A-2：用 KV Cache 前后的生成耗时对比
# ============================================================
# 结论先行：开启 KV Cache 后，每个新 token 不用把整段历史重新算一遍 K/V，
# 生成速度会明显变快。这里用 generate 的 use_cache 开关直接对比。
def benchmark_kv_cache(use_kv_cache, num_new_tokens=500):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=use_kv_cache)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    with torch.no_grad():
        inputs = gpt2_tokenizer("Hope is a", return_tensors="pt", add_special_tokens=False)

        start_time = time.time()
        # min/max 都设成同一个值，保证两次生成的 token 数完全一样，耗时才可比
        gpt2.generate(**inputs, max_new_tokens=num_new_tokens, min_new_tokens=num_new_tokens)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"生成 {num_new_tokens} 个 token 共耗时: {elapsed:.4f} 秒")
        print(f"平均每个 token: {elapsed / num_new_tokens:.4f} 秒")


# ============================================================
# Part B-1：MHA 的“最朴素循环版”——每个头一套独立的 Q/K/V
# ============================================================
# 这是理解多头注意力最直观的写法：num_heads 个头，就 new num_heads 套 Q/K/V 投影，
# 各算各的，最后把结果收集进一个 list。
# （self-attention.py 里的 MultiHeadAttention 是它的“张量并行版”：
#   用一个大 Linear + reshape 一次算完所有头，效率更高，但没这个直观。）
class MultiHeadAttentionScores(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads   # 头数，例如 8 / 16 / 32

        # 🌟 每个头都有自己独立的 W^Q / W^K / W^V
        # 用 ModuleList 装 num_attention_heads 个 Linear
        self.query_layers = nn.ModuleList(
            [nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)]
        )
        self.key_layers = nn.ModuleList(
            [nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)]
        )
        self.value_layers = nn.ModuleList(
            [nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)]
        )

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        all_attention_outputs = []   # 收集每个头的输出

        for i in range(self.num_attention_heads):
            # 第 i 个头，用它自己的投影层
            query_vectors = self.query_layers[i](hidden_states)   # (batch, seq_len, head_size)
            key_vectors = self.key_layers[i](hidden_states)       # (batch, seq_len, head_size)
            value_vectors = self.value_layers[i](hidden_states)   # (batch, seq_len, head_size)

            # QK^T：算 token 两两之间的相关度 (batch, seq_len, seq_len)
            # 注意：这个朴素版没做 /sqrt(d) 缩放，也没做 softmax，只是演示骨架
            attention_scores = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

            # 用分数对 value 加权求和 (batch, seq_len, head_size)
            attention_outputs = torch.matmul(attention_scores, value_vectors)
            all_attention_outputs.append(attention_outputs)

        return all_attention_outputs   # list，长度 = num_heads，每项 (batch, seq_len, head_size)


# ============================================================
# Part B-2：MQA（Multi-Query Attention）——多个 Q 头，共享 1 套 K/V
# ============================================================
# 和上面 MHA 的唯一区别：K 和 V 只有一份，被所有 Q 头共用。
# 这就是 MQA 省显存的核心：KV Cache 只需缓存 1 个头的 K/V。
# （GQA 是折中：把头分成几组，每组共享一套 K/V，介于 MHA 和 MQA 之间。）
class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads

        # Q 依然每个头一套
        self.query_layers = nn.ModuleList(
            [nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)]
        )

        # 🌟 核心区别：K / V 各只有一套，所有头共享
        self.key_layer = nn.Linear(hidden_size, attention_head_size)
        self.value_layer = nn.Linear(hidden_size, attention_head_size)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_size)
        all_attention_outputs = []

        for i in range(self.num_attention_heads):
            query_vectors = self.query_layers[i](hidden_states)   # (batch, seq_len, head_size) 每个头不同
            key_vectors = self.key_layer(hidden_states)           # (batch, seq_len, head_size) 所有头相同
            value_vectors = self.value_layer(hidden_states)       # (batch, seq_len, head_size) 所有头相同

            attention_scores = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))  # (batch, seq_len, seq_len)
            attention_outputs = torch.matmul(attention_scores, value_vectors)              # (batch, seq_len, head_size)
            all_attention_outputs.append(attention_outputs)

        return all_attention_outputs


if __name__ == "__main__":
    # Part B 是纯 torch，可以直接跑
    X = torch.rand(2, 4, 16)   # (batch=2, seq_len=4, hidden_size=16)

    mha = MultiHeadAttentionScores(hidden_size=16, num_attention_heads=4, attention_head_size=8)
    out_mha = mha(X)
    print("MHA 输出头数:", len(out_mha), "| 每个头形状:", out_mha[0].shape)

    mqa = MultiQueryAttention(hidden_size=16, num_attention_heads=4, attention_head_size=8)
    out_mqa = mqa(X)
    print("MQA 输出头数:", len(out_mqa), "| 每个头形状:", out_mqa[0].shape)

    # Part A 需要联网下载 gpt2，默认注释掉，需要时手动打开：
    # import transformers; print("transformers 版本:", transformers.__version__)
    # watch_attention_during_generation(num_new_tokens=3)
    # print("不使用 KV Cache:"); benchmark_kv_cache(use_kv_cache=False)
    # print("使用 KV Cache:");   benchmark_kv_cache(use_kv_cache=True)
