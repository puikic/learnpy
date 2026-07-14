"""理解 output_attentions 的作用

本文件由 output_attentions_tutorial.ipynb 转写而来，风格对齐 self-attention.py：
把原来的 Markdown 教学内容保留为注释，代码按“可运行的小节”组织。

============================================================
什么是 Attention（注意力）？
============================================================
想象你在读一句话："我爱吃苹果"。当模型处理"苹果"这个词时，它会：
  - 关注"我"    （谁在吃？）
  - 关注"爱吃"  （什么动作？）
  - 关注"苹果"本身
这种"关注程度"就是 attention scores（注意力分数）。

============================================================
output_attentions 的作用
============================================================
  - output_attentions = False（默认）：模型只给最终结果，不告诉你它怎么"关注"各个词
  - output_attentions = True         ：模型额外返回每一层、每个头的注意力分数，
                                        让你看到模型的"思考过程"

下面用代码演示两者的区别。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ============================================================
# 小节 1：对比 output_attentions = False vs True
# ============================================================
def compare_output_attentions():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer("I love Python", return_tensors="pt")   # input_ids: (1, seq_len)

    # ---------- 示例 1：默认（False）----------
    print("=" * 60)
    print("示例1：output_attentions = False（默认情况）")
    print("=" * 60)

    model_default = AutoModelForCausalLM.from_pretrained("gpt2")
    with torch.no_grad():
        outputs_default = model_default(**inputs)

    print(f"返回内容包含的字段：{outputs_default.keys()}")
    print(f"outputs.logits 形状: {outputs_default.logits.shape}")   # (1, seq_len, vocab)
    print(f"outputs.attentions 是否为 None: {outputs_default.attentions is None}")
    print("结论：默认只返回 logits（预测结果），不返回 attentions\n")

    # ---------- 示例 2：打开（True）----------
    print("=" * 60)
    print("示例2：output_attentions = True")
    print("=" * 60)

    # 🌟 关键：在 config 上把 output_attentions 打开，再用这个 config 加载模型
    config = AutoConfig.from_pretrained("gpt2")
    config.output_attentions = True
    model_with_attn = AutoModelForCausalLM.from_pretrained("gpt2", config=config)

    with torch.no_grad():
        outputs_with_attn = model_with_attn(**inputs)

    print(f"返回内容包含的字段：{outputs_with_attn.keys()}")
    print(f"outputs.logits 形状: {outputs_with_attn.logits.shape}")
    print(f"outputs.attentions 是否为 None: {outputs_with_attn.attentions is None}")

    if outputs_with_attn.attentions is not None:
        print("现在可以看到 attentions 了！")
        # attentions 是一个 tuple，长度 = 层数；每层形状 (batch, num_heads, seq_len, seq_len)
        print(f"  - GPT-2 共有 {len(outputs_with_attn.attentions)} 层")
        print(f"  - 第 1 层的 attention 形状: {outputs_with_attn.attentions[0].shape}")
        print("    解释: (batch_size=1, num_heads=12, seq_len=3, seq_len=3)")
        print("    含义: 1 个样本，12 个头，3 个 token，每个 token 对其他 3 个 token 的注意力分数")


# ============================================================
# 小节 2：把注意力分数“打印成表格”看清楚
# ============================================================
# 看模型处理 "I love Python" 时，每个词对其他词的关注程度。
def visualize_attention():
    # 用打开了 output_attentions 的模型
    config = AutoConfig.from_pretrained("gpt2")
    config.output_attentions = True
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text = "I love Python"
    inputs = tokenizer(text, return_tensors="pt")

    # 把 id 转回可读的 token 字符串，用来当表头
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    print(f"输入句子: '{text}'")
    print(f"分词结果: {tokens}\n")

    with torch.no_grad():
        outputs = model(**inputs)

    # 取第 1 层、第 1 个头：outputs.attentions[0] 是 (batch, num_heads, seq_len, seq_len)
    # 再 [0, 0] 取 batch0、head0 -> (seq_len, seq_len)
    first_layer_first_head = outputs.attentions[0][0, 0]   # (seq_len, seq_len)

    print("=" * 60)
    print("第 1 层、第 1 个头的注意力分数")
    print("=" * 60)
    print("含义：每一行表示该 token 对所有 token 的关注程度\n")

    # 打印表头（每列一个 token）
    print(f"{'':>10}", end="")
    for token in tokens:
        print(f"{token:>12}", end="")
    print()

    # 逐行打印：行是“当前 token”，列是“被关注的 token”
    for i, token in enumerate(tokens):
        print(f"{token:>10}", end="")
        for j in range(len(tokens)):
            score = first_layer_first_head[i, j].item()
            print(f"{score:>12.4f}", end="")
        print()

    # 挑一个词举例说明怎么读这张表
    print("\n" + "=" * 60)
    print("举例说明：")
    print("=" * 60)
    print(f"当模型处理 '{tokens[1]}' 这个词时：")
    print(f"  - 对 '{tokens[0]}' 的关注分数: {first_layer_first_head[1, 0].item():.4f}")
    print(f"  - 对 '{tokens[1]}' 的关注分数: {first_layer_first_head[1, 1].item():.4f}")
    print(f"  - 对 '{tokens[2]}' 的关注分数: {first_layer_first_head[1, 2].item():.4f}")
    print("\n分数越大，表示关注程度越高！")


# ============================================================
# 总结
# ============================================================
# config.output_attentions = True 的作用：
#   1. 默认(False)：模型只返回 logits（预测结果）——像老师只报答案。
#   2. 设为 True  ：额外返回所有层、所有头的注意力分数——像老师讲解每一步怎么想。
#
# 什么时候设 True？
#   - 想研究模型如何"理解"句子
#   - 想可视化注意力模式
#   - 想调试 / 分析模型行为
#   - 只要预测结果时不要开（会增加显存和计算开销）
#
# 常见坑：不打开 output_attentions 就去访问 outputs.attentions[0]，
# 会因为它是 (None, None, ...) 而报 TypeError: 'NoneType' object is not subscriptable。


if __name__ == "__main__":
    # 两段都需要联网下载 gpt2；网络就绪后按需运行。
    compare_output_attentions()
    print()
    visualize_attention()
