"""因果语言模型（Causal LM）数据预处理教学脚本

本文件由 causal-lm.ipynb 转写而来，风格对齐 self-attention.py：
把原来的 Markdown 讲解保留为注释，代码按“可运行的小节”组织。

主题：训练 GPT 类模型前，如何把一堆长短不一的文本，
      处理成固定长度 block_size 的训练样本。核心是两步：
        1) tokenize：文本 -> token id
        2) group_texts：把多条文本拼接后按 block_size 切成等长块

依赖 datasets / transformers，且第一段要联网下载数据集与分词器。
每个函数都能单独看懂；末尾的 __main__ 里有一段纯 Python 的逻辑演示，不需要联网。
"""


# ============================================================
# 小节 1：登录 HuggingFace（可选，下载私有资源才需要）
# ============================================================
def hf_login():
    from huggingface_hub import notebook_login
    notebook_login()


# ============================================================
# 小节 2：加载数据集并切分训练/测试集
# ============================================================
def load_eli5():
    from datasets import load_dataset

    # 只取前 5000 条，学习用足够了
    eli5 = load_dataset("dany0407/eli5_category", split="train[:5000]")

    # 切 20% 做测试集
    eli5 = eli5.train_test_split(test_size=0.2)
    print("一条原始样本:", eli5["train"][0])
    return eli5


# ============================================================
# 小节 3：分词（tokenize）
# ============================================================
# flatten() 把嵌套字段（如 answers.text）拍平成顶层列，方便后面按列名取数据。
def tokenize_dataset(eli5):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    eli5 = eli5.flatten()
    print("flatten 后的样本:", eli5["train"][0])

    # 🌟 分词函数：注意用了 batched=True，所以 examples 里每个字段是“一批样本的列表”
    # answers.text 是 list[list[str]]，先把每条的多段文本用空格拼成一整段，再一起分词
    def preprocess_function(examples, tokenizer):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,        # 一次喂一批（默认 1000 条）给函数，快很多
        num_proc=4,          # 4 进程并行
        remove_columns=eli5["train"].column_names,   # 丢掉原始列，只留分词结果
        fn_kwargs={"tokenizer": tokenizer},
    )
    print("分词后的样本:", tokenized_eli5["train"][0])
    return tokenized_eli5, tokenizer


# ============================================================
# 小节 4：理解 batched=True 到底给函数传了什么
# ============================================================
# 关键点：开启 batched=True 后，map 不是一条一条传，而是“一次抓一批”传给你的函数。
#
#   存储时：一个个独立样本
#       样本1: [1, 2, 3]
#       样本2: [4, 5]
#   传入函数时(examples)：变成“包含多个列表的大列表”
#       examples['input_ids'] -> [[1, 2, 3], [4, 5], ...]（默认 1000 个小列表）
#   sum(..., [])：把这些小列表“压扁”拼成一个超长列表
#
# 下面这个函数把传进来的结构打印出来，眼见为实：
def inspect_batch_structure(tokenized_eli5):
    def _inspect(examples):
        print("=" * 40)
        print(f"函数收到的 input_ids 类型: {type(examples['input_ids'])}")
        print(f"这是一个包含 {len(examples['input_ids'])} 个列表的大列表")
        print(f"前 3 个列表: {examples['input_ids'][:3]}")
        print("=" * 40)
        return examples

    print("开始模拟 map(batched=True)...")
    dummy_dataset = tokenized_eli5.select(range(5))   # 只取前 5 条模拟
    dummy_dataset.map(_inspect, batched=True, batch_size=5)


# ============================================================
# 小节 5：group_texts —— 把多条文本拼接后切成等长块
# ============================================================
# 作用：把多个短文本首尾拼接，再按固定 block_size 切分成等长样本。
#
# 举例（block_size = 4），有 3 个短句：[1,2] / [3,4,5] / [6]
#   1. 拼接        -> [1, 2, 3, 4, 5, 6]        (总长 6)
#   2. 算新长度    -> 6 // 4 * 4 = 4            (丢弃余数)
#   3. 切分        -> [1, 2, 3, 4]             (只剩 1 个样本，丢掉了 [5, 6])
#
# 为什么这么做？训练时通常要固定长度输入（128 / 1024 等）。
# 把短句拼起来切等长块，能避免大量 padding，不浪费算力。
def group_texts(examples, tokenizer, block_size):
    # 1) 把这一批里每个字段的多个小列表，拼接成一个超长列表
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # 2) 以第一个字段的长度为总长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # 3) 向下取整到 block_size 的整数倍（丢弃末尾不足一块的余数）
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # 4) 每 block_size 个 token 切成一块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # 🌟 因果 LM 的 label 就是 input_ids 本身（模型内部会自动错位：用第 t 位预测第 t+1 位）
    result["labels"] = result["input_ids"].copy()
    return result


def build_lm_dataset(tokenized_eli5, tokenizer, block_size=128):
    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer, "block_size": block_size},
    )
    print("成块后的样本:", lm_dataset["train"][0])
    return lm_dataset


# ============================================================
# 小节 6：纯 Python 演示 group_texts 的逻辑（不需要联网）
# ============================================================
def demo_group_texts():
    # 3 个短句，block_size = 4
    examples = {
        "input_ids": [[1, 2], [3, 4, 5], [6]],
        "attention_mask": [[1, 1], [1, 1, 1], [1]],
    }
    block_size = 4

    print("1. 原始数据:", examples["input_ids"])

    # 拼接：sum(list_of_lists, []) 是把多个列表压扁的常用技巧
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    print("2. 拼接后:", concatenated["input_ids"])

    total_length = len(concatenated["input_ids"])
    print("3. 总长度:", total_length)

    dropped_from = total_length
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    print(f"4. 截断后长度: {total_length} (丢弃了最后 {dropped_from - total_length} 个 token)")

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    print(f"5. 最终结果 (每块长度 {block_size}): {result['input_ids']}")


if __name__ == "__main__":
    # 纯 Python 的逻辑演示，直接可跑：
    demo_group_texts()

    # 完整流程需要联网下载数据集/分词器，按需手动打开：
    # eli5 = load_eli5()
    # tokenized_eli5, tokenizer = tokenize_dataset(eli5)
    # inspect_batch_structure(tokenized_eli5)
    # lm_dataset = build_lm_dataset(tokenized_eli5, tokenizer, block_size=128)
