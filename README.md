# learn_py

Python 学习项目，系统记录从语言基础、算法到 LLM（大语言模型）核心组件的实现与理解。

## 项目结构

```
learn_py/
├── algorithm/                      # 算法实现
│   ├── heap_sort.py                #   最小堆（MinHeap）实现
│   └── quick_sort.py               #   快速排序（Hoare 分割）
├── learn_llm/                      # LLM 核心组件学习
│   ├── self-attention.py               #   Self-Attention / MHA / GQA / KV Cache 实现
│   ├── transformer-block.py            #   LLaMA 风格 Transformer Block（RMSNorm/SwiGLU/因果注意力）
│   ├── cross-entropy.py                #   手写交叉熵 / 因果 LM 损失
│   ├── rl.py                           #   GRPO 损失函数实现
│   ├── dpo-ppo.py                      #   DPO / PPO 损失函数实现
│   ├── GroupedQueryAttention.py        #   GQA/MQA 朴素实现 + GPT-2 注意力观察
│   ├── causal-lm.py                    #   Causal LM 数据预处理（ELI5 数据集）
│   └── output_attentions_tutorial.py   #   Attention 可视化教程
├── agent/                          # AI Agent 实践
│   └── test-apikey.py              #   火山引擎 ARK API 调用示例
├── numpy/                          # NumPy 学习
│   └── np.py                       #   广播规则（broadcasting）演示
├── main.py                         # uv 项目入口
├── func.py                         # Python 引用语义（ListNode 演示）
├── list.py                         # 切片赋值与列表操作演示
├── test.ipynb                      # 环境自检（torch 版本与 GPU 可用性）
├── pyproject.toml                  # 项目配置与依赖
└── uv.lock                         # 依赖锁定
```

## 模块说明

### algorithm/ — 算法

#### `heap_sort.py`
最小堆（MinHeap）实现，支持 `push`、`pop` 和批量建堆（`init`）。
- **建堆方向**：从 `len(heap)//2 - 1` 倒序到 0，自底向上逐个下沉，保证子树先成堆。
- **`_down`**：`while True` 配合 `break` 终止，与左右孩子比较后交换。
- **`_up`**：`while i > 0` 配合 `break` 终止，父节点 `<=` 当前即停。

#### `quick_sort.py`
经典快速排序，采用 Hoare 风格的双指针分割。
- `split(arr)` 原地修改数组并返回 pivot 最终位置。
- `quick_sort(arr)` 递归返回新数组。

### learn_llm/ — LLM 核心组件

#### `self-attention.py`
手写实现多种注意力机制，便于对比学习从基础 Attention 到推理加速所需 KV Cache 的演进：

| 类 | 机制 | K/V 复制方式 | 说明 |
|---|---|---|---|
| `SelfAttention` | 单头注意力 | — | 无 `o_proj`，`value_proj` 同时承担投影与输出角色 |
| `MultiHeadAttention` | 多头注意力（MHA） | — | 含 `o_proj`，对多头拼接结果做特征融合 |
| `GroupedQueryAttentionV1` | 分组查询注意力 | `repeat_interleave` | 直观写法，相邻 Q 头共享同一 KV 头 |
| `GroupedQueryAttentionV2` | 分组查询注意力 | `repeat_kv`（expand + reshape） | 更接近 HuggingFace 生产实现 |
| `MultiHeadAttentionWithKVCache` | MHA + KV Cache | — | 在自回归解码时缓存历史 K/V，并沿序列维度拼接新 token 的 K/V |
| `GroupedQueryAttentionWithKVCache` | GQA + KV Cache | `repeat_interleave` | Cache 中只保存较少的 KV 头，计算 attention 前再扩展到 Q 头数，降低缓存显存占用 |

> **GQA 关键点**：Q 头数 `H` 必须是 KV 头数 `H_kv` 的整数倍。由于 `torch.matmul` 要求非单例维度匹配，无法靠广播自动处理 `H ≠ H_kv`，必须先用 `repeat_interleave` 或 `repeat_kv` 把 K/V 在头维度显式复制到与 Q 相同的形状。
>
> **KV Cache 关键点**：Decode 阶段每次只输入新 token 时，Q 的序列长度通常很短，但 K/V 需要包含历史上下文；因此缓存沿 `seq_len` 维度拼接历史 K/V。GQA 版本先缓存压缩后的 `num_kv_heads` 个 K/V 头，真正计算前再扩展，体现了 GQA 在推理缓存上的显存优势。

#### `transformer-block.py`
LLaMA 风格的 Pre-Norm Transformer Block，把主干组件拆开手写便于理解：
- `RMSNorm`：去掉减均值、只用均方根做缩放的归一化
- `SwiGLU`：门控前馈网络，`W_down(SiLU(W_gate(x)) * W_up(x))`
- `CausalSelfAttention`：带下三角 mask 的多头因果自注意力
- `TransformerBlock`：`x = x + Attn(Norm(x))`、`x = x + FFN(Norm(x))` 两条残差

#### `cross-entropy.py`
手写交叉熵损失，对齐 `F.cross_entropy` 并附自测：
- 数值稳定版 `log_softmax`（先减最大值防溢出）
- 支持 `ignore_index`（padding 不计入 loss），mean 时只除以有效 token 数
- `causal_lm_loss`：logits/labels 错位对齐，实现自回归语言模型损失

#### `rl.py`
GRPO（Group Relative Policy Optimization）损失函数实现：
- 组内 reward 标准化得到 group-relative advantage
- PPO 风格的 clipped objective：`min(ratio * adv, clamp(ratio) * adv)`
- 与参考模型的 KL 散度正则：`F.kl_div(log_probs, ref_log_probs, log_target=True)`
- 附带 `TinyLM` 最小可运行示例，可直接跑通整条 loss 流程

#### `dpo-ppo.py`
偏好对齐 / 强化学习的两类损失函数实现：
- `dpo_loss`：DPO 直接偏好优化，把 (chosen, rejected) 偏好对转成逻辑回归，无需奖励模型
- `ppo_policy_loss` / `ppo_value_loss`：PPO 的截断策略损失与价值损失
- `sequence_logprobs`：token 级 log prob 汇总为序列级

#### `GroupedQueryAttention.py`
GQA/MQA 学习脚本（由 notebook 转写为脚本）：
- Part A（需联网）：用 GPT-2 观察生成时的注意力分数、对比 KV Cache 的提速效果
- Part B（纯 torch，可直接跑）：用 `ModuleList` 写出最朴素的循环版 MHA / MQA，理解多头本质

#### `causal-lm.py`
基于 HuggingFace `transformers` 的 Causal LM 数据预处理（由 notebook 转写为脚本），使用 ELI5 数据集，讲解 tokenization、`batched=True` 的批处理机制与 `group_texts` 定长切块；末尾附纯 Python 演示，无需联网即可运行。

#### `output_attentions_tutorial.py`
GPT-2 的 `output_attentions` 配置演示（由 notebook 转写为脚本），可视化每一层、每个注意力头的注意力分数矩阵。

### agent/ — AI Agent 实践

#### `test-apikey.py`
火山引擎 ARK API 调用示例，通过 `OpenAI` 兼容接口发送多模态请求（图片 + 文本）。
- 依赖环境变量 `ARK_API_KEY`
- 依赖 `openai` 包（通过 conda 环境安装，未在 `pyproject.toml` 中声明）

### numpy/ — NumPy 基础

#### `np.py`
NumPy 广播规则演示，展示不同形状数组运算时的维度扩展行为。

### 根目录文件

| 文件 | 说明 |
|---|---|
| `main.py` | uv 项目入口模板，打印 `Hello from learnpy!` |
| `func.py` | 通过 `ListNode` 演示 Python 引用语义（变量是标签而非盒子） |
| `list.py` | 演示列表的切片赋值与浅拷贝行为 |
| `test.ipynb` | 环境自检，打印 `torch.__version__` 和 CUDA 可用性 |

## 环境配置

### uv 虚拟环境

```bash
# 安装 uv（如未安装）
pip install uv

# 同步依赖（会自动创建 .venv 并安装 pyproject.toml 中的依赖）
uv sync

# 运行脚本
uv run python algorithm/heap_sort.py
uv run python learn_llm/self-attention.py
```

要求 Python ≥ 3.13。`learn_llm/` 下原有的教学 notebook 已统一转写为 `.py` 脚本，可直接用 `uv run` 运行（其中依赖联网下载模型/数据集的部分在文件内已标注，按需手动开启）。

> 根目录的 `test.ipynb` 为环境自检 notebook，如需运行可在 Jupyter 中选择项目 kernel。

## 使用方式

```bash
# 运行算法
uv run python algorithm/heap_sort.py    # 输出堆排序结果
uv run python algorithm/quick_sort.py   # 输出快速排序结果

# 运行注意力实现（会打印多种注意力的输出张量）
uv run python learn_llm/self-attention.py

# 运行 Transformer Block（打印各组件与整块输出形状）
uv run python learn_llm/transformer-block.py

# 运行手写交叉熵（与 F.cross_entropy 对比）
uv run python learn_llm/cross-entropy.py

# 运行 GRPO 损失函数（内置 TinyLM 示例，可直接跑通）
uv run python learn_llm/rl.py

# 运行 DPO / PPO 损失函数（内置自测）
uv run python learn_llm/dpo-ppo.py

# 运行 causal-lm 数据预处理演示（末尾纯 Python 演示无需联网）
uv run python learn_llm/causal-lm.py

# 测试 ARK API（需先设置 ARK_API_KEY 环境变量）
$env:ARK_API_KEY = "your-api-key"        # PowerShell
uv run python agent/test-apikey.py
```

## 学习要点

本项目记录了以下关键知识点：

1. **数据结构**：堆的自底向上建堆、`_up`/`_down` 维护逻辑
2. **排序算法**：快排的 Hoare 分割与递归结构
3. **注意力机制**：从单头到 MHA、GQA、MQA、KV Cache 的演进，理解 `o_proj` 的必要性、`repeat_interleave` 的广播原理和自回归解码中的 K/V 缓存复用
4. **Transformer 主干**：LLaMA 风格的 RMSNorm、SwiGLU 门控前馈、Pre-Norm 残差结构
5. **损失函数**：手写交叉熵与 `ignore_index` 处理、因果 LM 的错位对齐
6. **强化学习 / 对齐**：GRPO 的 group-relative advantage、PPO clipped objective 与价值损失、DPO 偏好对齐、KL 正则
7. **LLM 工具链**：HuggingFace `transformers` 的 Causal LM 数据预处理、`output_attentions` 可视化
8. **Python 语义**：引用语义、切片赋值、可变默认参数陷阱
9. **NumPy**：广播规则的维度对齐机制
