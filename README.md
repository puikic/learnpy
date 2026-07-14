# learn_py

Python 学习项目，系统记录从语言基础、算法到 LLM（大语言模型）核心组件的实现与理解。

## 项目结构

```
learn_py/
├── algorithm/                      # 算法实现
│   ├── heap_sort.py                #   最小堆（MinHeap）实现
│   └── quick_sort.py               #   快速排序（Hoare 分割）
├── learn_llm/                      # LLM 核心组件学习
│   ├── self-attention.py           #   Self-Attention / MHA / GQA 实现
│   ├── rl.py                       #   GRPO 损失函数实现
│   ├── GroupedQueryAttention.ipynb #   GQA 教程（含 MHA/MQA 对比）
│   ├── causal-lm.ipynb             #   Causal LM 微调流程（ELI5 数据集）
│   └── output_attentions_tutorial.ipynb  # Attention 可视化教程
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
手写实现三种注意力机制，便于对比学习：

| 类 | 机制 | K/V 复制方式 | 说明 |
|---|---|---|---|
| `SelfAttention` | 单头注意力 | — | 无 `o_proj`，`value_proj` 同时承担投影与输出角色 |
| `MultiHeadAttention` | 多头注意力（MHA） | — | 含 `o_proj`，对多头拼接结果做特征融合 |
| `GroupedQueryAttentionV1` | 分组查询注意力 | `repeat_interleave` | 直观写法，相邻 Q 头共享同一 KV 头 |
| `GroupedQueryAttentionV2` | 分组查询注意力 | `repeat_kv`（expand + reshape） | 更接近 HuggingFace 生产实现 |

> **GQA 关键点**：Q 头数 `H` 必须是 KV 头数 `H_kv` 的整数倍。由于 `torch.matmul` 要求非单例维度匹配，无法靠广播自动处理 `H ≠ H_kv`，必须先用 `repeat_interleave` 或 `repeat_kv` 把 K/V 在头维度显式复制到与 Q 相同的形状。

#### `rl.py`
GRPO（Group Relative Policy Optimization）损失函数实现：
- 组内 reward 标准化得到 group-relative advantage
- PPO 风格的 clipped objective：`min(ratio * adv, clamp(ratio) * adv)`
- 与参考模型的 KL 散度正则：`F.kl_div(log_probs, ref_log_probs, log_target=True)`

#### `GroupedQueryAttention.ipynb`
GQA 教程 notebook，对比 MHA / MQA / GQA 的差异，使用 `ModuleList` 风格组织多头投影。

#### `causal-lm.ipynb`
基于 HuggingFace `transformers` 的 Causal LM 微调流程，使用 ELI5 数据集，演示 tokenization、grouping、training 全流程。

#### `output_attentions_tutorial.ipynb`
GPT-2 的 `output_attentions` 配置演示，可视化每一层、每个注意力头的注意力分数矩阵。

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

本项目使用两套环境：

### 1. uv 虚拟环境（用于 `.py` 脚本）

```bash
# 安装 uv（如未安装）
pip install uv

# 同步依赖（会自动创建 .venv 并安装 pyproject.toml 中的依赖）
uv sync

# 运行脚本
uv run python algorithm/heap_sort.py
uv run python learn_llm/self-attention.py
```

要求 Python ≥ 3.13。

### 2. conda 环境（用于 `.ipynb` notebook）

```bash
# 创建环境
conda create -n myenv3.11 python=3.11
conda activate myenv3.11

# 安装 notebook 所需依赖
pip install torch transformers datasets accelerate ipywidgets jupyter openai
```

在 Jupyter 中选择 kernel `myenv3.11` 即可运行 notebook。

## 使用方式

```bash
# 运行算法
uv run python algorithm/heap_sort.py    # 输出堆排序结果
uv run python algorithm/quick_sort.py   # 输出快速排序结果

# 运行注意力实现（会打印三种注意力的输出张量）
uv run python learn_llm/self-attention.py

# 运行 GRPO 损失函数（仅定义，需配合实际模型使用）
uv run python learn_llm/rl.py

# 测试 ARK API（需先设置 ARK_API_KEY 环境变量）
$env:ARK_API_KEY = "your-api-key"        # PowerShell
uv run python agent/test-apikey.py
```

## 学习要点

本项目记录了以下关键知识点：

1. **数据结构**：堆的自底向上建堆、`_up`/`_down` 维护逻辑
2. **排序算法**：快排的 Hoare 分割与递归结构
3. **注意力机制**：从单头到 MHA、GQA、MQA 的演进，理解 `o_proj` 的必要性、`repeat_interleave` 的广播原理
4. **强化学习**：GRPO 的 group-relative advantage、PPO clipped objective、KL 正则
5. **LLM 工具链**：HuggingFace `transformers` 的 Causal LM 微调、`output_attentions` 可视化
6. **Python 语义**：引用语义、切片赋值、可变默认参数陷阱
7. **NumPy**：广播规则的维度对齐机制
