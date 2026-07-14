import torch
import torch.nn as nn
import torch.nn.functional as F


def grpo_loss(
    model,
    ref_model,
    input_ids,        # (batch, seq_len)  batch = group size，同一个 prompt 采样出的多个 response
    attention_mask,   # (batch, seq_len)
    rewards,          # (batch,)          每个 response 拿到的打分
    old_log_probs,    # (batch,)          采样时那一版策略给出的整句 log prob（用于算 ratio）
    epsilon=0.2,      # PPO 裁剪范围
    beta=0.01,        # KL 惩罚系数
):
    # GRPO：不需要单独的 Critic/Value 网络，而是把“同一个 prompt 的一组 response”
    # 拿来互相比较，谁比这组的平均分高，谁就是好样本。

    # ==========================================
    # 步骤 1：当前策略给整句话打的 log prob
    # ==========================================
    logits = model(input_ids, attention_mask=attention_mask).logits  # (batch, seq_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)                         # (batch, seq_len, vocab)

    # token-level log prob -> sequence-level log prob
    # gather：在最后一维 vocab 上，把“实际生成的那个 token”对应的 log prob 抠出来
    # 这里假设 input_ids 本身就是 response 的 token 序列
    seq_log_probs = torch.gather(
        log_probs,                       # (batch, seq_len, vocab)
        dim=-1,
        index=input_ids.unsqueeze(-1),   # (batch, seq_len, 1)
    ).squeeze(-1).sum(dim=-1)            # (batch,)  每句话各 token log prob 求和 = 整句 log prob

    # ==========================================
    # 🌟 GRPO 核心：组内相对优势 (Group-relative Advantage)
    # ==========================================
    # 用这一组 reward 的均值和标准差做归一化：
    # 比组内平均好 -> advantage 为正 -> 鼓励；比平均差 -> 为负 -> 抑制。
    # 这一步替代了 PPO 里需要额外训练的 Value 网络。
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8       # +1e-8 防止整组分数相同时除以 0
    advantages = (rewards - mean_r) / std_r          # (batch,)

    # ==========================================
    # 步骤 3：PPO 式的重要性采样比率 (ratio)
    # ==========================================
    # ratio = 新策略概率 / 旧策略概率 = exp(新 log prob - 旧 log prob)
    # 衡量“现在的策略”相对“当初采样的策略”对这句话的偏好变化了多少
    ratio = torch.exp(seq_log_probs - old_log_probs)  # (batch,)

    # ==========================================
    # 步骤 4：裁剪目标 (Clipped Objective)
    # ==========================================
    # 裁剪的目的：不让策略一步更新迈得太大，训练更稳。
    # 取 unclipped 和 clipped 里更小(更保守)的那个。
    unclipped = ratio * advantages                                       # (batch,)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages  # (batch,)
    policy_loss = -torch.mean(torch.min(unclipped, clipped))             # 标量；负号是因为要最大化收益

    # ==========================================
    # 步骤 5：与参考模型的 KL 惩罚
    # ==========================================
    # 拉住策略别跑得离原始模型太远(防止“训崩”/胡说)。ref_model 不参与梯度，所以 no_grad。
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attention_mask).logits  # (batch, seq_len, vocab)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)                        # (batch, seq_len, vocab)

    # log_target=True 表示第二个参数传的已经是 log 概率
    kl = F.kl_div(
        log_probs,        # (batch, seq_len, vocab) 当前策略
        ref_log_probs,    # (batch, seq_len, vocab) 参考策略
        log_target=True,
        reduction="batchmean",
    )  # 标量

    # ==========================================
    # 步骤 6：总损失 = 策略损失 + KL 惩罚
    # ==========================================
    loss = policy_loss + beta * kl
    return loss


# ==========================================
# 一个可直接运行的最小示例：用一个假的语言模型跑通 grpo_loss
# ==========================================
class TinyLM(nn.Module):
    # 极简“语言模型”：把 token id 过 embedding，再线性映射回 vocab 维度当 logits。
    # 只是为了让 grpo_loss 能真正跑起来，不代表真实模型结构。
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)          # (batch, seq_len, hidden_dim)
        logits = self.head(h)              # (batch, seq_len, vocab)
        # 返回一个带 .logits 属性的对象，模仿 transformers 的输出接口
        return type("Output", (), {"logits": logits})


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, hidden_dim = 50, 8
    batch, seq_len = 4, 6                       # batch=4：同一个 prompt 采样出的 4 条 response

    model = TinyLM(vocab_size, hidden_dim)
    ref_model = TinyLM(vocab_size, hidden_dim)  # 参考模型：通常是训练前的原始模型

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))  # (batch, seq_len)
    attention_mask = torch.ones(batch, seq_len)                 # (batch, seq_len)
    rewards = torch.tensor([1.0, 0.2, -0.5, 0.8])               # (batch,) 每条 response 的打分
    old_log_probs = torch.randn(batch)                          # (batch,) 采样时的整句 log prob

    loss = grpo_loss(model, ref_model, input_ids, attention_mask, rewards, old_log_probs)
    print("GRPO loss:", loss.item())
