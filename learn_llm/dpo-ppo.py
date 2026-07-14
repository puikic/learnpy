import torch
import torch.nn.functional as F


def sequence_logprobs(logits, input_ids, loss_mask=None):
    """
    从 logits 计算每条序列的对数概率之和（token-level -> sequence-level）。
    logits:    (B, T, V) 模型输出
    input_ids: (B, T)    实际 token（这里简化为直接用 input_ids 当预测目标）
    loss_mask: (B, T)    1 表示计入（response 部分），0 表示忽略（prompt/padding）
    返回: (B,) 每条序列的 log prob 之和
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    # 取出每个位置"真实 token"的 log prob
    token_logp = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    if loss_mask is not None:
        token_logp = token_logp * loss_mask
    return token_logp.sum(dim=-1)  # (B,)


# ==========================================================
# DPO: Direct Preference Optimization
# 无需奖励模型 / 无需在线采样，直接用 (chosen, rejected) 偏好对训练
# ==========================================================
def dpo_loss(
    policy_chosen_logps,   # (B,) 当前策略对 chosen 的 log prob
    policy_rejected_logps, # (B,) 当前策略对 rejected 的 log prob
    ref_chosen_logps,      # (B,) 参考模型对 chosen 的 log prob（冻结）
    ref_rejected_logps,    # (B,) 参考模型对 rejected 的 log prob（冻结）
    beta=0.1,
):
    """
    DPO 核心：把"偏好"转成一个二分类逻辑回归。
    奖励隐式定义为 r = beta * log(policy / ref)，
    希望 chosen 的隐式奖励高于 rejected：
        loss = -log_sigmoid( beta * [ (πc-refc) - (πr-refr) ] )
    """
    # 每个样本相对参考模型的对数比值（log-ratio）
    pi_logratios = policy_chosen_logps - policy_rejected_logps      # 策略更偏好 chosen 多少
    ref_logratios = ref_chosen_logps - ref_rejected_logps          # 参考模型的偏好基线
    # 减去参考基线 -> "策略相比参考，额外偏好 chosen 多少"
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # 附带返回隐式奖励，便于监控 chosen/rejected 的奖励差距（reward margin）
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward


# ==========================================================
# PPO: Proximal Policy Optimization（clipped surrogate）
# RLHF 经典在线算法，用 clip 限制策略单步更新幅度
# ==========================================================
def ppo_policy_loss(
    logprobs,      # (B,) 或 (N,) 当前策略 log prob
    old_logprobs,  # (B,) 采样时旧策略的 log prob（固定）
    advantages,    # (B,) 优势函数（通常由 GAE 估计）
    epsilon=0.2,
):
    """
    PPO 截断目标：
        ratio = exp(logp - old_logp)              # 新旧策略概率比
        L = -E[ min(ratio*A, clip(ratio,1±eps)*A) ]
    clip 让 ratio 偏离 1 太远时梯度归零，防止一步更新过大导致策略崩坏。
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    # 取两者较小值 -> 悲观下界；负号因为要最大化目标 = 最小化 loss
    return -torch.min(unclipped, clipped).mean()


def ppo_value_loss(values, old_values, returns, epsilon=0.2):
    """
    PPO 价值函数损失（也做 clip，防止 value 更新过猛）。
    values / old_values / returns: (B,)
    """
    v_unclipped = (values - returns) ** 2
    v_clipped = (old_values + torch.clamp(values - old_values, -epsilon, epsilon) - returns) ** 2
    return 0.5 * torch.max(v_unclipped, v_clipped).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 4, 8, 50

    # ---- DPO 自测 ----
    # 模拟策略/参考模型对 chosen、rejected 两条回答的 logits
    pol_c = sequence_logprobs(torch.randn(B, T, V), torch.randint(0, V, (B, T)))
    pol_r = sequence_logprobs(torch.randn(B, T, V), torch.randint(0, V, (B, T)))
    ref_c = sequence_logprobs(torch.randn(B, T, V), torch.randint(0, V, (B, T)))
    ref_r = sequence_logprobs(torch.randn(B, T, V), torch.randint(0, V, (B, T)))
    loss, rc, rr = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    print("DPO loss:", loss.item(), "| reward margin:", (rc - rr).mean().item())

    # ---- PPO 自测 ----
    logp = torch.randn(B)
    old_logp = logp + 0.05 * torch.randn(B)  # 旧策略略有差异
    adv = torch.randn(B)
    ret = torch.randn(B)
    val = torch.randn(B)
    old_val = val + 0.05 * torch.randn(B)
    print("PPO policy loss:", ppo_policy_loss(logp, old_logp, adv).item())
    print("PPO value  loss:", ppo_value_loss(val, old_val, ret).item())
