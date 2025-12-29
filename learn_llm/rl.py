import torch
import torch.nn as nn
import torch.nn.functional as F

def grpo_loss_step(
    current_logits,     # 模型当前的输出 [B*G, Seq_Len, Vocab]
    old_log_probs,      # 采样时的 Log Probs [B*G, Seq_Len]
    ref_log_probs,      # 参考模型(SFT)的 Log Probs [B*G, Seq_Len]
    rewards,            # 标量奖励 [B*G] (注意：这里是打平后的 batch)
    mask,               #掩码，只计算回答部分的 Loss [B*G, Seq_Len]
    group_size=4,       # G: 一组采样的个数
    epsilon=0.2,        # PPO clip ratio
    beta=0.01           # KL penalty 系数
):
    """
    GRPO 核心计算逻辑
    B: Batch Size (Prompts)
    G: Group Size (每个 Prompt 采样的回复数)
    """
    
    # 1. 整理维度，恢复 [B, G] 的结构以计算 Group 统计量
    # 假设输入已经是 [B*G, ...]，我们需要先 reshape 回去处理 Advantage
    # rewards: [B*G] -> [B, G]
    num_total = rewards.size(0)
    batch_size = num_total // group_size
    rewards_grouped = rewards.view(batch_size, group_size)
    
    # 2. 计算 Group Relative Advantage (核心!)
    # 计算组内均值和方差
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True)
    
    # 标准化: (r - mean) / (std + eps)
    advantages = (rewards_grouped - group_mean) / (group_std + 1e-8)
    
    # 将 Advantage 拉回 [B*G, 1] 以便后续广播到每个 Token
    advantages = advantages.view(-1, 1) 
    
    # 3. 计算 Current Log Probs
    # current_logits: [N, L, V] -> 选出对应 Token 的概率
    # 这里假设输入 input_ids 已经被用来 gather 对应的 log_probs 了
    # 为了简化代码，假设 current_logits 已经是 gather 后的 current_log_probs [B*G, Seq_Len]
    # 如果面试官要求严谨，需写出 gather 过程：
    # current_log_probs = torch.gather(current_logits.log_softmax(-1), -1, input_ids.unsqueeze(-1)).squeeze(-1)
    
    current_log_probs = current_logits # 假设传入的就是 log_probs
    
    # 4. 计算 PPO Loss 部分
    # Ratio = exp(new - old)
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # Surrogate Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2) # 负号因为我们要最小化 Loss
    
    # 5. 计算 KL Divergence (近似方式)
    # D_KL = exp(ref - curr) - (ref - curr) - 1 ... 或者简单的 log_p - log_ref
    # GRPO 论文通常直接用无偏估计： log_p - log_ref
    # approx_kl = current_log_probs - ref_log_probs # 这是一种写法
    # 更标准的 KL 实现:
    per_token_kl = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    
    # 6. 组合 Loss (Mask 掉 padding 和 prompt 部分)
    loss = policy_loss + beta * per_token_kl
    
    # 应用 Mask (只对生成的 Tokens 计算 Loss)
    loss = (loss * mask).sum() / mask.sum()
    
    return loss