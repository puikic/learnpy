import torch
import torch.nn.functional as F

def grpo_loss(
    model,
    ref_model,
    input_ids,        # [B, T]
    attention_mask,   # [B, T]
    rewards,          # [B]，同一 prompt 下的一组 reward
    old_log_probs,    # [B]
    epsilon=0.2,
    beta=0.01
):
    """
    B = group size（同一个 prompt 的多个 response）
    """

    # 1. 当前策略 log probs
    logits = model(input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    # token-level -> sequence-level
    # 假设 input_ids 就是 response tokens
    seq_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1).sum(dim=-1)   # [B]

    # 2. Group-relative advantage（GRPO 核心）
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False) + 1e-8
    advantages = (rewards - mean_r) / std_r   # [B]

    # 3. PPO-style ratio
    ratio = torch.exp(seq_log_probs - old_log_probs)  # [B]

    # 4. Clipped objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # 5. KL with reference model
    with torch.no_grad():
        ref_logits = ref_model(input_ids, attention_mask=attention_mask).logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    kl = F.kl_div(
        log_probs,
        ref_log_probs,
        log_target=True,
        reduction="batchmean"
    )

    # 6. Total loss
    loss = policy_loss + beta * kl
    return loss
