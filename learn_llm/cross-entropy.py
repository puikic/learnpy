import torch
import torch.nn.functional as F


def log_softmax(logits, dim=-1):
    # 数值稳定版 log_softmax：先减最大值，防止 exp 溢出
    # log_softmax(x)_i = x_i - max(x) - log(sum_j exp(x_j - max(x)))
    max_val = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_val
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))
    return shifted - log_sum_exp


def cross_entropy(logits, target, ignore_index=-100, reduction="mean"):
    """
    手写交叉熵损失（等价 F.cross_entropy）
    logits: (N, C) 未归一化分数
    target: (N,)   真实类别索引，可能含 ignore_index（表示 padding，不计入 loss）
    """
    # (N, C) 每个类别的 log 概率
    log_probs = log_softmax(logits, dim=-1)

    # 有效位置掩码：非 padding 的 token 才参与 loss
    valid_mask = (target != ignore_index)  # (N,)

    # gather 需要合法索引，先把 ignore 的位置临时替换成 0，避免越界
    safe_target = target.clone()
    safe_target[~valid_mask] = 0

    # 取出每个样本“正确类别”对应的 log 概率，再取负 -> 负对数似然 NLL
    # gather 沿 dim=1 按索引取值 -> (N, 1) -> squeeze -> (N,)
    nll = -log_probs.gather(dim=1, index=safe_target.unsqueeze(1)).squeeze(1)  # (N,)

    # padding 位置置零，不贡献 loss
    nll = nll * valid_mask

    if reduction == "mean":
        # 关键：只除以“有效 token 数”，不是总数；clamp 防止全 padding 时除零
        return nll.sum() / valid_mask.sum().clamp(min=1)
    elif reduction == "sum":
        return nll.sum()
    return nll  # reduction="none"


def causal_lm_loss(logits, labels, ignore_index=-100):
    """
    自回归语言模型的交叉熵：用第 t 个位置的输出预测第 t+1 个 token
    logits: (B, T, V)
    labels: (B, T)
    """
    # 错位对齐：logits 去掉最后一个，labels 去掉第一个
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

    # 展平成 (N, V) 和 (N,) 后复用上面的 cross_entropy
    B, Tm1, V = shift_logits.shape
    return cross_entropy(
        shift_logits.view(B * Tm1, V),
        shift_labels.view(B * Tm1),
        ignore_index=ignore_index,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 基础版：与 F.cross_entropy 对比，应几乎相等
    logits = torch.randn(5, 10)
    target = torch.randint(0, 10, (5,))
    print("basic  mine:", cross_entropy(logits, target).item(),
          " ref:", F.cross_entropy(logits, target).item())

    # 2) 含 ignore_index（padding）：也应与官方一致
    target2 = target.clone()
    target2[0] = -100
    target2[3] = -100
    print("ignore mine:", cross_entropy(logits, target2, ignore_index=-100).item(),
          " ref:", F.cross_entropy(logits, target2, ignore_index=-100).item())

    # 3) 因果 LM 错位 loss
    B, T, V = 2, 6, 20
    lm_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    print("causal lm loss:", causal_lm_loss(lm_logits, labels).item())
