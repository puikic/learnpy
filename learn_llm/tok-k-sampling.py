import torch
import torch.nn.functional as F

def top_k_sampling(logits, k, temperature=1.0):
    # logits: [batch_size, vocab_size]
    logits = logits / temperature

    topk_logits, topk_ids = torch.topk(logits, k, dim=-1)
    probs = F.softmax(topk_logits, dim=-1)

    sample_idx = torch.multinomial(probs, num_samples=1)
    next_token = torch.gather(topk_ids, dim=-1, index=sample_idx)

    return next_token.squeeze(-1)