from torch import nn

class RMSNorm(nn.Module):
    """RMSNorm：不减均值，只按均方根缩放，仅学习逐特征 gamma。

    y_i = gamma_i · x_i / sqrt(mean(x^2) + eps)。
    eps 加在开方内部；半精度输入建议统计量升到 float32 再算。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(mean_square + self.eps)
