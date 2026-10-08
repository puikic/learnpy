import torch
import numpy as np 

# numpy 版
class RQVaeNumpy:
    def __init__(self, num_codebooks=4, codebook_size=512, dim=64):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim 
        self.codebooks = [
            np.random.randn(codebook_size, dim) for _ in range(num_codebooks)
        ]
    def forward(self, z):
        residual = z.copy()
        z_q = np.zeros_like(z)
        indices_list = []
        for d in range(self.num_codebooks):
            cb = self.codebooks[d] # (L, N)
            # 1.计算欧式距离
            # 利用广播直接先减后平方：(B, 1, N) - (1, L, N) -> (B, L, N)
            dist = np.sum((residual[:, None, :] - cb[None, :, :]) ** 2, axis=-1) # (B, L)
            # 2.匹配最近向量
            indices = np.argmin(dist, axis=-1) # (B,)
            quantized = cb[indices]
            # 3.更新残差和量化和
            residual = residual - quantized
            z_q = z_q + quantized 
            indices_list.append(indices)
        all_indices = np.stack(indices_list, axis=-1)
        return z_q, all_indices

class RQVaeTorch:
    def __init__(self, num_codebooks=4, codebook_size=512, dim=64):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim 
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)
        ])

    def forward(self, z):
        residual = z.clone()
        z_q = torch.zeros_like(z)
        indices_list = []
        for d in range(self.num_codebooks):
            emb_layer = self.codebooks[d]                   # 当前层 Embedding 模块
            cb_weight = emb_layer.weight                    # 提取权重矩阵 (L, N) 用于计算距离
            
            # 1. 计算欧氏距离
            # 等价：dist = torch.sum((residual.unsqueeze(1) - cb_weight.unsqueeze(0)) ** 2, dim=-1) # (B, L)
            dist = torch.sum((residual[:, None, :] - cb_weight[None, :, :]) ** 2, dim=-1) # (B, L)
            
            # 2. 匹配最近向量 (使用 Embedding 函数直接查表)
            indices = torch.argmin(dist, dim=-1)                  # (B,)
            quantized = emb_layer(indices)                        # (B, N) 通过 Embedding 提取

            # 3. 更新残差与量化和
            residual = residual - quantized
            z_q = z_q + quantized
            indices_list.append(indices)

        all_indices = torch.stack(indices_list, dim=-1)           # (B, D)

        # VQ/RQ-VAE 关键步：Straight-Through Estimator (STE) 复制梯度
        z_q_ste = z + (z_q - z).detach()

        return z_q_ste, all_indices