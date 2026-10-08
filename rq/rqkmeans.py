import numpy as np

class RQKMeansNumPy:
    def __init__(self, num_codebooks=4, codebook_size=512, dim=64):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        
        self.codebooks_list = [
            np.random.randn(codebook_size, dim) for _ in range(num_codebooks)
        ]

    def predict(self, x: np.ndarray):
        """
        x: (B, dim)
        """
        residual = x.copy()
        x_q = np.zeros_like(x)
        indices_list = []
        layer_mses = []

        for d in range(self.num_codebooks):
            cb = self.codebooks_list[d]                            # (L, N)
            
            # 1. 广播计算欧氏距离平方：先减后平方 (B, 1, N) - (1, L, N) -> (B, L)
            dist = np.sum((residual[:, None, :] - cb[None, :, :]) ** 2, axis=-1)

            # 2. 匹配最近的 Cluster 中心
            indices = np.argmin(dist, axis=-1)                   # (B,)
            quantized = cb[indices]                              # (B, N)

            # 3. 更新残差与累计量化结果
            residual = residual - quantized
            x_q = x_q + quantized
            
            # 4. 记录当前残差的均方误差 (MSE)
            mse = np.mean(residual ** 2)
            layer_mses.append(mse)
            indices_list.append(indices)

        all_indices = np.stack(indices_list, axis=-1)            # (B, D)
        return x_q, all_indices, layer_mses

import torch
import torch.nn as nn

class RQKMeansPyTorch(nn.Module):
    def __init__(self, num_codebooks=4, codebook_size=512, dim=64):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        
        # nn.ModuleList + nn.Embedding 结构
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim) for _ in range(num_codebooks)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (B, dim)
        """
        residual = x.clone()
        x_q = torch.zeros_like(x)
        indices_list = []
        layer_mses = []

        # 推断/聚类预测不需要计算梯度
        with torch.no_grad():
            for d in range(self.num_codebooks):
                emb_layer = self.codebooks[d]
                cb_weight = emb_layer.weight                      # (L, N)
                
                # 1. 广播计算欧氏距离平方：先减后平方 (B, 1, N) - (1, L, N) -> (B, L)
                dist = torch.sum((residual[:, None, :] - cb_weight[None, :, :]) ** 2, dim=-1)

                # 2. 匹配最近的 Cluster 中心
                indices = torch.argmin(dist, dim=-1)              # (B,)
                quantized = emb_layer(indices)                    # (B, N)

                # 3. 更新残差与累计量化结果
                residual = residual - quantized
                x_q = x_q + quantized
                
                # 4. 记录当前残差的均方误差 (MSE)
                mse = torch.mean(residual ** 2).item()
                layer_mses.append(mse)
                indices_list.append(indices)

        all_indices = torch.stack(indices_list, dim=-1)           # (B, D)
        return x_q, all_indices, layer_mses