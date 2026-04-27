import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, p_len, d_model, n_patches, dropout=0.2):
        super().__init__()
        # 1. Linear Projection: [Batch*M, N, P] -> [Batch*M, N, D]
        self.value_embedding = nn.Linear(p_len, d_model)

        # 2. Learnable Positional Encoding: [1, N, D]
        self.position_embedding = nn.Parameter(
            torch.zeros(1, n_patches, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is the patched output from Step 4: [Batch*M, N, P]
        x = self.value_embedding(x)  # Project to d_model (128)
        x = x + self.position_embedding  # Add position info
        return self.dropout(x)
