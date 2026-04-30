import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        m_feat,
        l_seq,
        t_pred,
        p_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        n_layers=3,
    ):
        super().__init__()
        self.m_feat = m_feat
        self.t_pred = t_pred

        # 1. Patching Logic [cite: 139]
        self.p_len = p_len
        self.stride = stride
        self.n_patches = (max(l_seq, p_len) - p_len) // stride + 1
        self.n_patches += 1  # Accounting for the padding mentioned in paper

        # 2. Embedding Layers [cite: 144]
        self.patch_embedding = nn.Linear(p_len, d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )

        # 3. Transformer Backbone [cite: 143]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 4. Forecasting Head
        self.head = nn.Linear(d_model * self.n_patches, t_pred)

    def instance_norm(self, x):
        mu = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(
            x.var(1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return (x - mu) / stdev, mu, stdev

    def forward(self, x):
        # x shape: [Batch, L, M]
        B, L, M = x.shape

        # A. Instance Normalization [cite: 153, 154]
        x, mu, stdev = self.instance_norm(x)

        # B. Channel Independence & Patching [cite: 133, 134]
        # Reshape to [Batch * M, L] then pad and unfold to [Batch * M, N, P]
        x = x.permute(0, 2, 1).reshape(B * M, L)
        last_val = x[:, -1:].repeat(1, self.stride)
        x = torch.cat([x, last_val], dim=-1)
        x = x.unfold(
            dimension=-1, size=self.p_len, step=self.stride
        )  # [Batch*M, N, P]

        # C. Embedding & Transformer [cite: 144, 150]
        x = self.patch_embedding(x) + self.pos_embedding
        x = self.transformer_encoder(x)  # [Batch*M, N, D]

        # D. Flatten & Head
        x = x.reshape(B * M, -1)  # Flatten patches
        x = self.head(x)  # [Batch*M, T]

        # E. Reshape back and Denormalize
        x = x.reshape(B, M, self.t_pred).permute(0, 2, 1)
        x = x * stdev + mu

        return x
