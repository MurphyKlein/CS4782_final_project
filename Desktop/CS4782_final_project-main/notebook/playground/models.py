import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, max_seq_len=512):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_len, d_model)
        max_patches = (max_seq_len - patch_len) // stride + 2
        self.position_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)
        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :x.size(1), :]
        return x


class PatchTST(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_heads=16, n_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.patch_len = patch_len

        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, seq_len)
        n_patches = (seq_len - patch_len) // stride + 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.bn = nn.BatchNorm1d(d_model)
        self.forecast_head = nn.Linear(n_patches * d_model, pred_len)
        self.pretrain_head = nn.Linear(d_model, patch_len)
        self.d_model = d_model
        self.N = n_patches

    def _instance_norm(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        return (x - mean) / std, mean, std

    def _instance_denorm(self, x, mean, std):
        return x * std[:, :, :] + mean[:, :, :]

    def _encode(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L)
        z = self.patch_embed(x)
        z = self.transformer(z)
        z = self.bn(z.reshape(-1, self.d_model)).reshape(B * C, self.N, self.d_model)
        return z, B, C

    def forecast(self, x):
        x, mean, std = self._instance_norm(x)
        z, B, C = self._encode(x)
        z = z.reshape(B * C, -1)
        out = self.forecast_head(z)
        out = out.reshape(B, C, self.pred_len)
        out = out.permute(0, 2, 1)
        out = self._instance_denorm(out, mean, std)
        return out

    def pretrain(self, x, mask_ratio=0.4):
        x, _, _ = self._instance_norm(x)
        B, L, C = x.shape
        xr = x.permute(0, 2, 1).reshape(B * C, L)
        xpad = torch.cat([xr, xr[:, -1:].expand(-1, self.patch_embed.stride)], dim=1)
        patches = self.patch_embed.value_embedding(
            xpad.unfold(1, self.patch_len, self.patch_embed.stride)
        )

        BC, N, _ = patches.shape
        n_mask = int(N * mask_ratio)
        noise = torch.rand(BC, N, device=x.device)
        ids_sort = noise.argsort(dim=1)
        mask = torch.zeros(BC, N, device=x.device)
        mask.scatter_(1, ids_sort[:, :n_mask], 1)

        pos = self.patch_embed.position_embedding[:, :N, :]
        inp = patches * (1 - mask.unsqueeze(-1)) + pos
        z = self.transformer(inp)
        z = self.bn(z.reshape(-1, self.d_model)).reshape(BC, N, self.d_model)
        recon = self.pretrain_head(z)
        target = xpad.unfold(1, self.patch_len, self.patch_embed.stride)
        loss = ((recon - target) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * self.patch_len + 1e-8)
        return loss

    def forward(self, x, mode="forecast"):
        if mode == "forecast":
            return self.forecast(x)
        if mode == "pretrain":
            return self.pretrain(x)
        raise ValueError(f"Unknown mode: {mode}")
