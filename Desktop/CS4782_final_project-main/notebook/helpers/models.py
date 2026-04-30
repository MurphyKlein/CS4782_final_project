import torch
import torch.nn as nn

from .datasets import WeatherDataset


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = torch.mean(x, dim=2, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps).detach()

            x = x - self.mean
            x = x / self.stdev
            x = x * self.affine.unsqueeze(0).unsqueeze(-1) + self.beta.unsqueeze(0).unsqueeze(-1)
        elif mode == "denorm":
            x = x - self.beta.unsqueeze(0).unsqueeze(-1)
            x = x / self.affine.unsqueeze(0).unsqueeze(-1)
            x = x * self.stdev
            x = x + self.mean

        return x


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


class PatchTST_SelfSupervised(nn.Module):
    def __init__(self, num_features, seq_len=512, patch_len=12, stride=12,
                 d_model=128, n_heads=16, n_layers=3, d_ff=256, dropout=0.2, mask_ratio=0.4):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        self.revin = RevIN(num_features)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, max_seq_len=seq_len)
        self.num_patches = (seq_len - patch_len) // stride + 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reconstruction_head = nn.Linear(d_model, patch_len)

    def forward(self, x):
        B, M, L = x.shape
        x = self.revin(x, mode="norm")
        x_independent = x.reshape(B * M, L)

        padded_x = self.patch_embedding.padding_patch_layer(x_independent.unsqueeze(1))
        raw_patches = padded_x.unfold(dimension=-1, size=self.patch_len, step=self.stride).squeeze(1)

        mask = torch.rand(B * M, raw_patches.size(1), device=x.device) < self.mask_ratio
        masked_patches = raw_patches.clone()
        masked_patches[mask] = 0

        embedded_patches = self.patch_embedding.value_embedding(masked_patches)
        embedded_patches = embedded_patches + self.patch_embedding.position_embedding[:, :embedded_patches.size(1), :]

        transformer_out = self.transformer_encoder(embedded_patches)
        reconstructed_patches = self.reconstruction_head(transformer_out)

        return reconstructed_patches, raw_patches, mask


class PatchTST_Supervised(nn.Module):
    def __init__(self, num_features, seq_len=512, pred_len=96, patch_len=12, stride=12,
                 d_model=128, n_heads=16, n_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.revin = RevIN(num_features)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, max_seq_len=seq_len)
        self.num_patches = (seq_len - patch_len) // stride + 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_dropout = nn.Dropout(dropout)
        self.linear_head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):
        B, M, L = x.shape
        x = self.revin(x, mode="norm")
        x_independent = x.reshape(B * M, L)
        embedded_patches = self.patch_embedding(x_independent)
        transformer_out = self.transformer_encoder(embedded_patches)
        flattened_out = transformer_out.reshape(B * M, -1)
        flattened_out = self.head_dropout(flattened_out)
        predictions = self.linear_head(flattened_out)
        predictions = predictions.reshape(B, M, self.pred_len)
        predictions = self.revin(predictions, mode="denorm")
        return predictions

    def load_pre_trained_backbone(self, pre_trained_model_dict, freeze_backbone=True):
        filtered_dict = {k: v for k, v in pre_trained_model_dict.items() if "reconstruction_head" not in k}
        self.load_state_dict(filtered_dict, strict=False)

        if freeze_backbone:
            for name, param in self.named_parameters():
                if "linear_head" not in name:
                    param.requires_grad = False
