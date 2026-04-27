import torch.nn as nn


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model=128,
        n_heads=16,
        ffn_dim=256,
        n_layers=3,
        dropout=0.2,
    ):
        super().__init__()

        # We use standard Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",  # As specified in paper A.1.4
            batch_first=True,
        )

        # The paper mentions BatchNorm can be better,
        # but for simplicity, the vanilla encoder uses LayerNorm.
        # We can wrap these in a standard Encoder.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(self, x):
        # x: [Batch*M, N, D]
        return self.encoder(x)  # Output remains [Batch*M, N, D]
