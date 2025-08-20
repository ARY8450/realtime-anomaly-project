import math
import torch
from typing import Optional, Dict, Any
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dim: last column remains zero for cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # register buffer (will create attribute 'pe' in module state)
        self.register_buffer("pe", pe)
        # keep a typed alias to use for indexing and to satisfy the type-checker
        self._pe = pe  # type: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        seq_len = x.size(1)
        # use the typed alias for slicing to avoid static type confusion
        return x + self._pe[:, :seq_len]


class TransformerAutoencoder(nn.Module):
    """
    Sequence-to-sequence Transformer Autoencoder.
    Input: (batch, seq_len, input_dim)
    Output: reconstructed sequence of same shape.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # input projection and output projection
        self.encoder_in = nn.Linear(input_dim, d_model)
        self.decoder_out = nn.Linear(d_model, input_dim)

        # positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # transformer encoder / decoder (use batch_first for (B,S,D))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # small init
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns encoder memory (B, S, d_model) and optionally pooled latent (B, d_model).
        """
        # x: (B, S, input_dim)
        x_emb = self.encoder_in(x)  # (B, S, d_model)
        x_emb = self.pos_enc(x_emb)
        x_emb = self.dropout(x_emb)
        memory = self.encoder(x_emb)  # (B, S, d_model)
        return memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input sequence.
        """
        # x: (B, S, input_dim)
        memory = self.encode(x)  # (B, S, d_model)

        # decoder target: zeros with same shape as memory (can be learned start tokens if desired)
        tgt = torch.zeros_like(memory)
        # optionally add positional encoding to tgt (helps decoder)
        tgt = self.pos_enc(tgt)
        tgt = self.dropout(tgt)

        decoded = self.decoder(tgt=tgt, memory=memory)  # (B, S, d_model)
        out = self.decoder_out(decoded)  # (B, S, input_dim)
        return out

    def latent_vector(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        Utility to get a fixed-size latent vector from sequence.
        mode: "mean" or "cls" (first token) or "max"
        """
        memory = self.encode(x)  # (B, S, d_model)
        if mode == "mean":
            return memory.mean(dim=1)
        if mode == "max":
            return memory.max(dim=1)[0]
        return memory[:, 0, :]  # cls-like


class Discriminator(nn.Module):
    """
    A small 1D-convolutional discriminator for sequences.
    Input: (B, S, input_dim)
    Output: logits (B,)
    """

    def __init__(self, input_dim: int, hidden: int = 64, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(n_layers):
            out_ch = hidden * (2 ** i) if i > 0 else hidden
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_ch // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, input_dim)
        returns logits: (B,)
        """
        # conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)  # (B, input_dim, seq_len)
        h = self.conv(x)  # (B, C_last, L_reduced)
        # global average pooling over length
        h = h.mean(dim=2)  # (B, C_last)
        logits = self.fc(h).squeeze(-1)  # (B,)
        return logits


class TAEGAN(nn.Module):
    """
    Wrapper that contains the Transformer Autoencoder (generator) and a discriminator.
    Provides convenience losses commonly used in training.
    """

    def __init__(
        self,
        input_dim: int,
        ae_kwargs: Optional[Dict[str, Any]] = None,
        disc_kwargs: Optional[Dict[str, Any]] = None,
        adv_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        if ae_kwargs is None:
            ae_kwargs = {}
        if disc_kwargs is None:
            disc_kwargs = {}
        self.generator = TransformerAutoencoder(input_dim=input_dim, **ae_kwargs)
        self.discriminator = Discriminator(input_dim=input_dim, **disc_kwargs)
        self.adv_weight = adv_weight
        self.recon_weight = recon_weight
        self.recon_loss_fn = nn.MSELoss(reduction="mean")
        self.adv_loss_fn = nn.BCEWithLogitsLoss()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def forward(self, x: torch.Tensor):
        """
        Forward passes generator and discriminator.
        Returns a dict with recon, disc_real_logits, disc_fake_logits
        """
        recon = self.reconstruct(x)
        disc_real = self.discriminator(x)
        disc_fake = self.discriminator(recon.detach())
        return {"recon": recon, "disc_real": disc_real, "disc_fake": disc_fake}

    def generator_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combined generator (autoencoder) loss: reconstruction + adversarial (fool disc)
        """
        recon = self.reconstruct(x)
        # reconstruction loss
        recon_loss = self.recon_loss_fn(recon, x)
        # adversarial loss: want discriminator to predict real for recon
        logits = self.discriminator(recon)
        target_real = torch.ones_like(logits)
        adv_loss = self.adv_loss_fn(logits, target_real)
        return self.recon_weight * recon_loss + self.adv_weight * adv_loss

    def discriminator_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard discriminator loss distinguishing real and reconstructed sequences.
        """
        with torch.no_grad():
            recon = self.reconstruct(x)
        real_logits = self.discriminator(x)
        fake_logits = self.discriminator(recon)
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)
        loss_real = self.adv_loss_fn(real_logits, real_targets)
        loss_fake = self.adv_loss_fn(fake_logits, fake_targets)
        return 0.5 * (loss_real + loss_fake)


if __name__ == "__main__":
    # small smoke test
    B, S, F = 4, 128, 3
    model = TAEGAN(input_dim=F)
    x = torch.randn(B, S, F)
    out = model(x)
    print("recon shape:", out["recon"].shape)
    print("disc_real logits shape:", out["disc_real"].shape)
    print("disc_fake logits shape:", out["disc_fake"].shape)
    print("gen loss:", model.generator_loss(x).item())
    print("disc loss:", model.discriminator_loss(x).item())