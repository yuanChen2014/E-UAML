
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Embedding sizes (can be adjusted to match the manuscript configuration)
    D_FACE = 512
    D_VOICE = 2048
    D_EAR = 1024
    D_GAIT = 512
    D_FUSED = 1024


    class SimpleMLPEncoder(nn.Module):
        """Lightweight MLP encoder for vector-like modalities (e.g., voice, gait features)."""
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class SimpleConvEncoder(nn.Module):
        """Lightweight convolutional encoder for image-based modalities (e.g., face, ear)."""
        def __init__(self, in_channels: int, output_dim: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(64, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.features(x)  # (B, 64, 1, 1)
            h = h.view(h.size(0), -1)
            return self.fc(h)


    class ModalityEncoders(nn.Module):
        """Encoders for four biometric modalities: face, voice, ear, and gait."""
        def __init__(self) -> None:
            super().__init__()
            # Placeholders; can be replaced with ResNet, VGG, TimeSformer, etc.
            self.face_encoder = SimpleConvEncoder(in_channels=3, output_dim=D_FACE)
            self.ear_encoder = SimpleConvEncoder(in_channels=1, output_dim=D_EAR)
            self.voice_encoder = SimpleMLPEncoder(input_dim=128, output_dim=D_VOICE)
            self.gait_encoder = SimpleMLPEncoder(input_dim=128, output_dim=D_GAIT)

        def forward(self, x_face, x_voice, x_ear, x_gait):
            e_face = self.face_encoder(x_face)
            e_voice = self.voice_encoder(x_voice)
            e_ear = self.ear_encoder(x_ear)
            e_gait = self.gait_encoder(x_gait)
            return e_face, e_voice, e_ear, e_gait


    class ModalityAttention(nn.Module):
        """Multi-head self-attention over modality embeddings (M=4)."""
        def __init__(self, d_model: int, num_heads: int = 4) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(inplace=True),
                nn.Linear(d_model * 4, d_model),
            )
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor):
            """x: (B, M, D), where M is the number of modalities."""
            h, w = self.attn(x, x, x)
            h = self.ln1(x + h)
            z = self.ffn(h)
            z = self.ln2(h + z)
            return z, w


    class FusionTransformer(nn.Module):
        """Lightweight Transformer encoder for fusing two attended embeddings (X and Y)."""
        def __init__(self, d_model: int = D_FUSED, num_heads: int = 8, num_layers: int = 2) -> None:
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, z_x: torch.Tensor, z_y: torch.Tensor):
            """z_x, z_y: (B, D) fused embeddings for individuals X and Y.


            Returns fused embeddings e_fused_x, e_fused_y of shape (B, D).
            """
            seq = torch.stack([z_x, z_y], dim=1)  # (B, 2, D)
            h = self.encoder(seq)
            e_fused_x = h[:, 0, :]
            e_fused_y = h[:, 1, :]
            return e_fused_x, e_fused_y


    def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


    class EUAML(nn.Module):
        """Core E-UAML model integrating modality encoders, attention, and transformer fusion.


        This class is a compact reference implementation aligned with the manuscript architecture.
        """
        def __init__(self) -> None:
            super().__init__()
            self.encoders = ModalityEncoders()

            # Projection layers to a common fused dimension
            self.proj_face = nn.Linear(D_FACE, D_FUSED)
            self.proj_voice = nn.Linear(D_VOICE, D_FUSED)
            self.proj_ear = nn.Linear(D_EAR, D_FUSED)
            self.proj_gait = nn.Linear(D_GAIT, D_FUSED)

            self.modality_attn = ModalityAttention(d_model=D_FUSED, num_heads=4)
            self.fusion = FusionTransformer(d_model=D_FUSED, num_heads=8, num_layers=2)

        def encode_modalities(self, x_face, x_voice, x_ear, x_gait):
            e_f, e_v, e_a, e_b = self.encoders(x_face, x_voice, x_ear, x_gait)
            e_f = l2_normalize(e_f)
            e_v = l2_normalize(e_v)
            e_a = l2_normalize(e_a)
            e_b = l2_normalize(e_b)
            return e_f, e_v, e_a, e_b

        def fuse(self, e_f, e_v, e_a, e_b):
            z_f = self.proj_face(e_f)
            z_v = self.proj_voice(e_v)
            z_a = self.proj_ear(e_a)
            z_b = self.proj_gait(e_b)

            # Stack over modality dimension: (B, 4, D_FUSED)
            z_all = torch.stack([z_f, z_v, z_a, z_b], dim=1)
            z_attended, attn_weights = self.modality_attn(z_all)
            z_mean = z_attended.mean(dim=1)  # (B, D_FUSED)
            return z_mean, attn_weights

        def forward(self, batch):
            """Forward pass for a batch of paired inputs.


            Expected keys in `batch`:
"
            "            - face_x, voice_x, ear_x, gait_x
"
            "            - face_y, voice_y, ear_y, gait_y
"
            "        """
"
            "        # Encode X
"
            "        e_fx, e_vx, e_ax, e_bx = self.encode_modalities(
"
            "            batch['face_x'], batch['voice_x'], batch['ear_x'], batch['gait_x']
"
            "        )
"
            "        # Encode Y
"
            "        e_fy, e_vy, e_ay, e_by = self.encode_modalities(
"
            "            batch['face_y'], batch['voice_y'], batch['ear_y'], batch['gait_y']
"
            "        )
"
            "
"
            "        # Fuse modalities for X and Y
"
            "        z_x, attn_x = self.fuse(e_fx, e_vx, e_ax, e_bx)
"
            "        z_y, attn_y = self.fuse(e_fy, e_vy, e_ay, e_by)
"
            "
"
            "        # Transformer fusion on the pair (X, Y)
"
            "        e_fused_x, e_fused_y = self.fusion(z_x, z_y)
"
            "
"
            "        return {
"
            "            'e_fused_x': e_fused_x,
"
            "            'e_fused_y': e_fused_y,
"
            "            'attn_x': attn_x,
"
            "            'attn_y': attn_y,
"
            "        }
"
