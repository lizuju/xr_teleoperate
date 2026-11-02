"""ACT (Zhao et al. 2023) for the R1_A7 + O6 dataset: four cameras and joints in, joint chunks out.

Training and inference only; nothing here talks to DDS, cameras or the robot.
Needs torchvision, so it runs in .venv-train, not in the teleop environment.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_SCHEMA = "r1_act_checkpoint_v1"


def sine_position(height, width, dim, device):
    """DETR 2D sine position for one camera's feature map, flattened to (h*w, dim)."""
    half = dim // 2
    y, x = torch.meshgrid(torch.linspace(0, 2 * math.pi, height, device=device),
                          torch.linspace(0, 2 * math.pi, width, device=device), indexing="ij")
    frequency = 10000 ** (2 * (torch.arange(half, device=device, dtype=torch.float32) // 2) / half)
    parts = []
    for axis in (y, x):
        phase = axis.flatten()[:, None] / frequency
        parts.append(torch.stack((phase[:, 0::2].sin(), phase[:, 1::2].cos()), dim=2).flatten(1))
    return torch.cat(parts, dim=1)


class ACTPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_cameras, chunk_size=40, hidden_dim=512, nheads=8,
                 dim_feedforward=3200, encoder_layers=4, decoder_layers=1, vae_encoder_layers=4,
                 latent_dim=32, dropout=0.1, pretrained=True):
        super().__init__()
        self.config = dict(state_dim=state_dim, action_dim=action_dim, num_cameras=num_cameras,
                           chunk_size=chunk_size, hidden_dim=hidden_dim, nheads=nheads,
                           dim_feedforward=dim_feedforward, encoder_layers=encoder_layers,
                           decoder_layers=decoder_layers, vae_encoder_layers=vae_encoder_layers,
                           latent_dim=latent_dim, dropout=dropout)
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
                            norm_layer=FrozenBatchNorm2d)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.camera_embed = nn.Parameter(torch.zeros(num_cameras, hidden_dim))
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

        def encoder(layers):
            layer = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        # Style encoder: only used while training, to summarise the demonstrated chunk.
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.vae_state = nn.Linear(state_dim, hidden_dim)
        self.vae_action = nn.Linear(action_dim, hidden_dim)
        self.vae_position = nn.Parameter(torch.zeros(1, chunk_size + 2, hidden_dim))
        self.vae_encoder = encoder(vae_encoder_layers)
        self.vae_output = nn.Linear(hidden_dim, latent_dim * 2)

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.token_position = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.encoder = encoder(encoder_layers)
        self.queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        layer = nn.TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        for parameter in (self.camera_embed, self.vae_cls, self.vae_position, self.token_position, self.queries):
            nn.init.normal_(parameter, std=0.02)

    def encode_images(self, images):
        batch, cameras = images.shape[:2]
        features = (images - self.image_mean) / self.image_std
        features = self.image_proj(self.backbone(features.flatten(0, 1)))
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2).reshape(batch, cameras, height * width, -1)
        position = sine_position(height, width, tokens.shape[-1], tokens.device).to(tokens.dtype)
        return (tokens + position + self.camera_embed[:, None]).flatten(1, 2)

    def encode_style(self, qpos, actions, is_pad):
        batch = qpos.shape[0]
        tokens = torch.cat((self.vae_cls.expand(batch, -1, -1), self.vae_state(qpos)[:, None],
                            self.vae_action(actions)), dim=1) + self.vae_position
        padding = torch.cat((torch.zeros(batch, 2, dtype=torch.bool, device=is_pad.device), is_pad), dim=1)
        mu, logvar = self.vae_output(self.vae_encoder(tokens, src_key_padding_mask=padding)[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, images, qpos, actions=None, is_pad=None):
        """With actions: training pass, (pred, mu, logvar). Without: the deployed z=0 pass."""
        if actions is None:
            mu = logvar = None
            latent = torch.zeros(qpos.shape[0], self.latent_dim, device=qpos.device, dtype=qpos.dtype)
        else:
            mu, logvar = self.encode_style(qpos, actions, is_pad)
            latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        tokens = torch.stack((self.latent_proj(latent), self.state_proj(qpos)), dim=1) + self.token_position
        memory = self.encoder(torch.cat((tokens, self.encode_images(images)), dim=1))
        decoded = self.decoder(self.queries.expand(qpos.shape[0], -1, -1), memory)
        return self.action_head(decoded), mu, logvar

    def parameter_groups(self, lr, lr_backbone):
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        chosen = {id(p) for p in backbone}
        return [{"params": [p for p in self.parameters() if id(p) not in chosen and p.requires_grad], "lr": lr},
                {"params": backbone, "lr": lr_backbone}]


def masked_l1(pred, target, is_pad):
    mask = (~is_pad)[..., None]
    return ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[-1]).clamp(min=1)


def act_loss(pred, target, is_pad, mu, logvar, kl_weight):
    l1 = masked_l1(pred, target, is_pad)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
    return l1 + kl_weight * kl, l1, kl


def augment_images(images, max_shift=4, jitter=0.2):
    """Per-image small shifts and brightness/contrast. Each camera moves independently."""
    batch, cameras, channels, height, width = images.shape
    flat = images.view(-1, channels, height, width)
    count = flat.shape[0]
    device, dtype = flat.device, flat.dtype
    if max_shift > 0:
        offset = torch.randint(-max_shift, max_shift + 1, (count, 2), device=device).to(dtype)
        theta = torch.zeros(count, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, 0, 2] = offset[:, 0] * 2 / width
        theta[:, 1, 2] = offset[:, 1] * 2 / height
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        flat = F.grid_sample(flat, grid, padding_mode="border", align_corners=False)
    if jitter > 0:
        brightness = 1 + (torch.rand(count, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 2 * jitter
        contrast = 1 + (torch.rand(count, 1, 1, 1, device=device, dtype=dtype) - 0.5) * 2 * jitter
        mean = flat.mean(dim=(1, 2, 3), keepdim=True)
        flat = ((flat - mean) * contrast + mean) * brightness
    return flat.clamp(0, 1).view(batch, cameras, channels, height, width)


def save_checkpoint(path, model, normalization, dataset_info, step, optimizer=None, metrics=None):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save({"schema": CHECKPOINT_SCHEMA, "config": model.config,
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "normalization": normalization, "dataset": dataset_info, "step": int(step),
                "optimizer": None if optimizer is None else optimizer.state_dict(),
                "metrics": metrics or {}}, temporary)
    os.replace(temporary, path)


def read_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: unknown checkpoint schema {data.get('schema')!r}")
    return data


def load_policy(path, device="cpu"):
    data = read_checkpoint(path, device)
    model = ACTPolicy(**data["config"], pretrained=False).to(device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, data


class ChunkPredictor:
    """Deployed inference: real-unit qpos and uint8 RGB in, real-unit action chunk out."""

    def __init__(self, model, normalization, device):
        self.model, self.device = model, device
        self.stats = {key: torch.as_tensor(np.asarray(normalization[key], dtype=np.float32), device=device)
                      for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")}

    @torch.inference_mode()
    def __call__(self, images, qpos):
        images = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2)[None].float() / 255
        qpos = torch.from_numpy(qpos).to(self.device)[None]
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            chunk = self.model(images, qpos)[0][0]
        return (chunk.float() * self.stats["action_std"] + self.stats["action_mean"]).cpu().numpy()
