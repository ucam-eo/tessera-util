import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
from typing import List, Tuple, Optional, Union, Dict
import logging
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ================================
# Multi-temporal Configuration
# ================================

# Maximum number of time steps
MAX_TIMESTEPS = 3 # original Prithvi only uses 3
PRETRAINED_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
REAL_DATA_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]

# ================================
# Pretrained Weights Configuration
# ================================

PRETRAINED_WEIGHTS = {
    "prithvi_eo_v2_600_tl": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        "hf_hub_filename": "Prithvi_EO_V2_600M_TL.pt",
    },
    "prithvi_eo_v2_300_tl": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        "hf_hub_filename": "Prithvi_EO_V2_300M_TL.pt",
    },
    "prithvi_eo_v2_600": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "hf_hub_filename": "Prithvi_EO_V2_600M.pt",
    },
    "prithvi_eo_v2_300": {
        "hf_hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "hf_hub_filename": "Prithvi_EO_V2_300M.pt",
    },
}

# ================================
# Prithvi ViT Core Components
# ================================

def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 3D sin/cos positional embeddings."""
    assert embed_dim % 16 == 0
    
    t_size, h_size, w_size = grid_size
    
    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4
    
    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))
    
    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)
    
    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)
    
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sincos positional embedding from grid."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    """Modified torch version of get_1d_sincos_pos_embed_from_grid()."""
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class PatchEmbed(nn.Module):
    """3D version of patch embedding"""
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (1, 224, 224),
        patch_size: Tuple[int, int, int] = (1, 16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert all(g >= 1 for g in self.grid_size), "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        if T % self.patch_size[0] != 0 or H % self.patch_size[1] != 0 or W % self.patch_size[2] != 0:
            warnings.warn(f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}.")
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x

class TemporalEncoder(nn.Module):
    """Temporal encoding for time-location coordinates"""
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: int = None):
        """
        Args:
            temporal_coords: year and day-of-year info with shape (B, T, 2).
            tokens_per_frame: number of tokens for each frame in the sample.
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = _get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding

class LocationEncoder(nn.Module):
    """Location encoding for lat-lon coordinates"""
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = _get_1d_sincos_embed_from_grid_torch(
                self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(
                self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)
        return embedding

class Attention(nn.Module):
    """Multi-head self-attention module compatible with Prithvi weights"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=proj_drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    """MLP module"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class PrithviViT(nn.Module):
    """Prithvi ViT Encoder for Multi-temporal Data"""
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int, int]] = (1, 16, 16),
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type = nn.LayerNorm,
        coords_encoding: List[str] = None,
        coords_scale_learn: bool = False,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
            
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        
        # Coordinate encodings
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)
        
        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        # Transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=True, 
                norm_layer=norm_layer, 
                drop_path=dpr[i],
                attn_drop=attn_drop,
                proj_drop=proj_drop
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize positional embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding weights
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)
    
    def interpolate_pos_encoding(self, sample_shape: Tuple[int, int, int]):
        """Interpolate positional encoding for different input sizes"""
        t, h, w = sample_shape
        t_patches = t // self.patch_embed.patch_size[0]
        h_patches = h // self.patch_embed.patch_size[1]
        w_patches = w // self.patch_embed.patch_size[2]

        if [t_patches, h_patches, w_patches] == self.patch_embed.grid_size:
            return self.pos_embed
        
        # For different temporal dimensions, re-compute position embedding
        if t_patches != self.patch_embed.grid_size[0]:
            new_grid_size = (t_patches, self.patch_embed.grid_size[1], self.patch_embed.grid_size[2])
            new_pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], new_grid_size, add_cls_token=True)
            new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0).to(self.pos_embed.device)
            return new_pos_embed
        
        return self.pos_embed
    
    def forward_features(
        self, 
        x: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        location_coords: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # Add time dimension
            x = x.unsqueeze(2)
        
        sample_shape = x.shape[-3:]
        
        # Embed patches
        x = self.patch_embed(x)
        
        # Interpolate positional embeddings
        pos_embed = self.interpolate_pos_encoding(sample_shape)
        
        # Add positional embeddings without cls token
        x = x + pos_embed[:, 1:, :]
        
        # Add temporal encoding if available
        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        
        # Add location encoding if available
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding
        
        # Append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x.clone())
        
        x = self.norm(x)
        out[-1] = x
        return out
    
    def prepare_features_for_image_model(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare features for image model"""
        out = []
        effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
        
        for x in features:
            x_no_token = x[:, 1:, :]  # Remove cls token
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            
            # Reshape to (batch, channels, height, width)
            # For multi-temporal, we flatten time into channels
            # Use reshape instead of view to handle non-contiguous tensors
            encoded = x_no_token.permute(0, 2, 1).reshape(
                x_no_token.shape[0], 
                self.embed_dim * effective_time_dim, 
                h, 
                h
            )
            out.append(encoded)
        return out

# ================================
# Checkpoint Loading Functions
# ================================

def select_patch_embed_weights(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    pretrained_bands: List[int],
    model_bands: List[int],
    encoder_only: bool = True
) -> Dict[str, torch.Tensor]:
    """Select patch embedding weights for specific bands"""
    
    # Get the correct key for patch embedding
    if encoder_only:
        patch_embed_key = "patch_embed.proj.weight"
    else:
        patch_embed_key = "encoder.patch_embed.proj.weight"
    
    if patch_embed_key not in state_dict:
        logger.warning(f"Key {patch_embed_key} not found in state dict")
        return state_dict
    
    # If bands match, no need to select
    if pretrained_bands == model_bands:
        return state_dict
    
    # Select appropriate band weights
    pretrained_weight = state_dict[patch_embed_key]
    model_in_chans = len(model_bands)
    
    # Create new weight tensor
    weight_shape = list(pretrained_weight.shape)
    weight_shape[1] = model_in_chans
    new_weight = torch.zeros(weight_shape, dtype=pretrained_weight.dtype)
    
    # Copy weights for matching bands
    for i, band in enumerate(model_bands):
        if band in pretrained_bands:
            pretrained_idx = pretrained_bands.index(band)
            new_weight[:, i] = pretrained_weight[:, pretrained_idx]
        else:
            # Initialize randomly for new bands
            logger.warning(f"Band index {band} not in pretrained bands, initializing randomly")
            new_weight[:, i] = torch.randn_like(new_weight[:, i]) * 0.02
    
    state_dict[patch_embed_key] = new_weight
    return state_dict

def checkpoint_filter_fn_vit(
    state_dict: Dict[str, torch.Tensor],
    model: PrithviViT,
    pretrained_bands: List[int],
    model_bands: List[int]
) -> Dict[str, torch.Tensor]:
    """Filter checkpoint for ViT encoder only model"""
    
    clean_dict = {}
    for k, v in state_dict.items():
        # Remove _timm_module prefix if present
        if "_timm_module." in k:
            k = k.replace("_timm_module.", "")
        
        # Skip decoder weights
        if "decoder" in k or "_dec" in k or k == "mask_token":
            continue
        
        # Skip encodings if not used
        if not model.temporal_encoding and "temporal_embed" in k:
            continue
        if not model.location_encoding and "location_embed" in k:
            continue
        
        # Handle encoder prefix
        if k.startswith("encoder."):
            k = k.replace("encoder.", "")
        
        # Keep positional embeddings from model (depends on num_frames)
        if "pos_embed" in k:
            v = model.pos_embed
        
        clean_dict[k] = v
    
    # Handle patch embedding weights for different bands
    state_dict = select_patch_embed_weights(clean_dict, model, pretrained_bands, model_bands, encoder_only=True)
    
    return state_dict

def load_pretrained_weights(
    model: PrithviViT,
    variant: str = "prithvi_eo_v2_600",
    model_bands: Optional[List[int]] = None,
    cache_dir: Optional[str] = None
):
    """Load pretrained weights from Hugging Face"""
    
    if variant not in PRETRAINED_WEIGHTS:
        logger.warning(f"No pretrained weights available for {variant}")
        return model
    
    try:
        # Download weights from Hugging Face
        logger.info(f"Downloading pretrained weights for {variant}...")
        
        # Download config.json first (for download count)
        _ = hf_hub_download(
            repo_id=PRETRAINED_WEIGHTS[variant]["hf_hub_id"],
            filename="config.json",
            cache_dir=cache_dir
        )
        
        # Download actual weights
        pretrained_path = hf_hub_download(
            repo_id=PRETRAINED_WEIGHTS[variant]["hf_hub_id"],
            filename=PRETRAINED_WEIGHTS[variant]["hf_hub_filename"],
            cache_dir=cache_dir
        )
        
        # Load state dict
        logger.info(f"Loading weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        
        # Prepare band indices (0-5 for the 6 bands)
        pretrained_band_indices = list(range(6))  # [0, 1, 2, 3, 4, 5]
        model_band_indices = model_bands or pretrained_band_indices
        
        # Filter and adjust weights
        state_dict = checkpoint_filter_fn_vit(
            state_dict, model, pretrained_band_indices, model_band_indices
        )
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info(f"Successfully loaded pretrained weights for {variant}")
        
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.info("Continuing with random initialization")
    
    return model

# ================================
# Neck and Decoder Implementation
# ================================

class SelectIndices(nn.Module):
    """Select specific indices from feature list"""
    def __init__(self, indices: List[int]):
        super().__init__()
        self.indices = indices
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [features[i] for i in self.indices]

class UperNetDecoder(nn.Module):
    """UperNet Decoder for semantic segmentation"""
    def __init__(
        self,
        in_channels: List[int],
        channels: int = 256,
        num_classes: int = 17,
        dropout_ratio: float = 0.1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        align_corners: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # PSP Module
        self.psp_modules = PSPModule(
            in_channels[-1], 
            channels, 
            (1, 3, 6, 8),
            norm_layer=norm_layer
        )
        
        # FPN-like structure
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in in_channels[:-1]:  # Exclude the last one
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, 1),
                    norm_layer(channels),
                    nn.ReLU(inplace=True)
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    norm_layer(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final classifier
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1),
            norm_layer(channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.classifier = nn.Conv2d(channels, num_classes, 1)
    
    def psp_forward(self, x):
        """PSP forward"""
        return self.psp_modules(x)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Features: [f1, f2, f3, f4] from backbone
        
        # PSP on the largest feature map
        psp_out = self.psp_forward(features[-1])
        
        # FPN-like upsampling and fusion
        fpn_outs = [psp_out]
        
        for i in range(len(features) - 2, -1, -1):
            lateral_out = self.lateral_convs[i](features[i])
            
            # Upsample previous feature
            prev_shape = lateral_out.shape[2:]
            upsample_out = F.interpolate(
                fpn_outs[-1], 
                size=prev_shape,
                mode='bilinear', 
                align_corners=self.align_corners
            )
            
            # Fusion
            fused_out = lateral_out + upsample_out
            fpn_outs.append(self.fpn_convs[i](fused_out))
        
        # Concatenate all feature maps
        fpn_outs = fpn_outs[::-1]  # Reverse to match original order
        
        # Resize all to the same size (largest)
        target_size = fpn_outs[0].shape[2:]
        resized_outs = []
        for out in fpn_outs:
            if out.shape[2:] != target_size:
                out = F.interpolate(
                    out, 
                    size=target_size,
                    mode='bilinear', 
                    align_corners=self.align_corners
                )
            resized_outs.append(out)
        
        # Concatenate and reduce channels
        concat_out = torch.cat(resized_outs, dim=1)
        out = self.fpn_bottleneck(concat_out)
        
        # Classification
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

class PSPModule(nn.Module):
    """Pyramid Scene Parsing Module"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bin_sizes: Tuple[int, ...] = (1, 3, 6, 8),
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        
        self.stages = nn.ModuleList()
        ch_per_bin = out_channels // len(bin_sizes)
        
        for bin_size in bin_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_channels, ch_per_bin, 1),
                    norm_layer(ch_per_bin),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        h, w = x.size()[2:]
        pyramids = [x]
        
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(pyramid)
        
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        return output

# ================================
# Complete Segmentation Model
# ================================

class PrithviSegmentationModel(nn.Module):
    """Complete Prithvi segmentation model for multi-temporal data"""
    def __init__(
        self,
        num_classes: int = 17,
        backbone_name: str = "prithvi_eo_v2_600",
        img_size: int = 64,
        num_frames: int = MAX_TIMESTEPS,
        pretrained: bool = False,
        use_time_location_encoding: bool = False,
        cache_dir: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        self.original_img_size = img_size
        self.backbone_name = backbone_name
        self.num_frames = num_frames
        
        # Calculate the nearest size divisible by patch size
        if "600" in backbone_name:
            patch_size = 14
        else:
            patch_size = 16
        self.target_img_size = ((img_size + patch_size - 1) // patch_size) * patch_size
        
        # Configure coordinate encodings
        coords_encoding = []
        if use_time_location_encoding:
            coords_encoding = ["time", "location"]
        
        # Configure backbone parameters
        if backbone_name == "prithvi_eo_v2_600" or backbone_name == "prithvi_eo_v2_600_tl":
            backbone_config = {
                'img_size': self.target_img_size,
                'patch_size': (1, 14, 14),
                'num_frames': num_frames,
                'in_chans': 6,
                'embed_dim': 1280,
                'depth': 32,
                'num_heads': 16,
                'mlp_ratio': 4.0,
                'coords_encoding': coords_encoding,
                'coords_scale_learn': use_time_location_encoding,
                'drop_path': 0.0,
            }
            selected_indices = [7, 15, 23, 31]
        else:
            # Default use prithvi_eo_v2_300
            backbone_config = {
                'img_size': self.target_img_size,
                'patch_size': (1, 16, 16),
                'num_frames': num_frames,
                'in_chans': 6,
                'embed_dim': 1024,
                'depth': 24,
                'num_heads': 16,
                'mlp_ratio': 4.0,
                'coords_encoding': coords_encoding,
                'coords_scale_learn': use_time_location_encoding,
                'drop_path': 0.0,
            }
            selected_indices = [5, 11, 17, 23]
        
        # Create backbone
        self.backbone = PrithviViT(**backbone_config)
        
        # Load pretrained weights if requested
        if pretrained:
            # Use the correct variant name for time-location encoding
            variant_name = backbone_name
            if use_time_location_encoding and not backbone_name.endswith("_tl"):
                variant_name = backbone_name + "_tl"
            
            model_bands = list(range(6))  # [0, 1, 2, 3, 4, 5]
            self.backbone = load_pretrained_weights(
                self.backbone, 
                variant=variant_name,
                model_bands=model_bands,
                cache_dir=cache_dir
            )
        
        # freeze encoder
        if freeze_encoder:
            logger.info("Freezing encoder parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False

            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)")
        
        # Neck: select specific layers
        self.neck = SelectIndices(selected_indices)
        
        # Calculate output channels
        effective_time_dim = backbone_config['num_frames']
        in_channels = [backbone_config['embed_dim'] * effective_time_dim] * len(selected_indices)
        
        # Decoder
        self.decoder = UperNetDecoder(
            in_channels=in_channels,
            channels=256,
            num_classes=num_classes,
            dropout_ratio=0.1,
        )
        
        self.use_time_location_encoding = use_time_location_encoding
  
    def forward(self, x, temporal_coords=None, location_coords=None, valid_mask=None):
        """
        Args:
            x: Input tensor (B, C, T, H, W)
            temporal_coords: Temporal coordinates (B, T, 2) - year and day of year
            location_coords: Location coordinates (B, 2) - lat and lon
            valid_mask: Mask for valid time steps (B, T) - 1 for valid, 0 for padded
        """
        # Resize input to target size if needed
        if x.shape[-1] != self.target_img_size or x.shape[-2] != self.target_img_size:
            # For 5D tensor (B, C, T, H, W), we need to handle interpolation differently
            B, C, T, H, W = x.shape
            # Reshape to (B*T, C, H, W) for interpolation
            x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            # Interpolate
            x_resized = F.interpolate(
                x_reshaped, 
                size=(self.target_img_size, self.target_img_size), 
                mode='bilinear', 
                align_corners=False
            )
            # Reshape back to (B, C, T, H, W)
            x = x_resized.reshape(B, T, C, self.target_img_size, self.target_img_size).permute(0, 2, 1, 3, 4)
        
        # Backbone forward
        if self.use_time_location_encoding:
            features = self.backbone.forward_features(x, temporal_coords, location_coords)
        else:
            features = self.backbone.forward_features(x)
        
        # Prepare features for image model
        features = self.backbone.prepare_features_for_image_model(features)
        
        # Neck: select specific layers
        selected_features = self.neck(features)
        
        # Decoder
        out = self.decoder(selected_features)
        
        # Resize output back to original size if needed
        if out.shape[-1] != self.original_img_size or out.shape[-2] != self.original_img_size:
            out = F.interpolate(
                out,
                size=(self.original_img_size, self.original_img_size),
                mode='bilinear',
                align_corners=False
            )
        
        return out

# ================================
# Multi-temporal Dataset and Data Loading
# ================================

def date_to_day_of_year(year, month, day):
    """Convert date to day of year"""
    # Days in each month (non-leap year)
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Check for leap year
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days_in_month[1] = 29
    
    # Calculate day of year
    day_of_year = sum(days_in_month[:int(month)-1]) + int(day)
    return day_of_year

class MultiTemporalCropDataset(Dataset):
    """Multi-temporal crop segmentation dataset for real data"""
    def __init__(
        self, 
        data_root: str,
        patch_names: List[str],
        max_timesteps: int = MAX_TIMESTEPS,
        transform=None,
        normalize=True,
        use_time_location_encoding=False,
        target_size: int = 64  # Add target size parameter
    ):
        self.data_root = Path(data_root)
        self.band_dir = self.data_root / "band_patch"
        self.label_dir = self.data_root / "label_patch"
        self.geo_dir = self.data_root / "geo_patch"
        self.time_dir = self.data_root / "time_patch"
        
        self.patch_names = patch_names
        self.max_timesteps = max_timesteps
        self.transform = transform
        self.normalize = normalize
        self.use_time_location_encoding = use_time_location_encoding
        self.target_size = target_size  # Store target size
        
        # Prithvi v2 normalization parameters
        self.mean = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
        self.std = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)
    
    def __len__(self):
        return len(self.patch_names)
    
    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        
        # Load multi-temporal band data
        band_path = self.band_dir / f"{patch_name}.npy"
        band_data = np.load(band_path).astype(np.float32)  # (H, W, C, T)
        
        # Rearrange to (T, H, W, C)
        band_data = band_data.transpose(3, 0, 1, 2)
        actual_timesteps = band_data.shape[0]
        
        # Resize spatial dimensions if needed
        if band_data.shape[1] != self.target_size or band_data.shape[2] != self.target_size:
            # Resize each timestep
            resized_data = []
            for t in range(band_data.shape[0]):
                # Convert to tensor for interpolation
                t_data = torch.from_numpy(band_data[t]).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                t_resized = F.interpolate(t_data, size=(self.target_size, self.target_size), 
                                        mode='bilinear', align_corners=False)
                resized_data.append(t_resized.squeeze(0).permute(1, 2, 0).numpy())  # (H, W, C)
            band_data = np.stack(resized_data, axis=0)  # (T, H, W, C)
        
        # Load label data
        label_path = self.label_dir / f"{patch_name}.npy"
        label_data = np.load(label_path).astype(np.int64)  # (H, W)
        
        # Resize label if needed
        if label_data.shape[0] != self.target_size or label_data.shape[1] != self.target_size:
            # Use nearest neighbor for label interpolation
            label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0).float()
            label_resized = F.interpolate(label_tensor, size=(self.target_size, self.target_size), 
                                        mode='nearest')
            label_data = label_resized.squeeze().numpy().astype(np.int64)
        
        # Pad or truncate temporal sequence
        if actual_timesteps > self.max_timesteps:
            # Sample uniformly
            indices = np.linspace(0, actual_timesteps - 1, self.max_timesteps, dtype=int)
            band_data = band_data[indices]
            valid_mask = np.ones(self.max_timesteps, dtype=bool)
        elif actual_timesteps < self.max_timesteps:
            # Pad with zeros
            pad_length = self.max_timesteps - actual_timesteps
            pad_shape = (pad_length,) + band_data.shape[1:]
            pad_data = np.zeros(pad_shape, dtype=band_data.dtype)
            band_data = np.concatenate([band_data, pad_data], axis=0)
            
            valid_mask = np.zeros(self.max_timesteps, dtype=bool)
            valid_mask[:actual_timesteps] = True
        else:
            valid_mask = np.ones(self.max_timesteps, dtype=bool)
        
        # Normalize band data
        if self.normalize:
            band_data = (band_data - self.mean) / self.std
        
        # Convert to tensor format (C, T, H, W)
        band_tensor = torch.from_numpy(band_data).permute(3, 0, 1, 2).float()
        label_tensor = torch.from_numpy(label_data).long()
        valid_mask_tensor = torch.from_numpy(valid_mask).bool()
        
        # Convert labels to 0-16 (remove background class 0)
        label_tensor = torch.clamp(label_tensor - 1, min=-1)
        
        # Generate temporal and location coordinates if needed
        temporal_coords = None
        location_coords = None
        
        if self.use_time_location_encoding:
            # Load time data - REAL DATA FORMAT: (T, 3) not (H, W, 3)
            time_path = self.time_dir / f"{patch_name}.npy"
            time_data = np.load(time_path).astype(np.float32)  # (T, 3) - year, month, day
            
            # Load geo data
            geo_path = self.geo_dir / f"{patch_name}.npy"
            geo_data = np.load(geo_path).astype(np.float32)  # (H, W, 2) - lon, lat
            
            # Create temporal coordinates for all timesteps
            temporal_coords_list = []
            for t in range(self.max_timesteps):
                if t < actual_timesteps and t < len(time_data):
                    # Use actual time data
                    year = int(time_data[t, 0])
                    month = int(time_data[t, 1])
                    day = int(time_data[t, 2])
                    day_of_year = date_to_day_of_year(year, month, day)
                    temporal_coords_list.append([year, day_of_year])
                elif actual_timesteps > 0 and len(time_data) > 0:
                    # For padded timesteps, use the last valid timestamp
                    last_idx = min(actual_timesteps - 1, len(time_data) - 1)
                    year = int(time_data[last_idx, 0])
                    month = int(time_data[last_idx, 1])
                    day = int(time_data[last_idx, 2])
                    day_of_year = date_to_day_of_year(year, month, day)
                    temporal_coords_list.append([year, day_of_year])
                else:
                    # Fallback - should not happen with real data
                    temporal_coords_list.append([2022, 1])
            
            temporal_coords = torch.tensor(temporal_coords_list, dtype=torch.float32)
            
            # For location coords, take center pixel's coordinates
            h_center, w_center = geo_data.shape[0] // 2, geo_data.shape[1] // 2
            lon = geo_data[h_center, w_center, 0]
            lat = geo_data[h_center, w_center, 1]
            location_coords = torch.tensor([lat, lon], dtype=torch.float32)
        
        if self.transform:
            band_tensor, label_tensor = self.transform(band_tensor, label_tensor)
        
        result = {
            'image': band_tensor,
            'label': label_tensor,
            'valid_mask': valid_mask_tensor
        }
        
        if temporal_coords is not None:
            result['temporal_coords'] = temporal_coords
        if location_coords is not None:
            result['location_coords'] = location_coords
            
        return result

def create_multitemporal_dummy_data(data_root: str, num_samples: int = 1000):
    """Create multi-temporal dummy dataset matching real data structure"""
    data_path = Path(data_root)
    band_path = data_path / "band_patch"
    label_path = data_path / "label_patch"
    geo_path = data_path / "geo_patch"
    time_path = data_path / "time_patch"
    
    # Create directories
    for p in [band_path, label_path, geo_path, time_path]:
        p.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating {num_samples} multi-temporal dummy samples...")
    
    for i in tqdm(range(num_samples), desc="Creating multi-temporal dummy data"):
        # Random patch size (64 or 128)
        patch_size = np.random.choice([64, 128])
        
        # Random number of time steps
        num_timesteps = np.random.randint(3, 24)
        
        # Create band data (H, W, C, T)
        band_data = np.random.randint(0, 4000, (patch_size, patch_size, 6, num_timesteps)).astype(np.float32)
        
        # Add temporal correlation
        for h in range(patch_size):
            for w in range(patch_size):
                for c in range(6):
                    base_value = band_data[h, w, c, 0]
                    for t in range(1, num_timesteps):
                        variation = np.random.normal(0, base_value * 0.1)
                        band_data[h, w, c, t] = np.clip(
                            band_data[h, w, c, t-1] + variation, 
                            0, 4000
                        )
        
        # Create label data (H, W)
        label_data = np.random.randint(0, 18, (patch_size, patch_size)).astype(np.int64)
        
        # Create geo data (H, W, 2) - Austria region
        # Austria approximately: 46.4-49.0N, 9.5-17.2E
        center_lat = np.random.uniform(46.4, 49.0)
        center_lon = np.random.uniform(9.5, 17.2)
        
        # Small variations across the patch
        lat_grid = np.ones((patch_size, patch_size)) * center_lat + \
                   np.random.normal(0, 0.001, (patch_size, patch_size))
        lon_grid = np.ones((patch_size, patch_size)) * center_lon + \
                   np.random.normal(0, 0.001, (patch_size, patch_size))
        
        geo_data = np.stack([lon_grid, lat_grid], axis=-1).astype(np.float32)
        
        # Create time data (H, W, 3) - year, month, day
        year = 2022
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)  # Safe for all months
        
        time_data = np.ones((patch_size, patch_size, 3)).astype(np.float32)
        time_data[:, :, 0] = year
        time_data[:, :, 1] = month
        time_data[:, :, 2] = day
        
        # Save data
        patch_name = f"patch_{i:06d}"
        np.save(band_path / f"{patch_name}.npy", band_data)
        np.save(label_path / f"{patch_name}.npy", label_data)
        np.save(geo_path / f"{patch_name}.npy", geo_data)
        np.save(time_path / f"{patch_name}.npy", time_data)
    
    logger.info(f"Multi-temporal dummy data created in {data_root}")

def simple_multitemporal_augmentation(image, label):
    """Simple data augmentation for multi-temporal data"""
    # Ensure correct data types
    image = image.float()  # (C, T, H, W)
    label = label.long()   # (H, W)
    
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        image = torch.flip(image, [-1])  # Flip width
        label = torch.flip(label, [-1])
    
    # Random vertical flip
    if torch.rand(1) > 0.5:
        image = torch.flip(image, [-2])  # Flip height
        label = torch.flip(label, [-2])
    
    # Random 90-degree rotation
    if torch.rand(1) > 0.5:
        k = torch.randint(1, 4, (1,)).item()
        image = torch.rot90(image, k, [-2, -1])  # Rotate height-width
        label = torch.rot90(label, k, [-2, -1])
    
    return image, label

# ================================
# Evaluation Metrics
# ================================

def calculate_iou(pred, target, num_classes, ignore_index=-1):
    """Calculate IoU"""
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        if ignore_index is not None:
            valid_mask = (target != ignore_index)
            pred_cls = pred_cls & valid_mask
            target_cls = target_cls & valid_mask
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return ious

def calculate_metrics(predictions, targets, num_classes=17, ignore_index=-1):
    """Calculate all evaluation metrics"""
    from sklearn.metrics import recall_score, classification_report
    
    # Flatten predictions and targets
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Remove ignore_index
    if ignore_index is not None:
        valid_mask = target_flat != ignore_index
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
    
    # Calculate accuracy
    accuracy = accuracy_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy())
    
    # Calculate F1 macro (remove f1_micro as it equals accuracy)
    f1_macro = f1_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='macro', zero_division=0)
    
    # Calculate macro recall (balanced accuracy)
    recall_macro = recall_score(target_flat.cpu().numpy(), pred_flat.cpu().numpy(), average='macro', zero_division=0)
    
    # Calculate IoU
    ious = calculate_iou(pred_flat, target_flat, num_classes, ignore_index=None)
    mean_iou = np.nanmean(ious)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'recall_macro': recall_macro,  # This is balanced accuracy
        'mean_iou': mean_iou,
        'ious': ious
    }

# ================================
# Training and Validation Functions
# ================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_time_location_encoding=False):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, dtype=torch.float32, non_blocking=True)
        targets = batch['label'].to(device, dtype=torch.long, non_blocking=True)
        valid_masks = batch['valid_mask'].to(device, dtype=torch.bool, non_blocking=True)
        
        # Get temporal and location coordinates if using time-location encoding
        temporal_coords = None
        location_coords = None
        if use_time_location_encoding:
            temporal_coords = batch.get('temporal_coords')
            location_coords = batch.get('location_coords')
            if temporal_coords is not None:
                temporal_coords = temporal_coords.to(device, dtype=torch.float32, non_blocking=True)
            if location_coords is not None:
                location_coords = location_coords.to(device, dtype=torch.float32, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, temporal_coords, location_coords, valid_masks)
        
        # Calculate loss (ignore background class -1)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device, epoch, num_classes=17, use_time_location_encoding=False, return_predictions=False):
    """Validate one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
        for batch in pbar:
            images = batch['image'].to(device, dtype=torch.float32, non_blocking=True)
            targets = batch['label'].to(device, dtype=torch.long, non_blocking=True)
            valid_masks = batch['valid_mask'].to(device, dtype=torch.bool, non_blocking=True)
            
            # Get temporal and location coordinates if using time-location encoding
            temporal_coords = None
            location_coords = None
            if use_time_location_encoding:
                temporal_coords = batch.get('temporal_coords')
                location_coords = batch.get('location_coords')
                if temporal_coords is not None:
                    temporal_coords = temporal_coords.to(device, dtype=torch.float32, non_blocking=True)
                if location_coords is not None:
                    location_coords = location_coords.to(device, dtype=torch.float32, non_blocking=True)
            
            # Forward pass
            outputs = model(images, temporal_coords, location_coords, valid_masks)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Collect predictions and targets
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    
    if return_predictions:
        return total_loss / len(dataloader), metrics, all_predictions, all_targets
    else:
        return total_loss / len(dataloader), metrics

# ================================
# Main Training Function
# ================================


def main():
    # Set parameters
    REAL_DATA_ROOT = "/mnt/e/Codes/btfm4rs/data/downstream/austrian_crop_prithvi"
    DUMMY_DATA_ROOT = "/mnt/e/Codes/btfm4rs/data/downstream/austrian_crop_prithvi_dummy"
    
    # Check if we should use real data or create dummy data
    use_real_data = os.path.exists(REAL_DATA_ROOT)
    
    if use_real_data:
        DATA_ROOT = REAL_DATA_ROOT
        logger.info("Using real dataset")
    else:
        DATA_ROOT = DUMMY_DATA_ROOT
        logger.info("Real dataset not found, will create dummy data")
    
    BATCH_SIZE = 4  # Reduced batch size due to temporal dimension
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    NUM_CLASSES = 17
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_PRETRAINED = True
    USE_TIME_LOCATION_ENCODING = True  # Enable time-location encoding for multi-temporal
    FREEZE_ENCODER = True  # Freeze encoder
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Max timesteps: {MAX_TIMESTEPS}")
    logger.info(f"Use time-location encoding: {USE_TIME_LOCATION_ENCODING}")
    logger.info(f"Freeze encoder: {FREEZE_ENCODER}")
    
    # Get all patch names
    band_dir = Path(DATA_ROOT) / "band_patch"
    band_files = []
    
    if band_dir.exists():
        band_files = [f.stem for f in band_dir.glob("*.npy")]
    
    # Create dummy data if not using real data and no band files found
    if not use_real_data and len(band_files) == 0:
        logger.info("Creating dummy dataset...")
        create_multitemporal_dummy_data(DATA_ROOT, num_samples=20)
        # Re-scan for band files after creating dummy data
        band_files = [f.stem for f in band_dir.glob("*.npy")]
    
    logger.info(f"Found {len(band_files)} multi-temporal patches")
    
    if len(band_files) == 0:
        logger.error("No patches found!")
        return
    
    # Dataset split (3:1:6)
    band_files = sorted(band_files)
    random.shuffle(band_files)
    total_samples = len(band_files)
    
    train_size = int(total_samples * 0.1)  # 10%
    val_size = int(total_samples * 0.1125)    # val/test = 1:7
    test_size = total_samples - train_size - val_size
    
    train_files = band_files[:train_size]
    val_files = band_files[train_size:train_size+val_size]
    test_files = band_files[train_size+val_size:]
    
    logger.info(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets and data loaders
    train_dataset = MultiTemporalCropDataset(
        DATA_ROOT, train_files, 
        max_timesteps=MAX_TIMESTEPS,
        transform=simple_multitemporal_augmentation, 
        normalize=True,
        use_time_location_encoding=USE_TIME_LOCATION_ENCODING,
        target_size=64  # Add target size
    )
    val_dataset = MultiTemporalCropDataset(
        DATA_ROOT, val_files, 
        max_timesteps=MAX_TIMESTEPS,
        transform=None, 
        normalize=True,
        use_time_location_encoding=USE_TIME_LOCATION_ENCODING,
        target_size=64
    )
    test_dataset = MultiTemporalCropDataset(
        DATA_ROOT, test_files, 
        max_timesteps=MAX_TIMESTEPS,
        transform=None, 
        normalize=True,
        use_time_location_encoding=USE_TIME_LOCATION_ENCODING,
        target_size=64
    )
    
    def collate_fn(batch):
        """Custom collate function to handle multi-temporal data correctly"""
        result = {}
        
        # Stack tensors
        result['image'] = torch.stack([item['image'] for item in batch])
        result['label'] = torch.stack([item['label'] for item in batch])
        result['valid_mask'] = torch.stack([item['valid_mask'] for item in batch])
        
        # Handle optional time-location data
        if 'temporal_coords' in batch[0]:
            result['temporal_coords'] = torch.stack([item['temporal_coords'] for item in batch])
        if 'location_coords' in batch[0]:
            result['location_coords'] = torch.stack([item['location_coords'] for item in batch])
        
        return result
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    # Create model
    model = PrithviSegmentationModel(
        num_classes=NUM_CLASSES,
        backbone_name="prithvi_eo_v2_600",  # Use the largest model
        img_size=64,  # Will handle both 64 and 128 internally
        num_frames=MAX_TIMESTEPS,
        pretrained=USE_PRETRAINED,
        use_time_location_encoding=USE_TIME_LOCATION_ENCODING,
        cache_dir="./pretrained_cache",
        freeze_encoder=FREEZE_ENCODER
    ).to(DEVICE)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_val_iou = 0.0
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, USE_TIME_LOCATION_ENCODING
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, DEVICE, epoch, NUM_CLASSES, USE_TIME_LOCATION_ENCODING
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val Recall (Macro/Balanced Acc): {val_metrics['recall_macro']:.4f}")
        logger.info(f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        logger.info(f"Val mIoU: {val_metrics['mean_iou']:.4f}")
        
        # Save best model
        if val_metrics['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
            }, 'best_multitemporal_model.pth')
            logger.info(f"Best model saved with mIoU: {best_val_iou:.4f}")
    
    # Test
    logger.info("\nRunning final test...")
    checkpoint = torch.load('best_multitemporal_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get predictions with return_predictions=True
    test_loss, test_metrics, test_predictions, test_targets = validate_epoch(
        model, test_loader, criterion, DEVICE, NUM_EPOCHS, NUM_CLASSES, USE_TIME_LOCATION_ENCODING, return_predictions=True
    )

    logger.info("=== Final Multi-temporal Test Results ===")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Recall (Macro/Balanced Acc): {test_metrics['recall_macro']:.4f}")
    logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"Test mIoU: {test_metrics['mean_iou']:.4f}")

    # Output per-class IoU
    logger.info("\nPer-class IoU:")
    for i, iou in enumerate(test_metrics['ious']):
        if not np.isnan(iou):
            logger.info(f"Class {i+1}: {iou:.4f}")

    # Generate classification report
    from sklearn.metrics import classification_report

    # Flatten predictions and targets, removing ignore index
    pred_flat = test_predictions.flatten().cpu().numpy()
    target_flat = test_targets.flatten().cpu().numpy()

    # Remove ignore_index (-1)
    valid_mask = target_flat != -1
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]

    # Generate classification report
    # Class names for the 17 crop classes (1-17, since we removed background 0)
    class_names = [f"Class_{i}" for i in range(1, 18)]

    logger.info("\n=== Classification Report ===")
    logger.info("\n" + classification_report(
        target_flat, 
        pred_flat, 
        labels=list(range(17)),
        target_names=class_names,
        zero_division=0,
        digits=4
    ))


if __name__ == "__main__":
    main()