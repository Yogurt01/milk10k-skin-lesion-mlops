"""Dual-stream multi-modal fusion network and losses.

Two image encoders + a tabular MLP, fused as tokens by a small Transformer; focal loss for the 11 classes.
"""
from typing import Any, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import SITES
from .metrics import CLASSES


class TabularEncoder(nn.Module):
    def __init__(self, d: int, n_sites: int = len(SITES)):
        super().__init__()
        self.site = nn.Embedding(n_sites + 1, 16)   # index 0 = unknown
        self.sex = nn.Embedding(3, 4)
        self.fst = nn.Embedding(7, 8)
        self.mlp = nn.Sequential(nn.Linear(16 + 4 + 8 + 2, d), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(d, d))

    def forward(self, site, sex, fst, age, age_missing):
        x = torch.cat([self.site(site), self.sex(sex), self.fst(fst),
                       age[:, None], age_missing[:, None]], dim=1)
        return self.mlp(x)


class CrossAttnBlock(nn.Module):
    """Bidirectional cross-attention between derm and clinical feature-map tokens (pre-norm, residual).

    Derm tokens query clinical tokens and vice versa, both from the block's input states. Where the clinical
    image is missing (clin_mask == False) the derm tokens get no update from it; the clinical stream itself is
    replaced by the [NULL] token downstream. The zero clinical input is ordinary data, so nothing is fully masked
    and no NaN can appear.
    """

    def __init__(self, d: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_dq, self.norm_ckv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_cq, self.norm_dkv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn_d = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)   # derm → clinical
        self.attn_c = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)   # clinical → derm
        self.ff_d = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d))
        self.ff_c = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, d))

    def forward(self, td, tc, clin_mask):
        kv_c, kv_d = self.norm_ckv(tc), self.norm_dkv(td)
        upd_d = self.attn_d(self.norm_dq(td), kv_c, kv_c, need_weights=False)[0]
        upd_c = self.attn_c(self.norm_cq(tc), kv_d, kv_d, need_weights=False)[0]
        td = torch.where(clin_mask[:, None, None], td + upd_d, td)
        tc = tc + upd_c
        return td + self.ff_d(td), tc + self.ff_c(tc)


def feature_tokens(backbone: Any, x: torch.Tensor) -> torch.Tensor:  # a timm model
    """timm forward_features → (B, N, C) tokens, for NCHW (CNN), NHWC or already-token (ViT) outputs."""
    f = backbone.forward_features(x)
    if f.ndim == 4:
        f = f.flatten(2).transpose(1, 2) if f.shape[1] == backbone.num_features else f.flatten(1, 2)
    return f


class MILKFusionNet(nn.Module):
    """Derm backbone + clinical backbone + tabular MLP, fused as tokens by a small Transformer.

    A missing clinical image (clin_mask == False) is replaced by a learned [NULL] token.
    With xattn_layers > 0 (P4a), the two image streams keep their feature-map tokens (9×9 at 288 px for
    ResNet18), exchange information through `xattn_layers` CrossAttnBlocks, and are then mean-pooled into the
    same 4-token fusion. xattn_layers = 0 is the original pooled model, with identical parameters.
    Returns (11-class logits, binary malignancy logit).
    """

    def __init__(self,
                 derm_name: str = "convnext_small.fb_in22k_ft_in1k_384",
                 clin_name: str = "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
                 d: int = 512, n_classes: int = len(CLASSES), pretrained: bool = True, xattn_layers: int = 0):
        super().__init__()
        self.derm = timm.create_model(derm_name, pretrained=pretrained, num_classes=0)
        self.clin = timm.create_model(clin_name, pretrained=pretrained, num_classes=0)
        self.derm_proj = nn.Linear(self.derm.num_features, d)
        self.clin_proj = nn.Linear(self.clin.num_features, d)
        self.tab = TabularEncoder(d)

        self.null_clin = nn.Parameter(torch.zeros(d))          # learned [NULL] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Parameter(torch.zeros(1, 4, d))     # CLS / derm / clin / tab
        nn.init.trunc_normal_(self.null_clin, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_cls = nn.Linear(d, n_classes)
        self.head_bin = nn.Linear(d, 1)
        # Created last, so the default (0 blocks) draws no extra random numbers and keeps the same state_dict.
        self.xattn = nn.ModuleList([CrossAttnBlock(d) for _ in range(xattn_layers)])

    def forward(self, derm, clin, clin_mask, site, sex, fst, age, age_missing):
        b = derm.shape[0]
        if len(self.xattn):
            td = self.derm_proj(feature_tokens(self.derm, derm))
            tc = self.clin_proj(feature_tokens(self.clin, clin))
            for blk in self.xattn:
                td, tc = blk(td, tc, clin_mask)
            z_d, z_c = td.mean(1), tc.mean(1)
        else:
            z_d = self.derm_proj(self.derm(derm))
            z_c = self.clin_proj(self.clin(clin))
        z_c = torch.where(clin_mask[:, None], z_c, self.null_clin.expand(b, -1))
        z_t = self.tab(site, sex, fst, age, age_missing)

        tokens = torch.cat([self.cls_token.expand(b, -1, -1),
                            torch.stack([z_d, z_c, z_t], dim=1)], dim=1) + self.type_emb
        h = self.norm(self.fusion(tokens)[:, 0])
        return self.head_cls(h), self.head_bin(h).squeeze(1)

    def forward_batch(self, batch: dict):
        """Convenience wrapper for a dict batch produced by `MILKDataset` + default collate."""
        return self(batch["derm"], batch["clin"], batch["clin_mask"], batch["site"],
                    batch["sex"], batch["fst"], batch["age"], batch["age_missing"])


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class weights `alpha`."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.alpha, reduction="none")
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()
