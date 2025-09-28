from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(sizes: List[int], act=nn.ReLU, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NodeFeaturizer(nn.Module):
    """Embeds discrete node attributes with learnable embeddings.

    Expects node feature columns in order:
    [Z, degree, total_valence, formal_charge, hybridization, aromatic, implicit_h, explicit_h, chiral_tag, map_num]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.emb_z = nn.Embedding(119, dim)  # elements up to 118
        self.emb_deg = nn.Embedding(8, dim)
        self.emb_val = nn.Embedding(9, dim)
        self.emb_charge = nn.Embedding(11, dim)  # [-5..+5] shifted
        self.emb_hybrid = nn.Embedding(8, dim)
        self.emb_arom = nn.Embedding(2, dim)
        self.emb_himp = nn.Embedding(9, dim)
        self.emb_hexp = nn.Embedding(9, dim)
        self.emb_chiral = nn.Embedding(5, dim)
        # map_num is not semantic, used for pointers; don't embed here
        self.proj = _make_mlp([9 * dim, dim], dropout=0.0)

    def forward(self, x: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        z, deg, val, charge, hybrid, arom, himp, hexp, chiral, mapnum = (
            x[:, 0], x[:, 1], x[:, 2], x[:, 3] + 5, x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
        )
        parts = [
            self.emb_z(z),
            self.emb_deg(deg.clamp(min=0, max=self.emb_deg.num_embeddings - 1)),
            self.emb_val(val.clamp(min=0, max=self.emb_val.num_embeddings - 1)),
            self.emb_charge(charge.clamp(min=0, max=self.emb_charge.num_embeddings - 1)),
            self.emb_hybrid(hybrid.clamp(min=0, max=self.emb_hybrid.num_embeddings - 1)),
            self.emb_arom(arom.clamp(min=0, max=1)),
            self.emb_himp(himp.clamp(min=0, max=self.emb_himp.num_embeddings - 1)),
            self.emb_hexp(hexp.clamp(min=0, max=self.emb_hexp.num_embeddings - 1)),
            self.emb_chiral(chiral.clamp(min=0, max=self.emb_chiral.num_embeddings - 1)),
        ]
        h = torch.cat(parts, dim=-1)
        h = self.proj(h)
        return h, mapnum


class GraphTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.ff = _make_mlp([dim, dim * 4, dim], dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        H = self.heads
        q = self.q(h).view(B, N, H, D // H).transpose(1, 2)  # B,H,N,d
        k = self.k(h).view(B, N, H, D // H).transpose(1, 2)
        v = self.v(h).view(B, N, H, D // H).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (D // H) ** 0.5  # B,H,N,N
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # B,H,N,d
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        return out

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Full-graph self-attention per graph (position awareness can be added externally)
        x = h
        h = self.ln1(h + self.dropout(self._attn(h)))
        h = self.ln2(h + self.dropout(self.ff(h)))
        return h


class GraphTransformer(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4, heads: int = 4, dropout: float = 0.1, max_pos: int = 512):
        super().__init__()
        self.node_enc = NodeFeaturizer(dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([GraphTransformerBlock(dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.max_pos = max_pos
        self.pos_emb = nn.Embedding(max_pos, dim)

    def encode_graph(self, node_feats: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        h, mapnum = self.node_enc(node_feats)
        # Position awareness via map number buckets
        pos = mapnum.clamp(min=0, max=self.max_pos - 1)
        h = h + self.pos_emb(pos)
        # Prepend a [CLS] token
        B = 1
        cls = self.cls_token.expand(B, -1, -1)  # 1,1,D
        h = torch.cat([cls.squeeze(0), h], dim=0).unsqueeze(0)  # 1,N+1,D
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        h = h.squeeze(0)  # N+1,D
        return h, mapnum


class EditPolicy(nn.Module):
    def __init__(self, dim: int, num_action_types: int):
        super().__init__()
        self.type_head = _make_mlp([dim, dim, num_action_types])
        self.ptr_i = _make_mlp([dim, dim, 1])  # per-node logits
        self.pair_scorer = _make_mlp([dim * 3, dim, 1])  # j conditioned on i
        self.order_head = _make_mlp([dim * 2, dim, 4])  # SINGLE/DOUBLE/TRIPLE/AROMATIC
        self.stereo_head = _make_mlp([dim * 2, dim, 3])  # NONE/E/Z
        self.prop_head = _make_mlp([dim * 2, dim, 2])  # formal_charge / chiral_tag
        self.charge_head = _make_mlp([dim * 2, dim, 11])  # [-5..+5]
        self.chiral_head = _make_mlp([dim * 2, dim, 5])

    def forward(self, h: torch.Tensor, mapnum: torch.LongTensor) -> Dict[str, torch.Tensor]:
        # h: [N+1,D] (with CLS at 0), nodes follow
        cls = h[0]
        nodes = h[1:]
        type_logits = self.type_head(cls)
        i_logits = self.ptr_i(nodes).squeeze(-1)
        return {"type": type_logits, "i": i_logits}

    def j_logits(self, h: torch.Tensor, i_index: int) -> torch.Tensor:
        nodes = h[1:]
        hi = nodes[i_index]
        N = nodes.size(0)
        hi_tiled = hi.expand(N, -1)
        feats = torch.cat([hi_tiled, nodes, hi_tiled * nodes], dim=-1)
        return self.pair_scorer(feats).squeeze(-1)

    def order_stereo_logits(self, h: torch.Tensor, i_index: int, j_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = h[1:]
        hi, hj = nodes[i_index], nodes[j_index]
        pair = torch.cat([hi, hj], dim=-1)
        return self.order_head(pair), self.stereo_head(pair)

    def change_atom_logits(self, h: torch.Tensor, i_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = h[1:]
        hi = nodes[i_index]
        cls = h[0]
        pair = torch.cat([cls, hi], dim=-1)
        prop = self.prop_head(pair)
        charge = self.charge_head(pair)
        chiral = self.chiral_head(pair)
        return prop, charge, chiral


class EditModel(nn.Module):
    def __init__(self, num_action_types: int, dim: int = 256, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = GraphTransformer(dim=dim, depth=depth, heads=heads, dropout=dropout)
        self.policy = EditPolicy(dim=dim, num_action_types=num_action_types)

    def encode(self, node_feats: torch.LongTensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        return self.encoder.encode_graph(node_feats)

    def forward(self, node_feats: torch.LongTensor) -> Dict[str, torch.Tensor]:
        h, mapnum = self.encode(node_feats)
        out = self.policy(h, mapnum)
        return out

