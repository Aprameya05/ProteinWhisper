import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GOTermGNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=256, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.conv3 = GATConv(hidden_dim*heads, out_dim, heads=1, concat=False, dropout=0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        x = self.dropout(self.relu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(self.relu(self.norm2(self.conv2(x, edge_index))))
        return self.norm3(self.conv3(x, edge_index))

class ProteinWhisperV2(nn.Module):
    """Zero-shot protein function annotation via ESM-2 + GO graph contrastive fusion."""
    def __init__(self, seq_dim=1280, go_dim=256, fusion_dim=512):
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(0.1))
        self.residue_attn = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.go_proj = nn.Sequential(
            nn.Linear(go_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        self.protein_go_attn = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(fusion_dim, fusion_dim))

    def forward(self, seq_emb, go_emb):
        seq_h = self.seq_proj(seq_emb).unsqueeze(0)
        go_h  = self.go_proj(go_emb)
        seq_h = self.residue_attn(seq_h)
        protein_vec = seq_h.mean(dim=1)
        go_context, _ = self.protein_go_attn(
            query=protein_vec.unsqueeze(1), key=go_h.unsqueeze(0), value=go_h.unsqueeze(0))
        go_context  = go_context.squeeze(0).squeeze(0)
        protein_vec = protein_vec.squeeze(0)
        protein_repr = self.output_mlp(torch.cat([protein_vec, go_context], dim=-1))
        return protein_repr, go_h
