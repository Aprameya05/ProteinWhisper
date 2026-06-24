import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

class CachedProteinDataset(Dataset):
    def __init__(self, train_data, embedding_dir):
        self.items = [p for p in train_data if os.path.exists(f"{embedding_dir}/{p['id']}.pt")]
        self.embedding_dir = embedding_dir
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        return {"seq_emb": torch.load(f"{self.embedding_dir}/{item['id']}.pt",
                map_location="cpu", weights_only=True),
                "go_indices": item["go_indices"], "id": item["id"]}

def collate_fn(batch):
    return {"seq_embs": [b["seq_emb"] for b in batch],
            "go_indices": [b["go_indices"] for b in batch],
            "ids": [b["id"] for b in batch]}

def info_nce_loss(protein_repr, go_h_proj, positive_go_indices, temperature=0.07):
    p = F.normalize(protein_repr.unsqueeze(0), dim=-1)
    g = F.normalize(go_h_proj, dim=-1)
    logits = torch.matmul(p, g.T).squeeze(0) / temperature
    if len(positive_go_indices) == 0:
        return torch.tensor(0.0, device=protein_repr.device)
    return -F.log_softmax(logits, dim=0)[positive_go_indices].mean()
