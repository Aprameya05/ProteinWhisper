import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def predict_go_terms(sequence, fusion_model, go_gnn, go_graph_gpu,
                     esm2, esm2_tokenizer, idx_to_term,
                     top_k=20, device="cuda"):
    """
    Predict GO function annotations for any protein sequence.

    Args:
        sequence:    amino acid string (e.g. "MKTAY...")
        top_k:       number of GO terms to return

    Returns:
        list of dicts: [{go_id, score, confidence}]
    """
    sequence = sequence[:512]

    with torch.no_grad():
        tok     = esm2_tokenizer(sequence, return_tensors="pt").to(device)
        out     = esm2(**tok, output_hidden_states=True)
        seq_emb = out.last_hidden_state[0, 1:-1, :]

        go_emb = go_gnn(go_graph_gpu.x, go_graph_gpu.edge_index)
        protein_repr, go_h = fusion_model(seq_emb, go_emb)

        protein_norm = F.normalize(protein_repr.unsqueeze(0), dim=-1)
        go_norm      = F.normalize(go_h, dim=-1)
        scores       = torch.matmul(protein_norm, go_norm.T).squeeze(0)

        topk_scores, topk_indices = scores.topk(top_k)
        s_min = scores.min().item()
        s_max = scores.max().item()

    return [
        {
            "go_id":      idx_to_term[idx.item()],
            "score":      topk_scores[i].item(),
            "confidence": float((topk_scores[i].item() - s_min) / (s_max - s_min + 1e-8))
        }
        for i, idx in enumerate(topk_indices)
    ]
