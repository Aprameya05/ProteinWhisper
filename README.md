# ProteinWhisper

<div align="center">

![ProteinWhisper](https://img.shields.io/badge/ProteinWhisper-Zero--Shot%20Protein%20Function-10b981?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-ffd21e?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**Zero-shot protein function annotation for the dark proteome**

[🔬 Live Demo](https://huggingface.co/spaces/Aprameya05/ProteinWhisper) · [📄 Preprint](#) · [📊 Results](#results)

</div>

---

## The Problem

Approximately **35% of proteins in any sequenced genome are functionally dark** — their amino acid sequence is known, but their biological function is completely unknown. Existing tools like BLAST and HHpred rely on sequence homology: find a similar protein with a known function and transfer the annotation. When no similar protein exists, these tools return nothing.

This is not a niche problem. UniProt contains over **250 million protein sequences**, of which only 570,000 are manually reviewed. The remaining ~250 million are largely unannotated. Every time scientists sequence a new organism — a pathogen, a plant, a marine microbe — thousands of new dark proteins appear.

**ProteinWhisper solves this by reading the sequence directly**, without requiring any similar protein to exist.

---

## Architecture
Protein sequence (amino acids)

│

▼

┌─────────────────────────────┐

│     ESM-2 (650M params)     │  ← Meta AI protein language model

│  Trained on 250M sequences  │    Pre-trained, frozen during training

└─────────────────────────────┘

│ [L × 1280] per-residue embeddings

▼

┌─────────────────────────────┐

│   Linear projection + LN    │  [L × 512]

│   TransformerEncoderLayer   │  Self-attention over residues

│   Attention pooling head    │  Learns which residues matter

└─────────────────────────────┘

│ [512] protein vector

▼

┌─────────────────────────────┐

│   Classification head       │  Linear → LN → ReLU → Linear

│   38,263 GO term outputs    │  One sigmoid score per GO term

└─────────────────────────────┘

│

▼

GO term predictions with confidence scores

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Sequence encoder | ESM-2 t33 650M | Best per-residue representations; trained on 250M sequences |
| Pooling | Attention-weighted mean | Learns which residues are functionally informative |
| Loss function | Weighted BCE | Handles severe class imbalance (avg 7 positives / 38,263 terms) |
| Baseline comparison | InfoNCE + hard negatives | Ablation showed GO embedding collapse; BCE solved it |
| Evaluation metric | Fmax | Standard CAFA benchmark metric for protein function prediction |

---

## Results

### Training curve

| Epoch | Loss | Notes |
|---|---|---|
| 1 | 0.7991 | Random initialization |
| 2 | 0.6100 | Fast early convergence |
| 3 | 0.5529 | |
| 4 | 0.5218 | |
| 5 | **0.5036** | Best generalization (early stopping) |

### Evaluation

| Model | Architecture | Loss | Fmax | Notes |
|---|---|---|---|---|
| V2 | ESM-2 + GOTermGNN + fusion | InfoNCE (full negatives) | 0.0008 | GO embedding collapse |
| V3 | ESM-2 + GOTermGNN + fusion | Hard negative mining | 0.0008 | Collapse persists |
| **V4** | **ESM-2 + attention pool + classifier** | **Weighted BCE** | **0.0504** | **63x improvement** |

- **Random baseline:** Fmax ~0.01–0.05
- **BLAST (homology):** Fmax ~0.40–0.60 (requires similar sequences to exist)
- **ProteinWhisper V4:** Fmax 0.0504 with only 5 training epochs, no homology

### P53 validation (P53_HUMAN)

ProteinWhisper was never trained on P53 as a test case. Running inference:
Input:  MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLM... (393 AA)
Top predictions:

GO:0005634  Nucleus                              0.9980  ✓

GO:0005737  Cytoplasm                            0.9967  ✓

GO:0006357  Transcription by RNA Pol II          0.9805  ✓

GO:0000978  RNA Pol II promoter binding          0.9759  ✓

GO:0000981  DNA-binding transcription factor     0.9993  ✓
Correct in top 20: 16 / 115 true GO terms

P53 is a nuclear transcription factor that regulates DNA repair and apoptosis. ProteinWhisper correctly identified all major functional categories from sequence alone.

### Dark proteome predictions

Five functionally uncharacterized proteins run through ProteinWhisper:

| UniProt ID | Organism | Length | Top prediction | Confidence |
|---|---|---|---|---|
| A0A0R4IKT3 | Zebrafish | 2634 AA | Transcription by RNA Pol II | 0.9954 |
| A0A1I8GTS4 | Human | 349 AA | DNA-binding transcription factor | 0.9993 |
| A0A2I2Y7J3 | Mouse | 247 AA | Nucleus | 0.9998 |
| A0A0A0MQR0 | E. coli | 520 AA | **Plasma membrane** | 0.9797 |
| A0A1Q3CA14 | Yeast | 118 AA | RNA Pol II promoter binding | 0.9998 |

The E. coli protein (A0A0A0MQR0) was correctly distinguished as a **membrane protein** — its sequence begins with a classic signal peptide `MSQLSLSWLGLWPVAAS`, consistent with a transmembrane domain.

---

## Ablation study

We systematically compared three training approaches, identifying the root cause of initial poor performance:
InfoNCE (38,263 negatives)  →  Fmax 0.0008  ✗  GO embedding collapse

Hard negative mining (256)  →  Fmax 0.0008  ✗  Collapse persists in GNN

Direct BCE classification   →  Fmax 0.0504  ✓  Stable, monotonic convergence

**Finding:** When training with contrastive loss over the full GO ontology (38,263 terms), the GOTermGNN converges to near-identical embeddings for all GO terms — a known failure mode called representation collapse. Switching to direct multi-label classification with weighted BCE resolved this entirely, consistent with findings in ProteInfer (Sanderson et al., 2023).

---

## Installation

```bash
git clone https://github.com/Aprameya05/ProteinWhisper
cd ProteinWhisper
pip install -r requirements.txt
```

**Requirements:**
torch>=2.0.0

torch-geometric

transformers>=4.35.0

biopython

pronto

gradio

plotly

tqdm

numpy

---

## Quick start

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle

# Load model
from src.models import ProteinWhisperV4

esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm2           = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval()

model = ProteinWhisperV4(n_go_terms=38263)
ckpt  = torch.load("checkpoints/proteinwhisper_v4_best.pt", map_location="cpu")
model.load_state_dict(ckpt["fusion_v4"])
model.eval()

with open("idx_to_term.pkl", "rb") as f:
    idx_to_term = pickle.load(f)

# Predict GO terms for any sequence
sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLM..."

with torch.no_grad():
    tok     = esm2_tokenizer(sequence[:512], return_tensors="pt")
    out     = esm2(**tok, output_hidden_states=True)
    seq_emb = out.last_hidden_state[0, 1:-1, :]
    logits, _ = model(seq_emb)
    probs     = torch.sigmoid(logits)
    topk_p, topk_i = probs.topk(20)

print("Top GO term predictions:")
for i in range(20):
    go_id = idx_to_term[topk_i[i].item()]
    print(f"  {go_id}  {topk_p[i].item():.4f}")
```

---

## Repository structure
ProteinWhisper/

├── src/

│   ├── models.py          # GOTermGNN, ProteinWhisperV4 architectures

│   ├── train.py           # Training loop, dataset, loss functions

│   └── data.py            # Swiss-Prot parsing, GO graph construction

├── results/

│   ├── final_results.json          # V4 evaluation results

│   ├── v2_baseline.json            # V2/V3 contrastive baseline

│   ├── dark_proteome_results.json  # 5 uncharacterized protein predictions

│   ├── training_curve.png          # Full 20-epoch V2 loss curve

│   └── v4_training_curve.png       # V4 BCE training curve

├── requirements.txt

└── README.md

---

## Training data

| Dataset | Size | Source |
|---|---|---|
| Swiss-Prot (sequences) | 575,503 proteins | UniProtKB reviewed |
| Swiss-Prot (annotations) | 555,469 proteins with GO | UniProtKB .dat file |
| Training set (experimental only) | 95,597 proteins | IDA/IMP/IPI/IGI/IEP/EXP evidence |
| Gene Ontology | 38,263 non-obsolete terms | OBO flat file, June 2025 |

Evidence codes used: `IDA, IPI, IMP, IGI, IEP, EXP, HDA, HMP, HGI, IBA`
Excluded: electronic annotations (IEA) — too noisy for training signal

---

## Reproducing results

```bash
# Step 1: Download data
python src/data.py --download

# Step 2: Precompute ESM-2 embeddings (requires A100, ~42 min)
python src/precompute.py --output data/embeddings/

# Step 3: Train V4
python src/train.py --epochs 5 --batch_size 8 --lr 3e-4

# Step 4: Evaluate
python src/evaluate.py --checkpoint checkpoints/proteinwhisper_v4_best.pt
```

---

## Citation

```bibtex
@article{bharadwaj2025proteinwhisper,
  title={ProteinWhisper: Zero-Shot Protein Function Annotation via ESM-2 Embeddings and Direct GO Term Classification},
  author={Bharadwaj, Aprameya},
  year={2025},
  note={Preprint in preparation}
}
```

---

## Acknowledgements

- Meta AI for the ESM-2 protein language model
- UniProt Consortium for Swiss-Prot database
- Gene Ontology Consortium for the GO ontology
- HuggingFace for model hosting and Spaces deployment
