# ProteinWhisper

**Zero-shot protein function annotation for the dark proteome.**

ProteinWhisper predicts Gene Ontology (GO) function annotations for proteins
with no known homologs — the ~35% of any proteome that existing tools cannot annotate.

## Architecture
- **ESM-2 (650M)** — per-residue sequence embeddings
- **GOTermGNN** — graph attention network over GO ontology DAG (38,263 terms)
- **Cross-modal fusion** — residue self-attention + protein-GO cross-attention
- **InfoNCE contrastive loss** — aligns protein representations to GO term space

## What is novel
No existing tool fuses ESM-2 sequence representations with GO ontology graph
structure via contrastive learning for zero-shot dark proteome annotation.

## Training data
- Swiss-Prot: 575,503 reviewed proteins
- 51,711 training proteins with experimental GO annotations
- Gene Ontology: 38,263 non-obsolete terms

## Setup
```bash
pip install -r requirements.txt
```

## Citation
Preprint in preparation.
