# ProteinWhisper

**Zero-shot protein function annotation for the dark proteome.**

🔬 **Live demo:** https://huggingface.co/spaces/Aprameya05/ProteinWhisper

## Results

| Version | Architecture | Epochs | Fmax |
|---------|-------------|--------|------|
| V2 | ESM-2 + GOTermGNN + InfoNCE | 20 | 0.0008 |
| V3 | ESM-2 + GOTermGNN + hard negatives | 3 | 0.0008 |
| V4 | ESM-2 + classification head | 5 | **0.0504** |

- V4 is 63x better than V2/V3
- P53 validation: 13-16/20 correct GO terms including nucleus, transcription factor activity, DNA binding

## Architecture
- **ESM-2 (650M)** — per-residue sequence embeddings
- **Residue self-attention** — contextualizes each position
- **Attention pooling** — weighted protein-level representation
- **Classification head** — direct prediction of 38,263 GO terms

## Training data
- Swiss-Prot: 575,503 reviewed proteins
- 95,597 training proteins with experimental GO annotations
- Gene Ontology: 38,263 non-obsolete terms

## Setup
```bash
pip install torch transformers biopython pronto
```

## Citation
Preprint in preparation.
