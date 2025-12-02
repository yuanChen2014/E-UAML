
# E-UAML: Extended Unified Adversarial Multimodal Learning for Kinship Verification

This repository provides a **minimal and reproducible reference implementation** of the E-UAML framework
described in our manuscript on multimodal kinship verification. The goal is to expose the core architectural
components in a clear, lightweight PyTorch implementation suitable for inspection and reuse.

## Overview

E-UAML integrates four biometric modalities:

- Face (RGB images)
- Voice (audio-derived spectrogram features)
- Ear (auricular images / contours)
- Gait (pose-based behavioral sequences)

The framework employs:

- Modality-specific deep encoders
- L2-normalized embeddings
- Adversarial modality alignment (via a gradient reversal layer and modality discriminator)
- Multi-head modality attention
- Transformer-based fusion
- A composite loss (contrastive kinship loss + attention regularization; adversarial loss can be added)

This repository focuses on a **clean skeleton** of the architecture rather than full-scale training on the
TALKIN-Family dataset, which is not redistributed here due to licensing and privacy considerations.

## Repository Structure

```text
.
├── notebooks/
│   └── E_UAML_implementation_English.ipynb   # Colab/Jupyter-ready skeleton implementation
├── src/
│   └── euaml_model.py                        # Core PyTorch model classes (encoders, attention, fusion)
├── requirements.txt                          # Minimal Python dependencies
└── README.md
```

## Installation

We recommend using Python 3.9+ and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook notebooks/E_UAML_implementation_English.ipynb
   ```
2. Run all cells to verify:
   - The four modality-specific encoders
   - Modality attention
   - Transformer fusion
   - Forward and backward passes with a dummy batch

For integration into other projects, you can import the core model from `src/euaml_model.py`:

```python
from src.euaml_model import EUAML
model = EUAML()
```

## Notes on Reproducibility

- The code is fully self-contained and does not rely on external private modules.
- Dataset loading is not included; users should adapt the data pipeline to their own multimodal dataset
  (e.g., TALKIN-Family or its augmented variants), respecting relevant licenses and ethical constraints.
- The architecture (modules, shapes, and loss definitions) follows the description in the manuscript, and
  serves as a reference implementation for reviewers and researchers.

## License

This code is released for **research and academic use**. Please cite the corresponding paper if you use
E-UAML or its implementation in your work.
