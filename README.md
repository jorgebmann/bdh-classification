# BDH Model Implementation

This repository contains the PyTorch implementation of the **BDH (Big Deep Hierarchy)** model.

## Project Structure

```
.
├── bdh/                # Main model package
│   ├── __init__.py     # Package exports
│   ├── model.py        # Core BDH model and attention mechanism
│   └── classifier.py   # Classification wrapper
├── scripts/            # Runnable scripts
│   ├── train.py        # Training script for SST-2 classification
│   ├── predict.py      # Inference script for trained models
│   └── analysis/       # Research and visualization tools
├── requirements.txt    # Project dependencies
└── CLASSIFICATION_TRAINING.md # Training documentation
```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model on the SST-2 sentiment analysis dataset:

```bash
python scripts/train.py
```

Arguments:
- `--max_iters`: Maximum training iterations (default: 20000)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 6e-4)

### Prediction

To use a trained model for prediction:

```bash
python scripts/predict.py
```

## Model Architecture

The BDH model features:
- **Linear Attention** with Rotary Positional Embeddings (RoPE)
- **Conceptual Space Projections** (Expansion/Compression)
- **Gated Activation Units**

For more details, refer to `bdh/model.py`.

## License

Copyright 2025 Pathway Technology, Inc.
