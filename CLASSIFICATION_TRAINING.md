# BDH Classification Training - Implementation Guide

## Overview

The BDH training pipeline has been updated to support binary sentiment classification on the SST-2 dataset. The model trains from scratch on the classification task using BPE tokenization (GPT-2 style).

## Files Modified/Created

### New Package Structure
- **`bdh/`**: Core package
  - `model.py`: Main BDH architecture and attention mechanism
  - `classifier.py`: Classification wrapper
  - `__init__.py`: Package exports

### Scripts
- **`scripts/train.py`**: Training script for SST-2
  - Loads SST-2 dataset from HuggingFace
  - Implements BPE text encoding (tiktoken)
  - Training loop with classification metrics
  - Validation evaluation with accuracy tracking
  - **CSV Logging**: Metrics are saved to `training_log.csv` for easy tracking
  
- **`scripts/predict.py`**: Inference script
  - Loads trained model
  - Runs prediction on text input

## Model Architecture

### BDHClassifier Structure
```
Input Text → BPE Encoding → BDH Core → Mean Pooling → Linear Head → Logits
             (max 256 tokens) (6 layers) (256-dim)     (256→2)     (2 classes)
```

### Configuration
- **Model Size**: ~25M parameters
- **Architecture**: 6 layers, 256 embedding dim, 4 attention heads
- **Vocab Size**: 50,304 (GPT-2 BPE)
- **Max Sequence Length**: 256 tokens
- **Output Classes**: 2 (negative/positive sentiment)

## Training Configuration

### Hyperparameters
```python
BLOCK_SIZE = 256          # Maximum sequence length
BATCH_SIZE = 32           # Training batch size
MAX_ITERS = 3000          # Total training iterations
LEARNING_RATE = 3e-4      # AdamW learning rate
WEIGHT_DECAY = 0.1        # L2 regularization
```

## Usage

### 1. Install Dependencies
```bash
# Install uv
pip install uv

# Install requirements
uv pip install --system -r requirements.txt
```

### 2. Run Training
```bash
# Default configuration
python scripts/train.py

# Custom configuration
python scripts/train.py --max_iters 5000 --batch_size 16 --learning_rate 1e-3
```

### 3. Run Prediction
```bash
python scripts/predict.py
```

## Output Files

### Training Log
Training progress is saved to `training_log.csv` in the current directory. This file contains the following columns:
- `step`: Current training iteration
- `loss`: Training loss
- `train_acc`: Training accuracy
- `val_acc`: Validation accuracy (only on validation steps)
- `val_loss`: Validation loss (only on validation steps)
- `lr`: Current learning rate

### Model Checkpoints
- `bdh_sst2_best.pth`: Model with the highest validation accuracy
- `bdh_sst2_checkpoint_*.pth`: Periodic checkpoints
- `bdh_sst2_final.pth`: Final trained model

## Model Loading

### Load Trained Model
```python
import torch
from bdh import BDHConfig, BDHClassifier

# Initialize model
config = BDHConfig(
    n_layer=6,
    n_embd=256,
    n_head=4,
    vocab_size=50304,
    dropout=0.1
)
model = BDHClassifier(config, num_classes=2)

# Load weights
model.load_state_dict(torch.load("bdh_sst2_best.pth"))
model.eval()
```

## Implementation Details

### Tokenization
We use `tiktoken` with the GPT-2 encoding. This allows for efficient subword tokenization, handling arbitrary text better than character-level or raw byte-level encoding for this model size.

### Forward Pass
1. **Embedding**: Token IDs → BDH embeddings
2. **BDH Core**: 6 layers of linear attention and modulation
3. **Pooling**: Mean pooling over the sequence length
4. **Classification**: Linear projection to class logits
