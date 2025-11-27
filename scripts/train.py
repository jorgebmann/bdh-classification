# train.py
# Training script for BDH text classification on SST-2
# Trains a BDH model from scratch for binary sentiment classification

import os
from contextlib import nullcontext
import torch
import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tiktoken

# Import model architectures
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from bdh import BDHConfig, BDHClassifier

# --- Configuration Section ---
BLOCK_SIZE = 256  # Maximum sequence length
BATCH_SIZE = 8    # Physical batch size
GRAD_ACCUM_STEPS = 8  # Effective batch size = 64
MAX_ITERS = 20000

# Learning Rate Schedule
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_ITERS = 2000
LR_DECAY_ITERS = 20000

WEIGHT_DECAY = 0.1
LOG_FREQ = 200
EVAL_FREQ = 1000  # Evaluate on validation set every N steps
CHECKPOINT_FREQ = 5000

# Compilation settings
USE_COMPILE = True

# --- Device and Dtype Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = ("bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (torch.amp.autocast(device_type=device.type, dtype=ptdtype) if "cuda" in str(device) else nullcontext())

# Fix for AttributeError on CPU or older PyTorch versions
if torch.cuda.is_available():
    try:
        # Try new PyTorch 2.0+ location
        scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
    except AttributeError:
        # Fallback for older versions
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
else:
    # On CPU, GradScaler is generally not needed or a no-op is sufficient
    # We create a dummy scaler that does nothing
    class MockScaler:
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
    scaler = MockScaler()

# --- Performance Optimizations ---
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Using device: {device} with dtype: {dtype}")

# --- Data Loading Section ---

# Initialize tokenizer globally
enc = tiktoken.get_encoding("gpt2")

def encode_text(text, max_length=BLOCK_SIZE):
    """
    Convert text to BPE tokens using tiktoken (GPT-2 encoding).
    """
    try:
        # Encode text to tokens
        ids = enc.encode(text, allowed_special={"<|endoftext|>"})
    except Exception:
        # Fallback for empty or weird strings
        ids = []
        
    # Truncate if too long
    if len(ids) > max_length:
        ids = ids[:max_length]
    
    # Pad if too short (using 50256 as padding, which is <|endoftext|>)
    padding_token = 50256 
    if len(ids) < max_length:
        ids.extend([padding_token] * (max_length - len(ids)))
    
    return torch.tensor(ids, dtype=torch.long)


def load_sst2_data():
    """
    Load SST-2 dataset from HuggingFace.
    
    Returns:
        train_dataset, validation_dataset
    """
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_batch(dataset, batch_size=BATCH_SIZE, indices=None):
    """
    Get a batch of data from the dataset.
    """
    if indices is None:
        indices = np.random.randint(0, len(dataset), size=batch_size)
    
    inputs = []
    labels = []
    
    for idx in indices:
        example = dataset[int(idx)]
        text = example["sentence"]
        label = example["label"]
        
        # Encode text
        encoded = encode_text(text)
        inputs.append(encoded)
        labels.append(label)
    
    inputs = torch.stack(inputs).to(device, non_blocking=True)
    labels = torch.tensor(labels, dtype=torch.long).to(device, non_blocking=True)
    
    return inputs, labels


@torch.no_grad()
def evaluate(model, dataset, max_batches=None):
    """
    Evaluate the model on a dataset.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    # Determine number of batches
    num_examples = len(dataset)
    num_batches_total = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE
    
    if max_batches is not None:
        num_batches_total = min(num_batches_total, max_batches)
    
    # Evaluate in batches
    for i in range(num_batches_total):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_examples)
        indices = list(range(start_idx, end_idx))
        
        inputs, labels = get_batch(dataset, batch_size=len(indices), indices=indices)
        
        with ctx:
            logits, loss = model(inputs, labels)
        
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / num_batches
    
    model.train()
    return accuracy, avg_loss


def print_sample_predictions(model, dataset, num_samples=5):
    """
    Print sample predictions for inspection.
    """
    model.eval()
    
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)
    
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    for idx in indices:
        example = dataset[int(idx)]
        text = example["sentence"]
        true_label = example["label"]
        
        # Encode and predict
        encoded = encode_text(text).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with ctx:
                logits, _ = model(encoded)
        
        pred_label = torch.argmax(logits, dim=-1).item()
        probs = torch.softmax(logits, dim=-1).squeeze()
        
        label_names = ["negative", "positive"]
        
        print(f"\nText: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"True: {label_names[true_label]} | Predicted: {label_names[pred_label]}")
        print(f"Confidence: neg={probs[0]:.3f}, pos={probs[1]:.3f}")
        print("-"*60)
    
    model.train()

# Learning Rate Scheduler
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return MAX_LR * (it + 1) / WARMUP_ITERS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BDH for text classification on SST-2.")
    parser.add_argument('--max_iters', type=int, default=MAX_ITERS,
                        help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=MAX_LR,
                        help='Learning rate')
    args = parser.parse_args()
    
    # Update config with command line args
    MAX_ITERS = args.max_iters
    BATCH_SIZE = args.batch_size
    MAX_LR = args.learning_rate
    
    # Load data
    train_dataset, val_dataset = load_sst2_data()
    
    # Initialize model
    print("\nInitializing BDH classifier...")
    model_config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=50304,  # GPT-2 vocab size (aligned to 64)
        dropout=0.0        # Disabled dropout for faster learning
    )
    
    model = BDHClassifier(model_config, num_classes=2).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    
    # Compilation
    if USE_COMPILE:
        print(f"Compiling the model...")
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, backend="aot_eager")
            print("Model compiled successfully with 'aot_eager' backend.")
        except Exception as e:
            print(f"Warning: torch.compile failed with error: {e}\nContinuing without compilation...")
    else:
        print("Compilation disabled, running in eager mode.")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    print(f"Batch size: {BATCH_SIZE} (Accumulated: {BATCH_SIZE*GRAD_ACCUM_STEPS})")
    
    model.train()
    loss_acc = 0.0
    loss_steps = 0
    best_val_accuracy = 0.0
    
    for step in range(MAX_ITERS):
        # Determine learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Gradient Accumulation Loop
        for micro_step in range(GRAD_ACCUM_STEPS):
            inputs, labels = get_batch(train_dataset, batch_size=BATCH_SIZE)
            with ctx:
                logits, loss = model(inputs, labels)
                loss = loss / GRAD_ACCUM_STEPS # Scale loss
            
            scaler.scale(loss).backward()
            loss_acc += loss.item() * GRAD_ACCUM_STEPS # Track scaled-up loss
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        loss_steps += 1
        
        # Logging
        if step > 0 and step % LOG_FREQ == 0:
            avg_loss = loss_acc / loss_steps
            
            # Calculate training accuracy on LAST batch
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                train_acc = (preds == labels).float().mean().item()
            
            print(f"Step: {step}/{MAX_ITERS} | loss: {avg_loss:.4f} | train_acc: {train_acc:.4f} | lr: {lr:.2e}")
            loss_acc = 0.0
            loss_steps = 0
        
        # Validation
        if step > 0 and step % EVAL_FREQ == 0:
            print(f"\n--- Evaluating at step {step} ---")
            val_accuracy, val_loss = evaluate(model, val_dataset)
            print(f"Validation accuracy: {val_accuracy:.4f} | Validation loss: {val_loss:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"New best validation accuracy: {best_val_accuracy:.4f}")
                torch.save(model.state_dict(), "bdh_sst2_best.pth")
            
            print("-" * 50)
        
        # Checkpointing
        if step > 0 and step % CHECKPOINT_FREQ == 0:
            print(f"\n--- Saving checkpoint at step {step} ---")
            torch.save(model.state_dict(), f"bdh_sst2_checkpoint_{step}.pth")
            print(f"Model checkpoint saved.")
            print_sample_predictions(model, val_dataset, num_samples=3)
            print("-" * 50)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training finished!")
    print("="*60)
    
    print("\nRunning final evaluation on full validation set...")
    val_accuracy, val_loss = evaluate(model, val_dataset)
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    
    # Detailed evaluation
    model.eval()
    all_preds = []
    all_labels = []
    num_examples = len(val_dataset)
    num_batches = (num_examples + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("\nGenerating predictions for confusion matrix...")
    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_examples)
        indices = list(range(start_idx, end_idx))
        inputs, labels = get_batch(val_dataset, batch_size=len(indices), indices=indices)
        with torch.no_grad(), ctx:
            logits, _ = model(inputs)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))
    
    print_sample_predictions(model, val_dataset, num_samples=10)
    
    print(f"\nSaving final model to bdh_sst2_final.pth...")
    torch.save(model.state_dict(), "bdh_sst2_final.pth")
    print("Final model saved successfully.")
    print(f"\nBest validation accuracy achieved: {best_val_accuracy:.4f}")
