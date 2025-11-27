import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from bdh import BDHConfig, BDHClassifier
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tiktoken

# --- Configuration ---
BATCH_SIZE = 8
BLOCK_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "bdh_sst2_best.pth"

print(f"Using device: {DEVICE}")

# Initialize tokenizer globally
enc = tiktoken.get_encoding("gpt2")

def load_model(model_path, device):
    """Initialize model and load weights."""
    print(f"Loading model from {model_path}...")
    
    # Initialize config and model
    config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=50304, # Updated to match training
        dropout=0.0
    )
    model = BDHClassifier(config, num_classes=2)
    
    # Load state dict with prefix handling
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Fix keys if they have _orig_mod prefix (from torch.compile)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run train.py first to train the model.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    model.to(device)
    model.eval()
    return model

def encode_text(text, max_length=BLOCK_SIZE):
    """Convert text to BPE tokens using tiktoken (GPT-2 encoding)."""
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

def get_validation_data():
    """Load SST-2 validation set."""
    print("Loading SST-2 validation dataset...")
    dataset = load_dataset("glue", "sst2", split="validation")
    print(f"Loaded {len(dataset)} validation examples.")
    return dataset

def evaluate_dataset(model, dataset, batch_size=BATCH_SIZE, device=DEVICE):
    """Run prediction on the full dataset and return true/predicted labels."""
    print("Running evaluation...")
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    # Process in batches
    num_examples = len(dataset)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Evaluating"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        
        batch_texts = dataset[start_idx:end_idx]["sentence"]
        batch_labels = dataset[start_idx:end_idx]["label"]
        
        # Encode batch
        encoded_inputs = torch.stack([encode_text(t) for t in batch_texts])
        encoded_inputs = encoded_inputs.to(device)
        
        with torch.no_grad():
            logits, _ = model(encoded_inputs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_labels)
        
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def analyze_results(true_labels, pred_labels, pred_probs, dataset):
    """Print metrics and analyze errors."""
    print("\n" + "="*60)
    print("ANALYSIS REPORT")
    print("="*60)
    
    # 1. Overall Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"]))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
    
    # 3. Error Analysis
    print("\n" + "-"*60)
    print("TOP ERROR ANALYSIS")
    print("-" * 60)
    
    # Find high confidence errors
    errors = []
    for i, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, pred_probs)):
        if true != pred:
            confidence = prob[pred]
            errors.append({
                "index": i,
                "text": dataset[i]["sentence"],
                "true": "Positive" if true == 1 else "Negative",
                "pred": "Positive" if pred == 1 else "Negative",
                "confidence": confidence
            })
    
    # Sort by confidence (descending)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    
    print(f"Total Misclassifications: {len(errors)}")
    print("\nTop 5 Most Confident Errors (Model was 'sure' but wrong):")
    
    for i, err in enumerate(errors[:5]):
        print(f"\n{i+1}. Confidence: {err['confidence']:.4f}")
        print(f"   Text: \"{err['text']}\"")
        print(f"   Predicted: {err['pred']} | True: {err['true']}")

    # 4. Correct Analysis
    print("\n" + "-"*60)
    print("SUCCESS ANALYSIS")
    print("-" * 60)
    
    # Find high confidence correct predictions
    corrects = []
    for i, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, pred_probs)):
        if true == pred:
            confidence = prob[pred]
            corrects.append({
                "index": i,
                "text": dataset[i]["sentence"],
                "label": "Positive" if true == 1 else "Negative",
                "confidence": confidence
            })
            
    corrects.sort(key=lambda x: x["confidence"], reverse=True)
    
    print("\nTop 5 Most Confident Correct Predictions:")
    for i, corr in enumerate(corrects[:5]):
        print(f"\n{i+1}. Confidence: {corr['confidence']:.4f}")
        print(f"   Text: \"{corr['text']}\"")
        print(f"   Label: {corr['label']}")

def main():
    # Load resources
    model = load_model(MODEL_PATH, DEVICE)
    dataset = get_validation_data()
    
    # Evaluate
    true_labels, pred_labels, pred_probs = evaluate_dataset(model, dataset)
    
    # Analyze
    analyze_results(true_labels, pred_labels, pred_probs, dataset)
    
    # Interactive Mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Type a sentence to classify (or 'q' to quit)")
    print("="*60)
    
    while True:
        try:
            text = input("\nEnter text: ")
            if text.lower() in ['q', 'quit', 'exit']:
                break
            
            encoded = encode_text(text).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(encoded)
                probs = torch.softmax(logits, dim=-1)[0]
                pred = torch.argmax(logits, dim=-1).item()
            
            label = "Positive" if pred == 1 else "Negative"
            conf = probs[pred].item()
            
            print(f"Prediction: {label}")
            print(f"Confidence: {conf:.4f}")
            print(f"Probs: Neg={probs[0]:.4f}, Pos={probs[1]:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
