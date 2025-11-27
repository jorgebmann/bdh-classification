import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bdh
from bdh import BDHConfig, BDHClassifier

# --- Configuration ---
MODEL_PATH = "bdh_sst2_best.pth"
TEXT = "The visual effects were stunning but the plot was boring."
LAYER_IDX = 5  # Look at the last layer
HEAD_IDX = 0   # Look at the first head

def load_model():
    config = BDHConfig(n_layer=6, n_embd=256, n_head=4, vocab_size=256, dropout=0.1)
    model = BDHClassifier(config, num_classes=2)
    
    if os.path.exists(MODEL_PATH):
        # Robust loading that handles both compiled and uncompiled models
        try:
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random weights instead.")
    else:
        print(f"Warning: Model checkpoint '{MODEL_PATH}' not found. Using random weights.")
    
    model.eval()
    return model

def get_hebbian_updates(model, text, layer_idx):
    # 1. Encode text (byte-level)
    byte_array = bytearray(text.encode('utf-8'))
    # Pad to at least a small amount if needed, but for viz we can use exact length
    # However, the model expects a batch dimension.
    inputs = torch.tensor(byte_array, dtype=torch.long).unsqueeze(0) # (1, T)
    
    # 2. Hook to capture Q (sparse), K (sparse), and V
    captured = {}
    
    def hook_fn(module, args, output):
        # args are (Q, K, V)
        captured['Q'] = args[0].detach()
        captured['K'] = args[1].detach()
        captured['V'] = args[2].detach()
    
    # Register hook on the specific attention layer
    # In bdh.py, self.attn is a single module reused in the loop!
    # We need to count calls to capture the correct layer.
    
    call_count = [0]
    def specialized_hook(module, args, output):
        if call_count[0] == layer_idx:
            hook_fn(module, args, output)
        call_count[0] += 1

    handle = model.bdh_core.attn.register_forward_hook(specialized_hook)
    
    # 3. Run Forward Pass
    with torch.no_grad():
        model(inputs)
    
    handle.remove()
    
    if not captured:
        raise ValueError(f"Could not capture attention values. Layer index {layer_idx} might be out of bounds.")
        
    return captured, inputs

def compute_plasticity(captured, head_idx=0):
    # Q, K are (B, nh, T, N)
    # V is (B, 1, T, D)
    
    K = captured['K'][0, head_idx]  # (T, N)
    V = captured['V'][0, 0]         # (T, D) - V is shared across heads/channels in this implementation details
    
    # Note: In bdh.py, V passed to attn is 'x' with shape (B, 1, T, D)
    # So V[0, 0] gives (T, D)
    
    T, N = K.shape
    _, D = V.shape
    
    # Calculate Hebbian Updates: Outer Product K^T * V at each step
    
    updates_norm = []
    memory_state_norm = []
    
    # Initialize Memory Matrix (Synapses)
    M = torch.zeros(N, D)
    
    print(f"Simulating Hebbian Plasticity for T={T} steps...")
    
    for t in range(T):
        k_t = K[t].unsqueeze(1) # (N, 1)
        v_t = V[t].unsqueeze(0) # (1, D)
        
        # The Hebbian Update (Synaptic Change)
        # "Neurons that fire together, wire together"
        delta_M = torch.matmul(k_t, v_t) # (N, D)
        
        # Update Memory
        M = M + delta_M
        
        # Record metrics
        # L2 norm of the update matrix represents the "magnitude of learning" at this step
        updates_norm.append(torch.norm(delta_M).item())
        
        # L2 norm of the memory matrix represents the "total information stored"
        memory_state_norm.append(torch.norm(M).item())
        
    return updates_norm, memory_state_norm

def plot_results(text, updates, memory):
    # Decode bytes back to chars for display
    tokens = list(text.encode('utf-8'))
    x = range(len(tokens))
    # Create readable labels
    char_labels = []
    for t in tokens:
        c = chr(t)
        # Keep printable ASCII, replace others
        if 32 <= t <= 126:
            char_labels.append(c)
        else:
            char_labels.append('?')
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Synaptic Plasticity (Rate of Change)
    sns.lineplot(x=x, y=updates, ax=ax1, color="#e74c3c", linewidth=2, marker="o", markersize=4)
    ax1.set_title(f"Synaptic Plasticity: Instantaneous Connection Strengthening (Layer {LAYER_IDX}, Head {HEAD_IDX})", fontsize=14)
    ax1.set_ylabel("Update Magnitude ||ΔM||")
    
    # Highlight spikes with text
    threshold = np.mean(updates) + 1.0 * np.std(updates)
    for i, val in enumerate(updates):
        if val > threshold:
            ax1.text(i, val, char_labels[i], ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Plot 2: Working Memory Capacity
    sns.lineplot(x=x, y=memory, ax=ax2, color="#2ecc71", linewidth=2)
    ax2.set_title("Accumulated Working Memory (Synaptic State Strength)", fontsize=14)
    ax2.set_ylabel("Memory Norm ||M||")
    ax2.set_xlabel("Token Position")
    
    # Set x-ticks to be the characters
    ax2.set_xticks(x)
    ax2.set_xticklabels(char_labels)
    
    plt.tight_layout()
    output_file = "hebbian_learning_viz.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    # plt.show() # Commented out for headless environments

if __name__ == "__main__":
    try:
        model = load_model()
        print(f"Analyzing text: '{TEXT}'")
        captured, _ = get_hebbian_updates(model, TEXT, LAYER_IDX)
        updates, memory = compute_plasticity(captured, HEAD_IDX)
        plot_results(TEXT, updates, memory)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

