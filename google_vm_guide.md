# Google VM Setup Guide

This guide provides step-by-step instructions for setting up and running the BDH classification project on a **Google Cloud VM instance**.

## Prerequisites

- A Google Cloud VM instance (Ubuntu/Debian recommended)
- SSH access to the VM
- Python 3 installed (usually pre-installed on Ubuntu VMs)

## Step-by-Step Installation

### 1. Connect to Your VM

```bash
# Use gcloud CLI or SSH directly
gcloud compute ssh <INSTANCE_NAME> --zone=<ZONE>
# Or use standard SSH
ssh <USERNAME>@<EXTERNAL_IP>
```

### 2. Install pip3 (if not already installed)

**For Ubuntu/Debian-based VMs:**

```bash
# Update package lists
sudo apt update

# Install pip3
sudo apt install python3-pip -y

# Verify installation
pip3 --version
```

**For CentOS/RHEL-based VMs:**

```bash
# Install pip3
sudo yum install python3-pip -y
# Or for newer versions:
sudo dnf install python3-pip -y
```

**Alternative: Install pip using get-pip.py**

If package manager doesn't work:

```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip
python3 get-pip.py

# Clean up
rm get-pip.py
```

### 3. Install uv using pipx (Recommended)

Modern Ubuntu/Debian systems use PEP 668 protection, which prevents installing packages system-wide. Using `pipx` avoids this issue:

```bash
# Install pipx
sudo apt install pipx -y

# Ensure pipx is in PATH
pipx ensurepath

# Install uv using pipx
pipx install uv

# Reload shell configuration (or restart SSH session)
source ~/.bashrc
# Or simply: exec $SHELL

# Verify uv is installed
uv --version
```

**Alternative: Install uv in a virtual environment**

If `pipx` is not available or you prefer a different approach:

```bash
# Create a temporary venv for uv
python3 -m venv /tmp/uv-env
source /tmp/uv-env/bin/activate

# Install uv
pip install uv

# Note: uv will be available from this venv
# You can add it to PATH or use the full path: /tmp/uv-env/bin/uv
```

### 4. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-username/bdh-classification.git
cd bdh-classification

# If your repo is private, use a Personal Access Token:
# git clone https://<TOKEN>@github.com/your-username/bdh-classification.git
```

### 5. Set Up Python Environment and Dependencies

**Option A: Using uv with virtual environment (Recommended)**

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (for RTX 4090/5080 and newer GPUs)
# For older GPUs (GTX 1060, etc.), use CUDA 11.8 instead:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies using uv (faster)
uv pip install -r requirements.txt
```

**Option B: Using uv run (Automatic environment management)**

If you installed `uv` globally with `pipx`:

```bash
# First, install PyTorch in a venv (uv run will reuse it)
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
deactivate

# Then use uv run for training (it will manage dependencies automatically)
uv run scripts/train.py --batch_size 64 --max_iters 5000
```

**Option C: Traditional pip workflow**

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 6. Verify GPU Access (if using GPU)

```bash
# Activate venv if not already activated
source venv/bin/activate

# Check if PyTorch can see the GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 7. Run Training

**Important**: Make sure your virtual environment is activated (`source venv/bin/activate`) before running training.

**Standard training:**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start training
python scripts/train.py --batch_size 64 --max_iters 5000
```

**Training with custom parameters:**

```bash
source venv/bin/activate
python scripts/train.py --batch_size 32 --max_iters 10000 --learning_rate 1e-3
```

**Background training (keeps running if you disconnect):**

```bash
source venv/bin/activate

# Run in background
nohup python scripts/train.py --batch_size 64 --max_iters 5000 > training_output.log 2>&1 &

# Check progress
tail -f training_output.log

# To stop background process, find PID and kill:
# ps aux | grep train.py
# kill <PID>
```

### 8. Monitor Training Progress

- **Console Output**: The script prints loss and accuracy every 200 steps
- **CSV Log**: A `training_log.csv` file is created and updated in real-time
  ```bash
  # View the log
  cat training_log.csv
  # Or watch it update
  watch -n 1 cat training_log.csv
  ```

### 9. Retrieve Results

After training finishes, copy the models and logs to your local machine.

**From your LOCAL terminal:**

```bash
# Download final model
gcloud compute scp <INSTANCE_NAME>:~/bdh-classification/bdh_sst2_final.pth ./

# Download best checkpoint
gcloud compute scp <INSTANCE_NAME>:~/bdh-classification/bdh_sst2_best.pth ./

# Download training log
gcloud compute scp <INSTANCE_NAME>:~/bdh-classification/training_log.csv ./

# Or use standard SCP
scp <USERNAME>@<EXTERNAL_IP>:~/bdh-classification/bdh_sst2_final.pth ./
scp <USERNAME>@<EXTERNAL_IP>:~/bdh-classification/bdh_sst2_best.pth ./
scp <USERNAME>@<EXTERNAL_IP>:~/bdh-classification/training_log.csv ./
```

### 10. Clean Up

1. **Verify**: Ensure you have all files (`.pth`, `.csv`) on your local machine
2. **Stop/Delete VM**: Go to Google Cloud Console and stop or delete the instance to stop being charged

## Troubleshooting

### "pip: command not found"
- Install pip3 using Step 2 above

### "externally-managed-environment" error
- Use `pipx` to install `uv` (Step 3) instead of installing directly with pip
- Or create a virtual environment first, then install packages there

### "uv: command not found"
- Make sure you've installed `uv` using `pipx` and reloaded your shell
- Run `source ~/.bashrc` or restart your SSH session
- Verify with `uv --version`

### CUDA/GPU not detected
- Ensure you've installed PyTorch with CUDA support (not CPU-only version)
- Check that NVIDIA drivers are installed: `nvidia-smi`
- Verify GPU is accessible: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Permission denied errors
- Use `sudo` for system-level installations (pipx, apt packages)
- For user-level installations, ensure you have write permissions to the target directory

## Quick Reference

```bash
# Complete setup (copy-paste friendly)
sudo apt update && sudo apt install python3-pip pipx -y
pipx ensurepath && pipx install uv
source ~/.bashrc
git clone https://github.com/your-username/bdh-classification.git
cd bdh-classification
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
python scripts/train.py --batch_size 64 --max_iters 5000
```

