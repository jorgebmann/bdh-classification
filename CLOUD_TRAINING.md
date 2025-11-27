# Cloud Training Guide: Vast.ai

This guide provides step-by-step instructions for deploying, training, and retrieving results using a **Vast.ai** GPU instance.

## 1. Renting a GPU Instance

1.  **Sign Up/Login**: Go to [vast.ai](https://vast.ai/) and create an account. Add credits (minimum $5-$10 is usually enough for a few hours of training).
2.  **Search for Instance**:
    *   Navigate to the "Console" or "Search" tab.
    *   **Filter**: Look for **RTX 5080** (16GB) or **RTX 4090** (24GB).
    *   **Image**: Select a pre-built PyTorch image to avoid manual setup.
        *   Recommended: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` (or similar).
    *   **Disk Space**: Allocate at least **30GB**.
3.  **Rent**: Click "Rent" on a suitable instance.
4.  **Wait for Startup**: Go to the "Instances" tab. Wait until the status is "Running" and the "Connect" button appears.

## 2. Connecting to the Instance

1.  **Get SSH Command**: Click the "Connect" button on your instance card. It will look like:
    ```bash
    ssh -p <PORT> root@<IP_ADDRESS>
    ```
2.  **Connect via Terminal**:
    *   Open your local terminal.
    *   Paste the SSH command and press Enter.
    *   (Optional) If asked for a password (and you didn't set up an SSH key), check the Vast.ai email/console for default credentials, but usually, key-based auth is preferred.

## 3. Setting Up the Codebase

Once connected to the remote terminal:

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/your-username/bdh-classification.git
    cd bdh-classification
    ```
    *(Note: If your repo is private, you may need to generate a GitHub Personal Access Token and use it in the URL: `https://<token>@github.com/...`)*

2.  **Install Dependencies (using `uv` for speed)**:
    ```bash
    # Install uv
    pip install uv

    # Install requirements
    uv pip install --system -r requirements.txt
    ```

## 4. Running Training

For a 16GB+ GPU (RTX 5080/4090), use larger batch sizes for efficiency.

1.  **Start Training**:
    ```bash
    python scripts/train.py --batch_size 64 --max_iters 5000
    ```
    *   **`--batch_size 64`**: Utilizes the large VRAM.
    *   **`--max_iters 5000`**: Reduced iterations since batch size is larger (faster convergence).

2.  **Background Training (Optional)**:
    To keep training running if you disconnect:
    ```bash
    nohup python scripts/train.py --batch_size 64 --max_iters 5000 > training_output.log 2>&1 &
    ```
    *   Check progress with: `tail -f training_output.log`

## 5. Monitoring Progress

*   **Console Output**: The script prints loss and accuracy every 200 steps.
*   **CSV Log**: A `training_log.csv` file is created and updated in real-time. You can `cat training_log.csv` to see the data.

## 6. Retrieving Results

After training finishes, copy the models and logs to your local machine.

**Run these commands from your LOCAL terminal (not the Vast.ai terminal):**

1.  **Download Final Model**:
    ```bash
    scp -P <PORT> root@<IP_ADDRESS>:/root/bdh-classification/bdh_sst2_final.pth ./
    ```

2.  **Download Best Checkpoint**:
    ```bash
    scp -P <PORT> root@<IP_ADDRESS>:/root/bdh-classification/bdh_sst2_best.pth ./
    ```

3.  **Download Training Log**:
    ```bash
    scp -P <PORT> root@<IP_ADDRESS>:/root/bdh-classification/training_log.csv ./
    ```

*Replace `<PORT>` and `<IP_ADDRESS>` with the values from your Vast.ai connect command.*

## 7. Clean Up

1.  **Verify**: Ensure you have all files (`.pth`, `.csv`) on your local machine.
2.  **Destroy Instance**: Go to the Vast.ai console and **Stop/Destroy** the instance to stop being charged.

