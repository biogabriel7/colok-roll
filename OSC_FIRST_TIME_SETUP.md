# Step-by-Step Guide to Creating a Conda Environment on the Ohio Supercomputer Center (OSC)

This guide assumes you have an active OSC account (request one at [osc.edu](https://www.osc.edu/) if needed). We'll walk through logging into OSC's OnDemand portal, opening a terminal, creating a Conda environment from a repository's environment.yml file, and setting up Jupyter notebooks.

---

## Step 1: Log in to OSC OnDemand

- Open the OSC OnDemand portal: [https://ondemand.osc.edu](https://ondemand.osc.edu)
- Log in with your OSC credentials.

---

## Step 2: Open a Terminal

- From the top navigation bar, click **Clusters**.
- Select **Ascend Shell Access** from the dropdown.
- A new browser tab will open with a terminal connected to OSC.

> **Note:** This connects you to a login node, which is appropriate for environment setup tasks. Heavy computational work should use compute nodes via Jupyter or SLURM.

---

## Step 3: Load the Miniconda Module

Load Miniconda to access Conda tools:

```bash
module load miniconda3/24.1.2-py310
conda activate
```

---

## Step 4: Configure Conda (First-Time Setup Only)

Optimize Conda channels for compatibility:

```bash
conda config --remove channels defaults
conda config --add channels conda-forge
conda config --set channel_priority strict
```

If dependency issues occur later, try:

```bash
conda config --set channel_priority flexible
```

---

## Step 5: Clone the Repository and Create the Conda Environment

### Clone the Repository

```bash
git clone https://github.com/biogabriel7/colok-roll.git
```

### Navigate to the Repository Directory

```bash
cd colok-roll
```

### Verify the Environment File Exists

```bash
ls environment.yml
```

### Create the Environment

This step takes several minutes. The command reads `environment.yml` and installs all required packages:

```bash
conda env create -f environment.yml
```

### Confirm Creation

List all environments to verify:

```bash
conda info --envs
```

You should see `colok-roll` in the list.

---

## Step 6: Activate and Test the Environment

### Activate the Environment

```bash
conda activate colok-roll
```

### Test the Installation

Verify that key packages are installed and GPU is available:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import cellpose; print('Cellpose: OK')"
```

> **Expected output:** `CUDA available: True`. If it shows `False`, the GPU packages may not have installed correctly.

---

## Step 7: Set Up Jupyter with the Conda Environment

### Install ipykernel

With the `colok-roll` environment still active:

```bash
conda install -y ipykernel
```

### Register the Environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name colok-roll --display-name "ColokRoll"
```

This makes `colok-roll` available as a kernel in Jupyter (displayed as "ColokRoll").

---

## Step 8: Launch Jupyter via OnDemand

1. Return to the OnDemand portal: [https://ondemand.osc.edu](https://ondemand.osc.edu)
2. Click **Interactive Apps** in the top navigation bar.
3. Select **Jupyter Notebook**.

### Configure the Jupyter Session

| Setting | Value |
|---------|-------|
| Cluster | **Ascend** |
| Project | Your project code |
| Number of cores | 4 |
| Amount of memory (GB) | 16 |
| Number of GPUs | **1** (required for Cellpose) |
| Wall time (hours) | 2 |

4. Click **Launch**.
5. Wait for the session to start (status changes from "Queued" to "Running").
6. Click **Connect to Jupyter**.

### Select the ColokRoll Kernel

- When creating a new notebook or opening an existing one, select **ColokRoll** from the kernel dropdown.

---

## Step 9: Verify Everything Works

In Jupyter, create a new notebook with the ColokRoll kernel and run:

```python
import torch
import numpy as np
import cellpose
from cellpose import models

print("=== Environment Check ===")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Cellpose: OK")
print("=== Setup complete! ===")
```

**Expected output:** `CUDA available: True` and a GPU name (e.g., `NVIDIA A100`). If CUDA shows `False`, check that you requested at least 1 GPU when launching the Jupyter session.

---

## Troubleshooting

### "conda: command not found"

Reload the module:

```bash
module load miniconda3/24.1.2-py310
conda activate
```

### Environment creation fails with dependency conflicts

Try flexible channel priority:

```bash
conda config --set channel_priority flexible
conda env create -f environment.yml
```

### Kernel not showing in Jupyter

Re-register the kernel:

```bash
conda activate colok-roll
python -m ipykernel install --user --name colok-roll --display-name "ColokRoll"
```

Then refresh the Jupyter page.

### Module conflicts

Clear all modules before loading Miniconda:

```bash
module purge
module load miniconda3/24.1.2-py310
```

### Need help?

- Email: support@osc.edu
- OSC documentation: [https://www.osc.edu/resources/technical_support](https://www.osc.edu/resources/technical_support)

---

## Quick Reference: Daily Use

After initial setup, your daily workflow is:

1. Go to [ondemand.osc.edu](https://ondemand.osc.edu)
2. **Interactive Apps → Jupyter Notebook**
3. Configure resources, click **Launch**
4. Select **ColokRoll** kernel
5. Run your analysis

---

## Managing Your Environment

**Deactivate:**
```bash
conda deactivate
```

**List installed packages:**
```bash
conda list -n colok-roll
```

**Update all packages:**
```bash
conda activate colok-roll
conda update --all
```

**Remove the environment (if needed):**
```bash
conda remove -n colok-roll --all
```

---

**Summary:** You now have a functional Conda environment on OSC, accessible through Jupyter notebooks. This setup allows you to manage dependencies efficiently and run your code in an isolated, reproducible environment.
