# Step-by-Step Guide to Creating a Conda Environment on the Ohio Supercomputer Center (OSC)

This guide assumes you have an active OSC account (request one at [osc.edu](https://www.osc.edu/) if needed). We'll walk through logging into OSC's OnDemand portal, launching a remote desktop session on the Ascend cluster, creating a Conda environment from a repository's environment.yml file, and setting up Jupyter notebooks. All steps use a bash shell.

---

## Step 1: Log in to OSC OnDemand

- Open the OSC OnDemand portal in a web browser and log in with your OSC credentials.
    
- From the top navigation bar, choose **Interactive Apps**.
    

---

## Step 2: Launch an Ascend Desktop Session

- In the dropdown under Interactive Apps, select **Ascend Desktop**.
    

### Select Session Parameters

- **Desktop Environment:** Choose **Xfce**
    
- **Project/Account:** Select the appropriate project number from the list. This determines where compute time is billed.
    
- **Node Type:** Choose **cpu** (default)
    
- **Number of Nodes/Cores/GPUs:** For basic environment setup, one node with a small number of cores is sufficient.
    
- **Runtime:** Enter the maximum wall time (e.g., 2 hours) your session should last.
    

### Launch the Session

- Click **Launch** at the bottom of the form.
    
- Wait for the session status to change from "Queued" to "Running". This can take a few moments depending on resource availability.
    
- When the session is running, click the **Connect** or **Launch Desktop** button. A new window or tab will open with your remote desktop.
    

---

## Step 3: Open a Terminal

- Inside the remote desktop, open a terminal application.
    
- You now have shell access on the compute node.
    

---

## Step 4: Load the Miniconda Module

- Load Miniconda to access Conda tools:
    
    ```bash
    module load miniconda3/24.1.2-py310
    conda activate
    ```
    

---

## Step 5: Configure Conda (First-Time Setup Only)

- Optimize Conda channels for compatibility:
    
    ```bash
    conda config --remove channels defaults
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    ```
    
- If dependency issues occur, try:
    
    ```bash
    conda config --set channel_priority flexible
    ```
    

---

## Step 6: Clone a Repository and Create the Conda Environment

### Clone the Repository

- Clone the repository.
    
    ```bash
    git clone https://github.com/biogabriel7/colok-roll.git
    ```
    

### Navigate to the Repository Directory

- Change to the repository directory:
    
    ```bash
    cd colok-roll
    ```
    

### Verify the Environment File

- Verify the environment.yml exists:
    
    ```bash
    ls environment.yml
    ```
    
- It should define the environment name and dependencies, e.g.:
    
    ```yaml
    name: colok-roll
    channels:
      - conda-forge
    dependencies:
      - python=3.10
      - numpy=1.24
      - pandas
    ```
    

### Create the Environment

- Create the environment from the environment.yml file:
    
    ```bash
    conda env create -f environment.yml
    ```
    
- This creates an environment named `colok-roll`.
    

### Confirm Creation

- Confirm creation by listing all environments:
    
    ```bash
    conda info --envs
    ```
    
- The `colok-roll` environment should appear in the list.
    

---

## Step 7: Activate and Test the Environment

### Activate the Environment

- Activate the environment:
    
    ```bash
    conda activate colok-roll
    ```
    

### Test the Installation

- Test that packages are correctly installed:
    
    ```bash
    python -c "import numpy; print(numpy.__version__)"
    ```
    
- Adjust the test command for your specific packages. The path should point to your environment.
    

---

## Step 8: Set Up Jupyter with the Conda Environment

### Install Jupyter in Your Environment

- Ensure Jupyter is installed in your environment:
    
    ```bash
    conda install jupyter
    ```
    

### Register the Environment as a Jupyter Kernel

- Install ipykernel:
    
    ```bash
    conda install ipykernel
    ```
    
- Register the environment as a Jupyter kernel:
    
    ```bash
    python -m ipykernel install --user --name colokroll --display-name "ColokRoll"
    ```
    
- This makes `colokroll` available as a kernel in Jupyter (visible as "ColokRoll").
    

### Access Jupyter via OSC's OnDemand Portal

- Log in to OnDemand.
    
- Navigate to **Interactive Apps > Jupyter Notebook**.
    
- Select your cluster (**Ascend**), set resources (e.g., 1 node, 1 core, 1 hour), and select **Cuda 13.3**.
    
    
- Launch the session. In the Jupyter interface, select "ColokRoll" from the kernel dropdown to use your environment.
    

### Important Note

- Run Jupyter jobs via OnDemand or SLURM to avoid overloading login nodes.
    

---

## Step 9: Additional Tips and Troubleshooting

### Manage the Environment

- **Deactivate the environment:**
    
    ```bash
    conda deactivate
    ```
    
- **List packages in the environment:**
    
    ```bash
    conda list -n colokroll
    ```
    
- **Remove the environment:**
    
    ```bash
    conda remove -n colokroll --all
    ```
    
- **Update all packages:** Activate the environment first, then run:
    
    ```bash
    conda update --all
    ```
    

### Avoid .bashrc Conflicts

- Comment out Conda initialization lines in `~/.bashrc` (between `# >>> conda initialize >>>` and `# <<< conda initialize <<<`) for cleaner sessions.
    

### Storage Considerations

- Use `$HOME` (~10-50GB quota) for personal environments.
    
- Use project spaces (e.g., `/fs/ess/P00000000/`) for team environments with larger storage needs.
    

### Troubleshooting

- **Module conflicts:** Run `module purge` to clear all loaded modules before loading Miniconda.
    
- **Check Conda information:**
    
    ```bash
    conda info
    ```
    
- **Get help:** Email support@osc.edu or use OnDemand's web terminal for assistance.
    

### Running Compute Jobs

- Use `srun` or `sbatch` for computational tasks.
    
- Avoid running heavy workloads on login nodes.
    

---

**Summary:** You now have a functional Conda environment on OSC, accessible through both the command line and Jupyter notebooks. This setup allows you to manage dependencies efficiently and run your code in an isolated, reproducible environment.
