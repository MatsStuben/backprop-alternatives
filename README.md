# backprop-alternatives

# IDUN Quick Guide (NTNU HPC)

This guide shows exactly how to: log in, upload/update files, run GPU jobs, check logs, delete folders, and download results.

---

## 1. Login to IDUN

```bash
ssh matswm@idun-login1.hpc.ntnu.no
```

This places you on the **login node** (don’t run heavy jobs here).

---

## 2. Activate your Conda environment

```bash
source ~/venvs/backprop-alt/bin/activate
```

Verify:

```bash
which python
```

---

## 3. Upload or update files on IDUN

You always edit code locally, then push to IDUN.

### A) Sync entire project (recommended)

```bash
rsync -av --progress \
~/Documents/Kyb/Semester9/Master/myproject/ \
matswm@idun-login1.hpc.ntnu.no:~/myproject/
```

This only copies changed files.

### B) Upload a single file

```bash
scp path/to/your_file.py \
matswm@idun-login1.hpc.ntnu.no:~/myproject/
```

---

## 4. SLURM job file

Edit with:

```bash
nano run_gpu.slurm
```

Example content:

```bash
#!/bin/bash
#SBATCH --job-name=backprop
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

source ~/venvs/backprop-alt/bin/activate
cd ~/myproject
python your_script.py
```

Save with: **CTRL+O**, then **ENTER**, then **CTRL+X**.

---

## 5. Submit a GPU job

```bash
sbatch run_gpu.slurm
```

Check status:

```bash
squeue -u matswm
```

Watch job output:

```bash
tail -f logs/<filename>.out
```

---

## 6. Delete old plots

Delete everything in the folder:

```bash
rm -rf ~/myproject/plots/*
```

Or recreate folder:

```bash
rm -rf ~/myproject/plots
mkdir ~/myproject/plots
```

---

## 7. Download plots/results to your Mac

To your Master folder:

```bash
scp -r matswm@idun-login1.hpc.ntnu.no:~/myproject/plots \
~/Documents/Kyb/Semester9/Master/plots
```

If folder missing locally:

```bash
mkdir -p ~/Documents/Kyb/Semester9/Master/plots
```

---

## 8. Project structure on IDUN

```
~/myproject/         # your code
~/myproject/plots/   # generated figures
~/myproject/logs/    # SLURM .out logs
~/venvs/backprop-alt # environment
```

---

## 9. Typical workflow

1. Edit code locally.
2. Upload to IDUN via rsync:

   ```bash
   rsync -av --progress ~/Documents/Kyb/Semester9/Master/myproject/ \
   matswm@idun-login1.hpc.ntnu.no:~/myproject/
   ```
3. SSH into IDUN.
4. Activate env:

   ```bash
   source ~/venvs/backprop-alt/bin/activate
   ```
5. Submit GPU job:

   ```bash
   sbatch run_gpu.slurm
   ```
6. View progress:

   ```bash
   ```

tail -f logs/*.out

````
7. Download plots:
```bash
scp -r matswm@idun-login1.hpc.ntnu.no:~/myproject/plots \
~/Documents/Kyb/Semester9/Master/plots
````
