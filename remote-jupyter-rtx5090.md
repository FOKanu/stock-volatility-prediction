# Remote Jupyter Notebook — RTX 5090

Command sequences to run Jupyter on your RTX 5090 workbench and open it in your Mac browser.

## Connection details

| Setting | Value |
|---------|-------|
| SSH host alias | `rtx5090` |
| Tailscale IP | `100.94.185.71` |
| SSH user | `friyekekanu` |
| SSH key | `~/.ssh/id_ed25519` |
| Jupyter port | `8888` |
| Remote workspace | `/home/friyekekanu/ai-workspace` |
| This project (remote) | `/home/friyekekanu/ai-workspace/experiments/stock-performance-prediction` |
| This project (local) | `/Users/franciskanu/University/Courses/Semester 2/BIT4.2C - Programming for Data Analytics (SS 26)/Final Project - Stock Performance Prediction` |

Port `8888` is already forwarded in `~/.ssh/config` via `LocalForward 8888 127.0.0.1:8888`.

---

## Daily workflow (recommended)

Uses your existing workbench scripts.

**Terminal 1 — start SSH tunnel (keep open):**

```bash
bash ~/ai-workspace/environment/scripts/connect-workbench.sh
```

Or from the setup repo:

```bash
bash "/Users/franciskanu/Remote Server - Setup/environment/scripts/connect-workbench.sh"
```

**Terminal 2 — start Jupyter on the 5090 and open browser:**

```bash
bash ~/ai-workspace/environment/scripts/start-jupyter-remote.sh experiments/stock-performance-prediction
```

**Browser (if it does not open automatically):**

```
http://localhost:8888/lab
```

Paste the token from Terminal 2 if prompted.

**Stop Jupyter when done:**

```bash
bash ~/ai-workspace/environment/scripts/start-jupyter-remote.sh --stop
```

---

## First-time project setup on the 5090

**1. Connect to the remote machine:**

```bash
ssh rtx5090
```

**2. Verify the RTX 5090:**

```bash
nvidia-smi
```

Expected: `NVIDIA GeForce RTX 5090`

**3. Create the project folder:**

```bash
mkdir -p ~/ai-workspace/experiments/stock-performance-prediction/notebooks
cd ~/ai-workspace/experiments/stock-performance-prediction
```

**4. Create and activate a Python environment (first time only):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyterlab notebook pandas numpy matplotlib seaborn scipy torch torchvision
```

GPU PyTorch (CUDA 12.4):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**5. Exit SSH:**

```bash
exit
```

---

## Upload project files from your Mac

**Create remote folders:**

```bash
ssh rtx5090 "mkdir -p ~/ai-workspace/experiments/stock-performance-prediction/notebooks"
```

**Upload the EDA notebook:**

```bash
scp "/Users/franciskanu/University/Courses/Semester 2/BIT4.2C - Programming for Data Analytics (SS 26)/Final Project - Stock Performance Prediction/notebooks/01_exploratory_data_analysis.ipynb" rtx5090:~/ai-workspace/experiments/stock-performance-prediction/notebooks/
```

**Upload the full project (optional):**

```bash
scp -r "/Users/franciskanu/University/Courses/Semester 2/BIT4.2C - Programming for Data Analytics (SS 26)/Final Project - Stock Performance Prediction/notebooks" rtx5090:~/ai-workspace/experiments/stock-performance-prediction/

scp -r "/Users/franciskanu/University/Courses/Semester 2/BIT4.2C - Programming for Data Analytics (SS 26)/Final Project - Stock Performance Prediction/Stock Market Dataset" rtx5090:~/ai-workspace/experiments/stock-performance-prediction/
```

---

## Manual start (without helper scripts)

**Terminal 1 — tunnel:**

```bash
ssh -N rtx5090
```

**Terminal 2 — start Jupyter on remote:**

```bash
ssh rtx5090 "cd ~/ai-workspace/experiments/stock-performance-prediction && source .venv/bin/activate && jupyter lab --no-browser --port 8888 --ip 127.0.0.1"
```

**Browser:**

```
http://localhost:8888/lab
```

---

## Confirm GPU inside a notebook

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected:

```
True
NVIDIA GeForce RTX 5090
```

---

## Preflight checks

**Test SSH (should print `ok` without password):**

```bash
ssh rtx5090 'echo ok'
```

**Run Mac setup checklist:**

```bash
cd "/Users/franciskanu/Remote Server - Setup"
bash environment/scripts/mac-client-setup.sh --checklist
```

**Ensure Tailscale is connected on your Mac before connecting.**

---

## Troubleshooting

| Problem | Action |
|---------|--------|
| `Connection refused` | Ensure Tailscale is up; remote OpenSSH running in WSL |
| Password prompt every time | `bash "/Users/franciskanu/Remote Server - Setup/environment/scripts/mac-client-setup.sh" --copy-key 100.94.185.71` |
| Tunnel up, Jupyter won't load | Start Jupyter: `bash ~/ai-workspace/environment/scripts/start-jupyter-remote.sh experiments/stock-performance-prediction` |
| Port 8888 in use on Mac | `pkill -f "ssh -N rtx5090"` then restart tunnel |
| Jupyter log on remote | `~/ai-workspace/logs/environment/jupyter.log` |

---

## Quick reference

```bash
# 1. Tunnel
bash ~/ai-workspace/environment/scripts/connect-workbench.sh

# 2. Jupyter (new terminal)
bash ~/ai-workspace/environment/scripts/start-jupyter-remote.sh experiments/stock-performance-prediction

# 3. Browser
open http://localhost:8888/lab
```
