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
| **Project (WSL)** | `~/ai-workspace/experiments/stock-prediction` |
| **TensorFlow GPU venv** | `~/ai-workspace/experiments/stock-prediction/.venv` |
| **Jupyter kernel** | `Python (stock-prediction-wsl)` |

Port `8888` is forwarded in `~/.ssh/config` via `LocalForward 8888 127.0.0.1:8888`.

---

## Daily workflow — TensorFlow TCN / BiGRU training

**Terminal 1 — SSH tunnel (keep open):**

```bash
bash ~/ai-workspace/environment/scripts/connect-workbench.sh
```

**Terminal 2 — start Jupyter with stock-prediction venv:**

```bash
ssh rtx5090 bash -s <<'EOF'
source ~/ai-workspace/experiments/stock-prediction/.venv/bin/activate
cd ~/ai-workspace/experiments/stock-prediction
jupyter lab --no-browser --port 8888 --ip 127.0.0.1
EOF
```

Or use the helper script on the 5090:

```bash
bash ~/ai-workspace/environment/scripts/start-jupyter-stock-tf.sh
```

**Browser:**

```
http://localhost:8888/lab
```

Paste the token from Terminal 2 if prompted.

**In JupyterLab:**

1. Open `04_ML_Model_Iteration-1/03b_model_tcn.ipynb`
2. **Kernel → Change Kernel → Python (stock-prediction-wsl)**
3. **Kernel → Restart Kernel**
4. **Run → Run All Cells**

**Stop Jupyter when done:**

```bash
ssh rtx5090 "pkill -f 'jupyter-lab.*--port 8888'"
```

---

## TensorFlow GPU setup (stock-prediction venv)

Dedicated venv at `~/ai-workspace/experiments/stock-prediction/.venv` — separate from PyTorch in `model-testing/.venv`.

| Package | Version |
|---------|---------|
| tensorflow | 2.21.0 |
| nvidia-cudnn-cu12 | ≥ 9.3 (installed: 9.23) |
| nvidia-cuda-nvcc-cu12 | ≥ 12.5 (libdevice for sm_120 JIT) |

**Why a separate venv?** PyTorch in `model-testing` pins `nvidia-cudnn-cu12==9.1.0`, which breaks TensorFlow cuDNN init. `stock-prediction/.venv` has no PyTorch.

**GPU env vars** (set in venv activate + Jupyter kernel `kernel.json`):

- `LD_LIBRARY_PATH` — WSL `/usr/lib/wsl/lib` + pip `nvidia/*/lib` paths
- `XLA_FLAGS=--xla_gpu_cuda_data_dir=.../nvidia/cuda_nvcc` — required for RTX 5090 Conv1D
- `TF_FORCE_GPU_ALLOW_GROWTH=true`

**Expected setup cell output:**

```
GPU Conv1D probe: OK
TensorFlow: 2.21.0
Python:     .../stock-prediction/.venv/bin/python
Device:     GPU (GPU:0)
```

**First training epoch:** TF JIT-compiles PTX for sm_120 (Blackwell). The first `model.fit` may take several minutes; later epochs are faster.

---

## Confirm GPU inside a notebook

The TCN notebook setup cell runs a Conv1D + `model.fit` probe automatically and **raises** if GPU is unavailable.

Manual check:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Expected: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

## First-time project setup on the 5090

**1. Verify GPU:**

```bash
ssh rtx5090 '/usr/lib/wsl/lib/nvidia-smi'
```

Expected: `NVIDIA GeForce RTX 5090`

**2. Create venv (already done after WSL migration):**

```bash
ssh rtx5090 bash -s <<'EOF'
cd ~/ai-workspace/experiments/stock-prediction
bash scripts/setup-wsl-venv.sh
EOF
```

**3. Project location (WSL native — preferred over /mnt/c):**

```bash
~/ai-workspace/experiments/stock-prediction
```

Processed data must exist at `data/processed/aapl_sequences.npz` (run `Analytics notebooks/02_preprocessing.ipynb` first).

---

## Upload project files from Mac

```bash
scp -r "/path/to/Stock-Prediction--/04_ML_Model_Iteration-1" rtx5090:~/ai-workspace/experiments/stock-prediction/
scp -r "/path/to/Stock-Prediction--/data/processed" rtx5090:~/ai-workspace/experiments/stock-prediction/data/
```

---

## Preflight checks

```bash
ssh rtx5090 'echo ok'
ssh rtx5090 'source ~/ai-workspace/experiments/stock-prediction/.venv/bin/activate && export LD_LIBRARY_PATH=... && python -c "import tensorflow as tf; print(bool(tf.config.list_physical_devices(\"GPU\")))"'
```

---

## Troubleshooting

| Problem | Action |
|---------|--------|
| `GPU available: False` | Kernel → **Python (stock-prediction-wsl)** → Restart Kernel |
| `DNN library initialization failed` | Wrong cuDNN (9.1 from PyTorch venv). Use `stock-prediction-wsl` kernel, not `model-testing` |
| `Autotuner could not compile` / libdevice error | Ensure `nvidia-cuda-nvcc-cu12` installed; kernel has `XLA_FLAGS` set |
| `Connection refused` | Tailscale up; tunnel running: `connect-workbench.sh` |
| Port 8888 in use on Mac | `pkill -f "ssh -N rtx5090"` then restart tunnel |
| Project root wrong in notebook | Notebook resolves via `data/processed/aapl_sequences.npz`; cwd should be repo root or `04_ML_Model_Iteration-1/` |
| Jupyter log | `~/ai-workspace/logs/environment/jupyter-stock-tf.log` |

---

## Quick reference

```bash
# 1. Tunnel
bash ~/ai-workspace/environment/scripts/connect-workbench.sh

# 2. Jupyter (stock-prediction project + venv)
bash ~/ai-workspace/environment/scripts/start-jupyter-stock-tf.sh

# 3. Browser
open http://localhost:8888/lab

# 4. In notebook: Kernel → Python (stock-prediction-wsl) → Restart → Run All
```
