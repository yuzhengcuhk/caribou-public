# CARIBOU — Artifact Evaluation README

**CARIBOU** is a privacy-preserving GNN framework that couples a **Contractive Graph Layer (CGL)** with **convergent privacy accounting**, enabling deeper aggregation under **edge-level DP (EDP)** and **node-level DP (NDP)** with superior utility.

This README guides evaluators to (1) obtain the code, (2) build the environment, and (3) reproduce the results referenced in the AE appendix: **PU** (privacy–utility), **CRV** (curves/visualizations), and **OV** (overhead).

---

## 1) Obtain the Code

We recommend cloning via HTTPS.

```bash
git clone https://github.com/yuzhengcuhk/caribou-public.git
cd caribou-public/
```

---

## 2) Build the Environment

We provide a bootstrap script that creates a conda environment and installs the core stack (PyTorch 1.13.1 + CUDA 11.7, PyG extensions, DP libraries, scientific stack).

> Run from the repository root `caribou_public/`.

```bash
chmod +x ./setup_minimal_env.sh
./setup_minimal_env.sh

# Activate and run a quick sanity check
conda activate caribou-minimal
python train.py mlp-dp --dataset cora --epsilon 2
```

- The script creates an environment **`caribou-minimal`** and installs core libraries from the CUDA 11.7 wheel index.

---

## 3) Privacy–Utility (PU) Reproduction

Convenience scripts regenerate the privacy–utility tables/figures. For AE runtime, we scope to **computers** as a quick sanity check and compare **CARIBOU** against the strongest baseline (**GAP**).

> Run from `caribou_public/`.

```bash
chmod +x AE/PU/pu1_run_scripts_computers.sh
./AE/PU/pu1_run_scripts_computers.sh   # PU-1

chmod +x AE/PU/pu2_run_scripts_computers.sh
./AE/PU/pu2_run_scripts_computers.sh   # PU-2

chmod +x AE/PU/pu3_run_scripts_computers.sh
./AE/PU/pu3_run_scripts_computers.sh   # PU-3
```

- Outputs are written under `AE_outputs/PU/`.
- Each **test** (i.e., each table entry) produces a `.txt` report containing **top** or **mean** accuracy over three runs.

**Interpreting accuracy.** Training seeds are randomized; some fluctuations are expected. The **Chains** datasets are noise-sensitive and may show larger variation. Despite this, **CARIBOU** exceeds competing baselines in most settings (Claim C1).

**Hyperparameter coverage.** To keep AE **runtime modest** as required, the released scripts sweep only a **subset** of the paper’s hyperparameters. The observed top accuracy may therefore be lower than the maxima reported in the paper. We will release all scripts in the final public version to reproduce the full experiments in this paper.

---

## 4) Curves and Visualizations (CRV)

We execute notebooks **headlessly** via `nbconvert` using the environment’s Jupyter kernel.

### 4.1 Register the kernel (one time)

```bash
# inside the caribou-minimal environment
python -m pip install jupyter
python -m pip install seaborn==0.13.2

python -m ipykernel install --user   --name caribou-minimal   --display-name "Python (caribou-minimal)"
```

### 4.2 Reproduce the figures

> Run from `caribou_public/`. 

```bash
jupyter nbconvert --to notebook --execute --inplace   --ExecutePreprocessor.kernel_name=caribou-minimal   ./AE/CRV/eps_hop_plots.ipynb   # CRV-eps, CRV-K

jupyter nbconvert --to notebook --execute --inplace   --ExecutePreprocessor.kernel_name=caribou-minimal   ./AE/CRV/degree_plots.ipynb    # CRV-D

jupyter nbconvert --to notebook --execute --inplace   --ExecutePreprocessor.kernel_name=caribou-minimal   ./AE/CRV/heatmap.ipynb         # CRV-H
```

- Outputs (plots, figures) are **written back in place** to the `.ipynb` files.  
  Open the notebooks to inspect results.
- Outputs are written under `AE_outputs/CRV/`.
- Terminal-only viewing:
  ```bash
  jupyter nbconvert --to html ./AE/CRV/eps_hop_plots.ipynb && xdg-open AE/CRV/eps_hop_plots.html
  ```

---

## 5) Overhead (OV)

Two scripts measure computational overhead under **EDP** and **NDP**, respectively.

> Run from `caribou_public/`.

```bash
chmod +x AE/OV/ove_run_scripts.sh
./AE/OV/ove_run_scripts.sh   # EDP overhead

chmod +x AE/OV/ovn_run_scripts.sh
./AE/OV/ovn_run_scripts.sh   # NDP overhead
```

- Outputs:
  - EDP → `AE_outputs/OV/ove/`
  - NDP → `AE_outputs/OV/ovn/`
- **Interpretation.** Absolute latency/memory are hardware-dependent, so numbers may differ from the paper. With sufficient compute, **qualitative trends and method orderings** should match our findings.

---

## 6) Repository Layout (High Level)

```
core/              # DP-GNN methods, models, modules, privacy, datasets, trainer, args, utils
AE/                # Artifact Evaluation assets
  PU/              # privacy–utility runners and scripts
  CRV/             # notebooks for epsilon/K/degree/heatmap figures
  OV/              # overhead (latency/memory) runners
AE_outputs/        # generated outputs (created on first run)
setup_minimal_env.sh  # environment bootstrap script
```

---

## 7) Troubleshooting

- **Permission denied on scripts**  
  `chmod +x <script>.sh` or run via `bash <script>.sh`. 
- **Wrong Jupyter kernel**  
  Register the env (`ipykernel install`) and pass `--ExecutePreprocessor.kernel_name=caribou-minimal`.
- **Missing packages**  
  Ensure the env is active: `conda activate caribou-minimal`, then `python -m pip install <pkg>`.
- **C++ runtime error (e.g., GLIBCXX not found)**  
  Install `pandas` via conda-forge and ensure `libstdcxx-ng/libgcc-ng` are present in the env.
- **Disk space issues**  
  Clear caches: `pip cache purge`, `conda clean --all -y`; remove old environments; prune large logs in `AE_outputs`.

---

## 8) License

**CARIBOU** is released under the **MIT License**. You may use, modify, and redistribute the software with attribution; the software is provided *as is*, without warranty. See `LICENSE` for the full text. Third-party dependencies remain under their respective licenses.

---

## 9) Citation

If you use CARIBOU or this artifact, please cite the paper (see the main manuscript).

---
