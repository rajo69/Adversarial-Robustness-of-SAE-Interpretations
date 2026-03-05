# CLAUDE.md — Adversarial Robustness of SAE Interpretations

## Project Overview
Empirical investigation of whether Sparse Autoencoder (SAE) interpretations of language
model internals are robust to adversarial input perturbations. Three experiments:
1. Feature stability under PGD attack (Exp 1)
2. Output-preserving adversarial attacks (Exp 2 — novel contribution)
3. Layer-wise sensitivity profiling (Exp 3)

## Model & SAE Configuration
- Primary: Gemma 2 2B + Gemma Scope SAEs (residual stream, 16k width, medium L0)
- Fallback: GPT-2 Small + SAE Lens pretrained SAEs
- Hardware: Google Colab T4 (16GB VRAM)
- Evaluation data: WikiText-2 test set, 100-200 prompts, 64-128 tokens each

## Code Conventions
- Python 3.10+, PyTorch
- Type hints on all functions
- Docstrings on all functions (numpy style)
- torch.no_grad() wherever gradients aren't needed
- .detach() intermediate tensors in attack loops to prevent memory leaks
- Fixed random seed: 42 everywhere (torch, numpy, random)
- Save all raw results to results/ as JSON
- Save all figures to figures/ at 300 DPI
- Pin library versions in requirements.txt
- Progress bars (tqdm) on all loops over prompts

## File Structure
- notebooks/ — Jupyter notebooks 01-04 (one per experiment + setup)
- src/ — Reusable modules (attacks.py, metrics.py, sae_utils.py, eval_utils.py, data.py, plot_config.py)
- figures/ — All generated plots
- results/ — Raw JSON/CSV results for reproducibility
- analysis/ — Post-hoc analysis (insights.md, summary_statistics.json)
- tests/ — Unit tests (pytest)
- data/ — Preprocessed evaluation prompts
- README.md — Full writeup (mirrors gated attention project structure)

## Key Design Decisions Log
(Update this as decisions are made)
- [ ] Model choice confirmed (Gemma 2 2B or GPT-2 fallback?)
- [ ] SAE layer and width selected
- [ ] Epsilon range calibrated: [0.01, 0.05, 0.1, 0.5]
- [ ] Output-preservation threshold (delta) values set: [0.01, 0.1, 0.5, 1.0]

## Current Status
- [ ] Notebook 01: Setup and baseline
- [ ] Notebook 02: Feature stability (Exp 1)
- [ ] Notebook 03: Output-preserving attacks (Exp 2)
- [ ] Notebook 04: Layer-wise sensitivity (Exp 3)
- [ ] Results analysis (analysis/insights.md)
- [ ] README writeup
- [ ] Code cleanup and final review

## Known Issues / Blockers
(Update as encountered)

## Commands
- Run all tests: `pytest tests/ -v`
- Install deps: `uv pip install -r requirements.txt`
- Lint: `ruff check src/ tests/`
- Validate notebooks (local, CPU only): `jupyter nbconvert --to notebook --execute notebooks/01_setup_and_baseline.ipynb --ExecutePreprocessor.timeout=60`

## Environment Notes
- Local machine: CPU only (Windows 11). No GPU execution locally.
- GPU execution: Google Colab free tier, T4 (16GB VRAM).
- Local package management: uv venv + uv pip install
- Workflow: write code locally → push to GitHub → execute in Colab → pull results back
