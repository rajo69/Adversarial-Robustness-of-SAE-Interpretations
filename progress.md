# Progress Log — SAE Adversarial Robustness

## Session 1: 2026-03-05
### Goals
- [x] Set up project directory structure
- [x] Create CLAUDE.md, progress.md, requirements.txt
- [x] Implement src/ modules (attacks, metrics, sae_utils, eval_utils, data)
- [ ] Load model and SAE in Colab
- [ ] Verify baseline metrics

### What happened
- Project scaffolded by Claude Code.
- All src/ modules written with type hints, docstrings, and unit tests.
- Directory structure: notebooks/, src/, tests/, figures/, results/, analysis/, data/
- Requirements pinned in requirements.txt.
- No GPU code executed locally; all execution will happen in Colab.

### Key numbers
(To be filled after Colab run: baseline perplexity, L0, reconstruction error, memory usage)

### Decisions made
- Model: Gemma 2 2B (primary), GPT-2 Small (fallback if OOM on T4)
- SAE: Gemma Scope residual stream, layer 12, width 16k, average_l0_82
- Epsilon range: [0.01, 0.05, 0.1, 0.5]
- Output-preservation thresholds (delta): [0.01, 0.1, 0.5, 1.0]
- Evaluation set: WikiText-2 test set, 150 prompts, 64-128 tokens, seed=42
- PGD: 20 steps, step_size = epsilon / 4

### Next session priorities
- Open Colab, mount Drive, run Notebook 01 (setup + baseline)
- Confirm model + SAE fit within 16GB VRAM
- Record baseline L0 and reconstruction FVU
- If OOM, immediately fall back to GPT-2 Small (do not debug OOM)

---

## Session 2: [DATE]
### Goals
- [ ] Run Notebook 02 (Experiment 1: Feature Stability)
- [ ] Produce cosine similarity vs epsilon plot

### What happened
(Fill in)

### Key numbers
(Fill in)

### Decisions made
(Fill in)

### Next session priorities
(Fill in)

---

## Session 3: [DATE]
### Goals
- [ ] Run Notebook 03 (Experiment 2: Output-Preserving Attack)

### What happened
(Fill in)

### Key numbers
(Fill in)

### Decisions made
(Fill in)

### Next session priorities
(Fill in)

---

## Session 4: [DATE]
### Goals
- [ ] Run Notebook 04 (Experiment 3: Layer-wise Sensitivity)

### What happened
(Fill in)

### Key numbers
(Fill in)

### Decisions made
(Fill in)

### Next session priorities
(Fill in)

---

## Session 5: [DATE]
### Goals
- [ ] Analysis pass: spawn analyst agent on results/
- [ ] Write README

### What happened
(Fill in)

### Key numbers
(Fill in)

### Decisions made
(Fill in)

### Next session priorities
(Fill in)
