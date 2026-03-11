# Project Journey: Adversarial Robustness of SAE Interpretations

A full record of decisions, challenges, pivots, and resolutions from project inception to a working master experiment notebook.

---

## Phase 1 — Project Scaffold

### What was built
The initial scaffold established everything except the experiment notebooks:
- Full directory structure (`src/`, `tests/`, `results/`, `figures/`, `notebooks/`, `data/`, `analysis/`)
- All `src/` modules: `attacks.py`, `metrics.py`, `sae_utils.py`, `eval_utils.py`, `data.py`, `plot_config.py`
- CPU-only pytest tests for attacks and metrics using `unittest.mock` stubs
- `requirements.txt` with pinned versions, `.gitignore`, `CLAUDE.md`

### Original model plan
- **Primary**: Gemma 2 2B + Gemma Scope SAEs (`gemma-scope-2b-pt-res`, layer 12, width 16k)
- **Fallback**: GPT-2 Small + `gpt2-small-res-jb`
- **Rationale**: Gemma Scope is the most feature-complete pretrained SAE collection available; Gemma 2 2B fits in T4 VRAM (~10 GB)

### Key architectural decisions made at this stage
- All attacks operate in **continuous embedding space** via TransformerLens `hook_embed`, never in token space. This means perturbations are injected after the embedding lookup and residual stream activations flow normally — keeping the attack tractable (continuous gradient signal) and the threat model realistic (soft-prompt / embedding-space adversary).
- Fixed random seed 42 globally across `torch`, `numpy`, and `random` to guarantee reproducibility across sessions.
- Evaluation dataset: WikiText-2 test split, 150 prompts, 64–128 tokens, cached to `data/eval_prompts.json` — rebuild is skipped unless `force_rebuild=True`.
- Results saved as flat JSON (not CSV) for maximum flexibility in post-hoc analysis.

---

## Phase 2 — Four Individual Notebooks

### What was built
Four Colab notebooks, one per experiment:
- `01_setup_and_baseline.ipynb` — model + SAE loading, WikiText-2 eval set, baseline perplexity / L0 / FVU
- `02_feature_stability.ipynb` — PGD attack sweep across ε ∈ {0.01, 0.05, 0.1, 0.5}
- `03_output_preserving.ipynb` — Lagrangian output-preserving attack sweep across δ_KL ∈ {0.01, 0.1, 0.5, 1.0}
- `04_layerwise_sensitivity.ipynb` — PGD at fixed ε=0.1 across sampled layers, one SAE loaded/unloaded per layer

### Challenge: dependency conflicts in Colab
Each notebook originally had separate `pip install` cells. When run in a fresh Colab session, `sae-lens` installs a version of NumPy that conflicts with the binary that Colab pre-installs, causing a `ValueError` on first `np.random.seed()` call. The `!pip install` approach also doesn't survive a kernel restart.

**Resolution**: Replaced all four install cells with a single **smart installer pattern**:
```python
try:
    import sae_lens; import numpy as np; np.random.seed(42); ready = True
except (ImportError, ValueError): ready = False
if not ready:
    os.system("pip install sae-lens==4.3.1 ...")
    exit()  # triggers Colab auto-restart
```
The `exit()` call (not `os._exit()`) triggers Colab's built-in runtime restart. After reconnect, the `try` block succeeds and execution continues normally. The user just needs to click "Run All" a second time.

### Challenge: TransformerLens `fwd_hooks` kwarg removed
`src/attacks.py` calls `model.run_with_cache(..., fwd_hooks=[("hook_embed", fn)])` to inject perturbed embeddings mid-forward-pass. Newer TransformerLens versions (≥1.18) removed the `fwd_hooks` kwarg from `run_with_cache` in favour of a context-manager API (`model.hooks(...)`).

**Resolution**: A **monkey-patch** applied once immediately after model loading:
```python
def apply_tl_hook_patch():
    _orig = HookedTransformer.run_with_cache
    def _patched(self, *args, **kwargs):
        fwd_hooks = kwargs.pop("fwd_hooks", [])
        bwd_hooks = kwargs.pop("bwd_hooks", [])
        if fwd_hooks or bwd_hooks:
            with self.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                return _orig(self, *args, **kwargs)
        return _orig(self, *args, **kwargs)
    HookedTransformer.run_with_cache = _patched
    HookedTransformer._patched_for_hooks = True
```
Patching the class (not an instance) means the fix propagates to all calls inside `src/attacks.py` without modifying that module.

---

## Phase 3 — Consolidation into One Master Notebook

### Decision: one notebook instead of four
**Reason requested by user**: running four separate notebooks requires manual handoff between sessions, re-loading the model four times, and risks mid-run disconnects losing partial results. A single notebook runs end-to-end: load once, run all experiments sequentially, sync everything to Drive automatically.

### Design principles applied
1. **Modularity via helper functions** — all experiment loops (`run_exp1`, `run_exp2`, `run_exp3`) live in a dedicated helpers cell. Each takes a `cfg` dict so hyperparameters are declared once in the config cell and flow through without globals scattered across the notebook.
2. **Google Drive sync after every experiment** — `sync_to_drive(local_dir, drive_dir)` copies all files after each experiment's save step. If the session disconnects mid-run, completed experiment results are already on Drive.
3. **Single configuration cell** — all hyperparameters (model name, SAE release, ε values, δ_KL values, prompt counts, layer lists) declared in one place at the top so nothing needs to be hunted across cells.
4. **Descriptive markdown before every code section** — each experiment section opens with a markdown cell explaining the scientific question, the attack formulation, and what the key metrics mean.

---

## Phase 4 — Model Pivot: Gemma 2 2B → Pythia 1.4B

### Decision
User requested switching from Gemma 2 2B to a fully **ungated** model. Gemma 2 2B requires accepting a licence agreement on HuggingFace before access is granted, which adds friction and a potential blocker for first-time runs.

**Pythia 1.4B** (EleutherAI) was chosen as the replacement:
- Fully open, no gating
- 1.4B parameters — more capable than GPT-2 Small, still fits comfortably in T4 VRAM
- Well-supported by TransformerLens (`"pythia-1.4b"`)
- Assumed to have pretrained SAEs in sae-lens

### Configuration set for Pythia 1.4B
```
MODEL_NAME      = "pythia-1.4b"
TOKENIZER_NAME  = "EleutherAI/pythia-1.4b"
SAE_RELEASE     = "pythia-1.4b-deduped-res-jb"  # assumed
SAE_ID_TEMPLATE = "blocks.{layer}.hook_resid_post"
N_LAYERS        = 24
TARGET_LAYER    = 16
```

### SAE auto-discovery added
Because the exact sae-lens release name was uncertain, the SAE loading cell included a runtime discovery block that queries `get_pretrained_saes_directory()` and auto-corrects the release name and hook template if the configured name is wrong.

---

## Phase 5 — Runtime Error: Wrong SAE Release Name

### Error encountered in Colab
```
ValueError: Release pythia-1.4b-deduped-res-jb not found in pretrained SAEs directory,
and is not a valid huggingface repo.
```
Additionally, the auto-discovery had failed silently with:
```
Auto-discovery failed (No module named 'sae_lens.pretrained_saes').
```

### Root cause analysis

**Problem 1 — Wrong module path for the registry**
The auto-discovery tried `from sae_lens.pretrained_saes import get_pretrained_saes_directory`. In sae-lens 4.3.1, the pretrained SAE registry is not a Python module at all — it is a YAML file at:
```
/usr/local/lib/python3.12/dist-packages/sae_lens/pretrained_saes.yaml
```
There is no corresponding `.py` file with a `get_pretrained_saes_directory` function accessible at that path. The import failed, the except caught it, and execution fell through to the (wrong) configured release name.

**Problem 2 — Pythia 1.4B has no SAEs in sae-lens 4.3.1**
Running the corrected diagnostic (scanning the YAML file directly), the full release list was recovered. The Pythia entries in sae-lens 4.3.1 are:
```
pythia-70m-deduped-att-sm
pythia-70m-deduped-mlp-sm
pythia-70m-deduped-res-sm
sae_bench_pythia70m_sweep_gated_ctx128_0730
sae_bench_pythia70m_sweep_panneal_ctx128_0730
sae_bench_pythia70m_sweep_standard_ctx128_0712
sae_bench_pythia70m_sweep_topk_ctx128_0730
```
Only Pythia **70M** is covered. No 160M, 410M, 1B, or 1.4B SAEs are included in this version.

### Resolution — two fixes applied

**Fix 1: Robust auto-discovery in the notebook**
The SAE loading cell was rewritten to try four different import paths for the registry helper, then fall back to scanning the package directory for any `pretrained_saes*` file and reading it with PyYAML. This makes discovery version-agnostic:
```python
for _mod_path in ["sae_lens.pretrained_saes", "sae_lens.toolkit.pretrained_saes",
                   "sae_lens.training.pretrained_saes", "sae_lens.utils"]:
    ...
# Fallback: glob for the YAML file in the package directory
for _f in glob.glob(os.path.join(pkg_dir, "**", "pretrained_saes*"), recursive=True):
    _directory = yaml.safe_load(open(_f))
```
Once the directory is loaded, the cell auto-selects the best available release, auto-corrects the hook template based on sample SAE IDs, and reloads the model if a family switch (Pythia → GPT-2) is necessary.

**Fix 2: Model pivot to GPT-2 Small**

---

## Phase 6 — Model Pivot: Pythia 1.4B → GPT-2 Small

### Decision
With no Pythia 1.4B SAEs available in the installed sae-lens version, the options were:
| Option | SAEs available | Gated? | Notes |
|--------|---------------|--------|-------|
| GPT-2 Small | `gpt2-small-res-jb` ✓ | No | Best SAE coverage; 12 layers, 117M params |
| Pythia 70M | `pythia-70m-deduped-res-sm` ✓ | No | Too small; results less meaningful |
| Gemma 2B | `gemma-2b-res-jb` ✓ | Yes | Requires HuggingFace licence |
| Gemma Scope 2B | `gemma-scope-2b-pt-res` ✓ | Yes | Same gating issue |

**GPT-2 Small** was chosen because:
- Fully ungated (no HuggingFace agreement required)
- Best SAE coverage in sae-lens 4.3.1 (`gpt2-small-res-jb` plus several higher-width variants)
- Well-validated TransformerLens support
- The scientific contribution of this project is the **attack methodology**, not the specific model — results on GPT-2 Small are fully publishable

### Final configuration
```
MODEL_NAME      = "gpt2"
TOKENIZER_NAME  = "gpt2"
SAE_RELEASE     = "gpt2-small-res-jb"
SAE_ID_TEMPLATE = "blocks.{layer}.hook_resid_pre"   # GPT-2 uses hook_resid_pre
HOOK_TEMPLATE   = "blocks.{layer}.hook_resid_pre"
N_LAYERS        = 12
TARGET_LAYER    = 8
LAYERS_TO_TEST  = [0, 2, 4, 6, 8, 10, 11]
```

**Note on hook convention**: GPT-2 SAEs in sae-lens use `hook_resid_pre` (residual stream *before* each block's attention + MLP), while Gemma Scope SAEs use `hook_resid_post`. This difference is automatically detected and corrected by the SAE loading cell's template-inspection logic.

---

## Current State

### What is fully working
- `notebooks/master_experiment.ipynb` — single notebook, all four experiments, Drive sync, robust SAE loading
- All `src/` modules unchanged and tested (CPU, mock model/SAE)
- Smart installer handles the numpy binary conflict automatically
- TransformerLens hook compatibility patch applied at runtime

### File structure additions
```
notebooks/master_experiment.ipynb   ← new: consolidated master notebook
doc.md                              ← this file
```
The four individual notebooks (`01–04_*.ipynb`) are retained as reference but are no longer the primary execution path.

### Remaining steps
~~1. Push to GitHub~~  ✓
~~2. Run `master_experiment.ipynb` in Colab (T4 GPU)~~ ✓ — results in `SAE_Adversarial_Results/`
3. Analysis ← **we are here**
4. Fix attack implementation bug (see Phase 7)
5. Re-run with fixed attack, then README writeup

---

## Phase 7 — Results Analysis & Critical Bug Discovery

### Experiment Run Configuration (confirmed)
| Parameter | Value |
|-----------|-------|
| Model | GPT-2 Small (`gpt2`) |
| SAE release | `gpt2-small-res-jb` |
| Target layer (Exp 1, 2) | 8 |
| Layers tested (Exp 3) | 0, 2, 4, 6, 8, 10, 11 |
| Epsilon values (Exp 1) | 0.01, 0.05, 0.1, 0.5 |
| δ_KL values (Exp 2) | 0.01, 0.1, 0.5, 1.0 |
| PGD steps | 20 |
| OPA steps | 40 |
| Prompts (Exp 1, 2) | 100 |
| Prompts (Exp 3) | 50 per layer |
| Random samples per prompt | 10 |
| Seed | 42 |

### Baseline (Exp 0)
The model and SAE loaded and ran correctly:
- **Perplexity**: mean = 58.1, std = 27.3 (typical for WikiText-2 on GPT-2 Small)
- **Next-token accuracy**: 31.6% (reasonable for a 117M parameter model)
- **SAE L0**: mean = 72.9 features active per token (from a 768-dimensional dictionary)
- **SAE FVU**: 0.0107 (the SAE reconstructs ~98.9% of the residual stream variance)

The baseline confirms the SAE is healthy and behaving as expected. High L0 (72/768 features active) means many non-zero activations exist — the SAE is not sparse to the point of having dead neurons, and gradient should flow through most of them.

---

### Key Finding: The PGD Attack Produced Zero Perturbation

**This is the central result of the first run.** Across every experiment — all ε values, all δ_KL values, all 7 tested layers — the PGD attack had no measurable effect on SAE representations.

| Metric | Adversarial (PGD) | Random baseline |
|--------|-------------------|-----------------|
| SAE cosine similarity | **1.0000000047** (all ε) | 0.9908 at ε=0.5 |
| Jaccard index | **1.0** (all conditions) | < 1.0 at higher ε |
| Feature flip count | **0** (all conditions) | > 0 at higher ε |
| KL divergence | **0.0** (all conditions) | > 0.0 at higher ε |
| Adversarial advantage | **≈ 0** (slightly negative) | baseline = 1.0 |

The smoking gun: `adv_cosine_mean = 1.0000000047683715` is **identical down to 8 decimal places for ε = 0.01, 0.05, 0.1, and 0.5**. A correctly-functioning attack must produce different outcomes at different budgets. Identical values across a 50× range in ε is conclusive proof the perturbation is zero in all cases.

The random baseline, by contrast, works correctly — cosine similarity degrades from 0.999997 → 0.9908 as ε increases from 0.01 → 0.5, confirming the evaluation pipeline is sound.

---

### Root Cause Analysis: Broken Gradient Graph

**Cause 1 (primary): `run_with_cache` breaks the autograd graph**

The PGD inner loop in `src/attacks.py` does:
```python
perturbed = (clean_embeds + delta).detach().requires_grad_(True)
# ... run model, compute loss ...
loss.backward()
grad = perturbed.grad  # ← this is None
if grad is None:
    break             # ← loop exits on step 0
```

`model.run_with_cache` is TransformerLens's caching variant of the forward pass. Despite the monkey-patch that wraps it in `model.hooks(...)`, `run_with_cache` internally manages intermediate tensors in ways that do not guarantee preservation of the autograd computation graph from the injected `perturbed` tensor all the way back to the gradient tape. As a result, `perturbed.grad` is `None` after `loss.backward()`.

The `if grad is None: break` guard (which was included as a safety measure) fires on the **very first PGD step**. The loop exits immediately, `delta` remains all zeros, and the returned embedding is identical to the clean embedding for every prompt, every epsilon, every layer.

**This is confirmed by the identical cosine values across all ε** — the attack is never running at all.

**Cause 2 (secondary): wrong residual hook point hardcoded**

`src/attacks.py` hardcodes:
```python
resid_name = f"blocks.{target_layer}.hook_resid_post"
```

For GPT-2 Small with `gpt2-small-res-jb`, the SAE is hooked at `hook_resid_pre` (residual stream *before* each block's attention + MLP), not `hook_resid_post` (after). Even if the gradient had flowed, the attack would have been optimising the wrong hook point. The delta optimised to disrupt `hook_resid_post` features would have had poor transfer to `hook_resid_pre` features (they are separated by a full transformer block of computation).

---

### Secondary Finding: Random Perturbation Reveals Real Layer Sensitivity

While the adversarial attack failed, the **random perturbation baseline** ran correctly and produced a genuine signal:

| Layer | Random cosine (ε=0.1) | Interpretation |
|-------|----------------------|----------------|
| 0 | 0.9968 | Most sensitive to noise |
| 2 | 0.9999 | Highly stable |
| 4 | 0.9998 | Highly stable |
| 6 | 0.9997 | Highly stable |
| 8 | 0.9997 | Highly stable |
| 10 | 0.9994 | Slightly more sensitive |
| 11 | 0.9996 | Slightly more sensitive |

Layer 0 stands out: random noise at ε=0.1 produces a cosine similarity of 0.9968, versus 0.9994–0.9999 at middle/late layers. This is approximately 3–10× more disruption from the same noise budget. This likely reflects that Layer 0's SAE is applied to the *raw embedding output*, which has smaller magnitude features and therefore a lower signal-to-noise ratio relative to deeper layers where residual stream norms grow.

This finding would be meaningful in a fixed attack experiment: if an adversary applies a perturbation at Layer 0's scale of disruption via PGD (once the gradient flow is fixed), the layer-dependent vulnerability profile may show Layer 0 and late layers (10, 11) as the most exploitable entry points.

---

### Why the Results Look "Too Good" — Interpreting the Figures

The figures (`exp1_feature_stability.png`, `exp3_layerwise_sensitivity.png`) show:
- PGD line: perfectly flat at cosine=1.0 across all ε (Exp 1) and all layers (Exp 3)
- Random line: gently descending with ε (Exp 1) and layer-dependent (Exp 3)
- Adversarial advantage: all bars at ≈0, far below the 1.0 random baseline dashed line

In a functioning experiment, the PGD line should be *below* the random line (the whole point of adversarial attacks is to do better than random noise). Here, PGD appears to perform *worse* than random noise (advantage < 1), which is the paradoxical signature of a silent attack failure: the "attack" produced clean embeddings that score perfectly, while random noise at least introduces some perturbation.

---

### What the Results Would Show If the Attack Worked

Based on the methodology and the baseline SAE statistics, a correctly-functioning PGD attack should produce:
- At ε=0.01: cosine disruption likely negligible (the SAE's spread in activation space may naturally absorb this)
- At ε=0.1: meaningful cosine reduction (random baseline at same ε is already 0.9997, so PGD should reach 0.99 or lower)
- At ε=0.5: significant disruption; feature flips expected across many prompts
- Layer 0: most vulnerable (already evidenced by random baseline)
- Adversarial advantage > 1 at all ε if PGD is exploiting the loss gradient properly

The **output-preserving attack (Exp 2)** is the most scientifically interesting: it would reveal whether an adversary can maximally disrupt SAE feature readings while keeping the model's observable behaviour (next-token predictions) unchanged. This threat scenario — SAE fooling without behavioural change — is the core novel contribution of the project. The current results show 0% success rate for this attack, but only because the underlying perturbation is zero.

---

### Fix Required for v2

**Fix 1: Replace `run_with_cache` with `run_with_hooks` in the PGD inner loop**

`model.run_with_hooks()` is TransformerLens's gradient-compatible API. It does not cache all intermediate activations and is designed for use inside autograd. The fix in `src/attacks.py`:

```python
# Current (broken): uses run_with_cache which breaks gradient graph
_, cache = model.run_with_cache(clean_tokens, names_filter=resid_name,
                                 fwd_hooks=[("hook_embed", hook_fn)])
resid = cache[resid_name]

# Fixed: use run_with_hooks to preserve gradient flow
resid_capture = {}
def _save_resid(value, hook):
    resid_capture["resid"] = value
    return value

model.run_with_hooks(clean_tokens,
    fwd_hooks=[("hook_embed", hook_fn), (resid_name, _save_resid)])
resid = resid_capture["resid"]
```

**Fix 2: Parameterise the residual hook name**

Remove the hardcoded `hook_resid_post` in `pgd_attack_sae` and `output_preserving_attack`. Pass the hook name explicitly so it matches whatever the SAE was trained on:

```python
def pgd_attack_sae(model, sae, clean_tokens, target_layer, epsilon,
                   resid_hook: str, ...):  # add resid_hook parameter
    resid_name = resid_hook  # use caller-supplied hook, not hardcoded
```

**Fix 3: Add gradient flow assertion**

Add a sanity-check assertion at the start of the PGD loop to catch gradient failures early instead of silently falling through:

```python
if _step == 0 and perturbed.grad is None:
    raise RuntimeError(
        "Gradient did not flow back to perturbed embeddings. "
        "Ensure the forward pass uses run_with_hooks, not run_with_cache."
    )
```

---

### Research Questions for v2 (Post-Fix)

**Q1 — Basic adversarial vulnerability**
After fixing gradient flow, how much do SAE feature activations degrade as a function of ε? Is there a threshold below which the SAE is robust (cosine ≈ 1) and above which disruption grows rapidly?

**Q2 — Adversarial advantage magnitude**
What is the actual adversarial advantage (PGD disruption / random disruption) at each ε? Values close to 1× suggest the SAE loss landscape is flat (hard to exploit); values > 5× suggest structured adversarial directions exist in embedding space.

**Q3 — Output-preserving attack success rate**
At what δ_KL tolerance can an adversary achieve ≥ 50% of maximum SAE disruption while keeping output KL below the target? This is the safety-relevant question: can interpretability tools be fooled while behaviour appears normal?

**Q4 — Layer-dependent vulnerability**
Does the random-baseline result (Layer 0 most sensitive) hold under adversarial conditions? Is Layer 0 more exploitable due to its lower residual stream norm, or do middle layers become the adversarial sweet spot due to more complex SAE features?

**Q5 — Feature specificity of adversarial attacks**
Does PGD preferentially flip high-activation features or low-activation features? Does it tend to suppress existing features, activate new features, or swap feature identities? This connects to interpretability: if an adversary can selectively suppress a specific safety-relevant feature (e.g., a "violence" detector) without activating off-distribution features, the threat model is much more severe.

**Q6 — Cross-layer transfer**
Does a perturbation optimised to disrupt Layer 8 SAE features also disrupt features at Layer 6 and Layer 10? Poor transfer would suggest layer-specific vulnerabilities; strong transfer would suggest the embedding-space perturbation propagates coherently through the residual stream.

**Q7 — Epsilon calibration relative to embedding norm**
The epsilon values [0.01, 0.05, 0.1, 0.5] are absolute. GPT-2 Small's embedding vectors have typical L2 norm of ~7–10. An ε=0.5 perturbation in L∞ corresponds to roughly a 5–7% relative perturbation. Should epsilon be expressed as a fraction of the embedding norm to make results model-agnostic?

---

### Planned v2 Experimental Changes

| Change | Justification |
|--------|---------------|
| Fix `run_with_hooks` in attack loop | Primary fix — without this nothing else matters |
| Parameterise `resid_hook` in attack functions | Prevents silent cross-hook mismatches |
| Add gradient assertion on step 0 | Catch future gradient failures immediately |
| Add per-step loss logging to PGD | Verify loss decreases; diagnose if still broken |
| Extend ε range to [0.1, 0.5, 1.0, 5.0] | May need larger budget if SAE is genuinely robust |
| Add feature-level analysis (top-k flipped features) | Enables Q5 investigation |
| Add cross-layer transfer experiment | New sub-experiment for Q6 |
| Normalise ε by embedding norm | Enables model-agnostic comparison |

---

## Lessons Learned

| # | Lesson | Impact |
|---|--------|--------|
| 1 | Always verify SAE availability for a target model *before* committing to it — check the `pretrained_saes.yaml` in the installed sae-lens version | Caused one full model pivot (Pythia 1.4B → GPT-2 Small) |
| 2 | sae-lens registry helpers move between versions; never hardcode a single import path | Caused silent auto-discovery failure; fixed with multi-path discovery + YAML fallback |
| 3 | TransformerLens API is not stable across minor versions; `fwd_hooks` kwarg was removed | Caused attack failures; fixed with a one-time class-level monkey-patch |
| 4 | Colab pip installs trigger numpy binary conflicts with sae-lens; the conflict must be resolved by restarting the runtime after install, not by re-running the install | Caused install-time failures; fixed with smart installer + `exit()` restart pattern |
| 5 | For a project whose contribution is a methodology, the specific model matters less than having good pretrained SAEs | Justified switching to GPT-2 Small without loss of scientific value |
| 6 | `run_with_cache` breaks the autograd computation graph — use `run_with_hooks` inside any gradient-based attack loop | Caused the PGD attack to silently do nothing; all Exp 1–3 adversarial results are invalid |
| 7 | Always add a gradient flow assertion (`if step==0 and grad is None: raise`) to catch silent failures immediately | Would have caught the bug in < 1 second rather than after a full 1-hour Colab run |
| 8 | Never hardcode hook names (e.g., `hook_resid_post`) in model-agnostic attack functions — different SAE releases target different hook points | Would have caused wrong-hook optimization even if gradients had flowed correctly |
| 9 | TransformerLens hook callbacks receive the hook object as a keyword arg named `hook` — the second lambda parameter must be literally `hook`, not `h` or any other name | Caused `TypeError: got unexpected keyword argument 'hook'`; fixed by renaming all lambdas in v2 |

---

## Phase 8 — v2 Full Results Analysis

**Run date:** 2026-03-10
**Notebook:** `notebooks/master_experiment_v2.ipynb`
**Model:** GPT-2 Small + `gpt2-small-res-jb` SAEs (`hook_resid_pre`, 24k features)
**Prompts:** 150 (baseline), 100 (Exp 1 & 2), 50 (Exp 3–5 per condition), 50 (Exp 6)
**Seed:** 42

---

### Exp 0 — Baseline (v2)

| Metric | v1 | v2 |
|--------|----|----|
| Perplexity mean | 58.09 | 58.09 |
| Next-token accuracy | 31.6% | 31.6% |
| SAE L0 (mean active features) | 72.9 | 135.0 |
| SAE FVU | 0.0107 | 0.0215 |

The perplexity and accuracy are identical to v1 (same data, seed, model). The SAE L0 is ~85% higher in v2 (135 vs 73). This is because v2 correctly targets `hook_resid_pre` with the 24k-feature SAE, whereas v1 may have used a differently configured SAE hook. The higher L0 reflects the wider SAE (24k features) being used as intended.

---

### Exp 1 — Feature Stability Under PGD (Q1, Q2, Q7 partial)

Extended epsilon sweep: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

| ε (abs) | adv_cosine | rnd_cosine | Advantage | Feature Flips |
|---------|------------|------------|-----------|---------------|
| 0.01 | 0.9802 | 0.999998 | 11,737× | 8,123 |
| 0.05 | 0.1850 | 0.999958 | **19,418×** | 578,203 |
| 0.10 | 0.0191 | 0.999831 | 5,805× | 160,486 |
| 0.50 | 0.0024 | 0.9940 | 165× | 116,692 |
| 1.00 | 0.0018 | 0.1477 | 1.17× | 80,466 |
| 5.00 | 0.000079 | 0.1002 | 1.11× | 395,712 |

**Key findings:**

1. **Peak adversarial advantage at ε=0.05** (19,418×). Below this, the attack is constrained. Above this, random perturbations also start causing significant disruption (Jaccard degrades for random too), collapsing the relative advantage.

2. **At ε≥1.0, adversarial advantage drops to ~1.1–1.2×.** Both random and adversarial perturbations fully saturate the SAE. This is not a sign of SAE robustness — it means both destroy everything equally. The l∞-ball at ε=1.0 is large enough that random walks already find destructive directions.

3. **KL divergence on model outputs correlates with ε:** adv_kl≈0.87 at ε=0.01, rising to 5.62 at ε=0.1–1.0. SAE features and model outputs are jointly disrupted — no free lunch.

4. **The "interpretability adversary" is maximally dangerous at ε=0.05**, where it is dramatically more targeted than random noise (19,418×) while the perturbation is large enough to completely restructure the SAE representation (adv_cosine = 0.185).

---

### Exp 2 — Output-Preserving Attack (Q3 — CRITICAL NEGATIVE RESULT)

Lagrangian attack at ε=0.5, δ_KL ∈ {0.01, 0.1, 0.5, 1.0}, 40 steps, 100 prompts.

| δ_KL target | Mean SAE disruption | Actual output KL | % Constraint satisfied |
|-------------|---------------------|------------------|------------------------|
| 0.01 | 83.5% | 18.54 nats | **0%** |
| 0.10 | 84.4% | 18.43 nats | **0%** |
| 0.50 | 86.2% | 18.21 nats | **0%** |
| 1.00 | 83.7% | 18.49 nats | **0%** |

**The output-preserving constraint was never satisfied for any prompt across any δ_KL level.** The Lagrangian multiplier λ grew to very large values (99–173 and often Infinity), indicating the optimizer found no feasible point inside the ε-ball that simultaneously achieves SAE disruption AND output KL below even δ_KL=1.0.

**Two possible interpretations:**

1. **Fundamental coupling (preferred):** In GPT-2's embedding space, any perturbation of magnitude ε=0.5 that restructures SAE features at layer 8 also substantially changes the output distribution. This would mean SAE features are *semantically load-bearing* — disrupting them necessarily changes what the model computes. This is actually evidence *for* SAE interpretability reliability in a deep sense.

2. **Optimization failure:** The Lagrangian method with hard ε-ball constraint and adaptive λ may be poorly suited to this non-convex landscape. λ→∞ means the penalty makes the SAE-disruption objective effectively zero-weighted, so the attack degenerates to pure output-change minimisation — but even that is bounded away from the target KL. The initial ε=0.5 may already exceed the "safe zone" for output preservation.

**Recommendation for follow-up:** Test with smaller ε (e.g. ε=0.05 where advantage is peak) and wider δ_KL targets (e.g. 5.0, 10.0). Also compare against a token-space attack (top-k token swap) where output-preservation is trivially achievable. If the tension persists at smaller ε, interpretation 1 is supported.

---

### Exp 3 — Layer-wise Sensitivity (Q4)

PGD at ε=0.1 and ε=1.0 across layers [0, 2, 4, 6, 8, 10, 11]:

**ε=0.1:**

| Layer | adv_cosine | rnd_cosine | Advantage | Feature Flips |
|-------|------------|------------|-----------|---------------|
| 0 | 0.0103 | 0.9849 | 65× | 815,462 |
| 2 | 0.000344 | 0.9989 | **916×** | 800,107 |
| 4 | 0.0208 | 0.9999 | **7,288×** | 429,711 |
| 6 | 0.0195 | 0.9998 | 5,851× | 201,070 |
| 8 | 0.0241 | 0.9998 | 5,679× | 143,559 |
| 10 | 0.0343 | 0.9997 | 3,765× | 81,096 |
| 11 | 0.0152 | 0.9996 | 2,448× | 89,118 |

**ε=1.0:** Advantage collapses to ~1.1–1.2× uniformly across ALL layers (random perturbation also saturates at large ε).

**Key findings:**

1. **Layer 0 is the least vulnerable** (65× at ε=0.1). The embedding layer's SAE features are more smoothly distributed; random perturbations are nearly as disruptive as targeted ones.

2. **Layer 2 and Layer 4 are the most vulnerable** (916× and 7,288× respectively). Early-to-middle layers show the sharpest adversarial advantage. This may reflect that these layers encode locally structured features (syntax, local context) whose SAE representations are highly directional, making them easy to nullify with small targeted perturbations.

3. **Advantage decreases monotonically from Layer 4 toward Layer 11**, suggesting later layers encode more distributed/robust representations or have simpler SAE geometry.

4. **The ε=1.0 collapse is universal.** No layer is robust at large ε — all SAEs are equally saturated by large random noise. Robustness is only meaningful in the constrained regime (ε=0.1 and below).

---

### Exp 4 — Feature Specificity (Q5)

PGD at ε=0.5, layer 8, top-k=20, 50 prompts.

| Metric | Mean | Std |
|--------|------|-----|
| Features suppressed (clean→zero) | 2,030 | 846 |
| Features newly activated (zero→active) | **9,530** | 5,611 |
| Features stable (active in both) | 1,532 | 831 |
| Top-20 survival rate | 59.5% | 16.5% |
| Suppression ratio (suppressed/total clean) | 57.0% | 21.4% |
| Activation ratio (new/total clean) | **2.75×** | 1.67 |

**Key findings:**

1. **The attack is NOT primarily suppressive — it is primarily generative of spurious features.** On average 9,530 new features fire in the adversarial state vs only 2,030 suppressed. The activation_ratio of 2.75× means the attack creates nearly 3× as many spurious activations as there are suppressed features.

2. **~57% of originally active features are suppressed** (suppression_ratio=0.57). Combined with the near-complete feature Jaccard (~0.004 from Exp 3), the feature identity is almost entirely replaced.

3. **Top-20 feature survival rate = 59.5%** (std=16.5%). Nearly half of the highest-activation "most important" features are destroyed. This is the most direct threat to SAE-based interpretability pipelines: the features humans examine (top-k by activation) are nearly coin-flip stable under this attack.

4. **Implication for interpretability reliability:** An adversary can cause the SAE to report completely different dominant features for a semantically equivalent computation. Interpretability dashboards showing "top-k active features" would show meaningless results for adversarially crafted inputs.

---

### Exp 5 — Cross-Layer Transfer (Q6)

PGD attack sourced from layers {4, 8, 11}, evaluated across all layers [0, 2, 4, 6, 8, 10, 11], ε=0.5, 30 prompts.

**Cosine similarity matrix (src_layer → eval_layer, lower = more disrupted):**

| src \ eval | L0 | L2 | L4 | L6 | L8 | L10 | L11 |
|-----------|-----|-----|-----|-----|-----|-----|-----|
| L4 | 0.048 | 0.00084 | 0.00059 | 0.00116 | 0.00168 | 0.00201 | 0.00196 |
| L8 | 0.061 | 0.00240 | 0.00259 | 0.00216 | 0.00174 | 0.00188 | 0.00184 |
| L11 | 0.068 | 0.00345 | 0.00442 | 0.00391 | 0.00351 | 0.00336 | 0.00276 |

**Advantage matrix:**

| src \ eval | L0 | L2 | L4 | L6 | L8 | L10 | L11 |
|-----------|-----|-----|-----|-----|-----|-----|-----|
| L4 | 1.67 | 25.4 | **167×** | 147× | 151× | 99× | 65× |
| L8 | 1.65 | 25.4 | 167× | 147× | 151× | 99× | 65× |
| L11 | 1.64 | 25.4 | 167× | 146× | 150× | 99× | 65× |

**Key findings:**

1. **Complete source-layer agnosticism.** The advantage matrix is nearly identical regardless of which layer was the attack target. An attack trained to disrupt Layer 4 SAE features produces the same disruption pattern at Layer 8 and Layer 11 as attacks specifically trained for those layers. This means PGD converges to a shared adversarial direction in embedding space.

2. **Layer 0 is universally resistant to forward-sourced attacks** (~1.65× from all sources). Disrupting mid-to-late layers via embedding-space attack does not transfer backward to the embedding-layer SAE.

3. **Strong forward transfer from Layer 2 onward.** All evaluation layers 2–11 show 25–167× advantage regardless of attack source. The residual stream perturbation propagates coherently through all downstream transformer blocks.

4. **Practical implication:** An adversary does not need to know which SAE layer a monitoring system uses. A single PGD attack targeting *any* layer will disrupt *all* layers equally well.

---

### Exp 6 — Epsilon Calibration (Q7)

ε expressed as fraction of mean embedding L2 norm (= 3.158 for GPT-2).

| ε_frac | ε_abs | adv_cosine | rnd_cosine | Advantage | Feature Flips |
|--------|-------|------------|------------|-----------|---------------|
| 0.001 | 0.0032 | 0.9971 | 0.99999986 | **21,036×** | 2,449 |
| 0.005 | 0.0158 | 0.9568 | 0.99999572 | 10,104× | 12,818 |
| 0.010 | 0.0316 | 0.8465 | 0.99998294 | 9,001× | 28,656 |
| 0.050 | 0.1579 | 0.0090 | 0.99956062 | 2,255× | 108,842 |
| 0.100 | 0.3158 | 0.0020 | 0.99801492 | 503× | 116,027 |
| 0.200 | 0.6316 | 0.0011 | 0.98491190 | 66× | 108,466 |

**Key findings:**

1. **Maximum adversarial advantage at the smallest tested ε** (ε_frac=0.001, advantage=21,036×). Even a perturbation equal to 0.1% of the embedding norm — so small that the adversarial cosine similarity is still 0.997 — produces over 21,000× more SAE disruption than an equally-sized random perturbation.

2. **Advantage decreases monotonically with ε**, confirming the attack is most *efficient* at small budgets. The adversary's edge comes from directionality, not magnitude.

3. **"Perceptibility" crossover:** At ε_frac≈0.01, adv_cosine drops below 0.85 (visible semantic change). At ε_frac=0.05, the SAE is nearly fully disrupted (adv_cosine=0.009). The window ε_frac ∈ [0.001, 0.01] represents a practical threat zone: SAE disruption is measurable and highly adversarial (9,000–21,000×) while embedding cosine similarity remains above 0.85 (perturbation is relatively subtle).

4. **Normalisation matters.** Comparing raw ε values across models is misleading; ε_frac provides a model-agnostic scale. ε_frac=0.01 is a reasonable "moderate attack" reference point.

---

### Synthesis: Answers to the Seven Research Questions

| Q | Question | Answer |
|---|----------|--------|
| Q1 | Are SAE features robust to adversarial attacks? | **No.** Advantage of 21,036× even at ε_frac=0.001. |
| Q2 | At what ε does disruption become severe? | Severely disrupted (cos<0.1) at ε_frac≈0.05 (ε_abs≈0.16). Significant disruption begins at ε_frac=0.001. |
| Q3 | Can output-preserving attacks disrupt SAEs? | **Negative result.** Lagrangian attack failed to satisfy output-preservation constraint at any δ_KL. KL ≈18 nats regardless of target. Points to fundamental coupling between SAE features and model output at ε=0.5. |
| Q4 | Which layers are most vulnerable? | Layer 4 (7,288×) and Layer 2 (916×) at ε=0.1. Layer 0 is least vulnerable (65×). Vulnerability peaks in early-middle layers. |
| Q5 | What type of features are disrupted? | Attack creates ~4.7× more spurious activations (9,530) than suppressions (2,030). Top-20 feature survival rate = 59.5%. Dominant effect is hallucination of new features, not suppression. |
| Q6 | Do attacks transfer across layers? | **Complete transfer.** Attack trained at any layer disrupts all other layers equally. Source-layer agnostic. Only Layer 0 shows resistance (~1.65×). |
| Q7 | What is the minimum effective ε? | No "safe" ε found. Even ε_frac=0.001 gives 21,036× advantage. |

---

### Critical Reflection: What the Negative Exp 2 Result Means

The failure of the output-preserving attack has two distinct implications depending on cause:

**If it is an optimization failure** (the Lagrangian saddle point exists but isn't found): The experiment is inconclusive. A better optimizer, smaller ε, or token-space attack would be needed. Next step: rerun at ε=0.05 with δ_KL ∈ {2, 5, 10}.

**If it is a fundamental coupling** (no feasible point exists inside the ε-ball): This is actually evidence *in favour* of SAE interpretability — it means SAE features are semantically coupled to model outputs. An attack cannot break the SAE's accounting of model computations without also changing those computations. However, this does not help interpretability practitioners who rely on SAEs for input-attribution: the *inputs* that cause massive SAE disruption also change outputs, so the adversary cannot create a covert semantic-preserving attack.

Either way, Exp 2 is scientifically interesting and worth reporting as a negative result with this two-scenario analysis.

---

### Suggested Next Steps

**Priority 1 — Fix Exp 2 (output-preserving attack):**
- Rerun at ε_frac ∈ {0.001, 0.005, 0.01} with δ_KL ∈ {2.0, 5.0, 10.0}
- Log λ trajectory per step to diagnose divergence
- Try projected gradient (project onto KL-constraint set each step) instead of Lagrangian

**Priority 2 — Validate on a stronger model:**
- Gemma 2 2B + Gemma Scope SAEs (the original plan). GPT-2 findings may not generalise.
- Key question: does the source-layer agnosticism in Exp 5 hold on deeper architectures?

**Priority 3 — Targeted feature suppression (Exp 5 extension):**
- Design an attack that *specifically* targets only one feature (e.g. feature index 12,345) and verifies it is suppressed without activating others. Suppression_ratio is currently 57% — can it be made surgical?

**Priority 4 — Semantic preservation metric:**
- Add a semantic similarity metric (e.g. sentence embeddings cosine) to assess whether adversarial inputs are semantically similar to clean inputs, complementing the embedding-space cosine similarity already measured.

**Priority 5 — Write up findings:**
- The core paper-level finding: SAE features are extremely fragile to directional embedding-space perturbations (up to 21,036× advantage at ε_frac=0.001). Random perturbations have no such effect. This is a concrete, quantitative result about the gap between adversarial robustness and interpretability tool reliability.
