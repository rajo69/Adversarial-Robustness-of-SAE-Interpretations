"""Data pipeline module for the SAE Adversarial Robustness project.

This module handles loading the WikiText-2 test set, tokenizing it, and
producing a fixed evaluation set of 150 prompts (64-128 tokens each).
The evaluation set is saved to a JSON file for reproducibility across runs.

Typical usage
-------------
    from src.data import prepare_dataset

    eval_set = prepare_dataset(model_name="google/gemma-2-2b", save=True)

Or run as a script to build and inspect the dataset:

    python src/data.py
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_N_PROMPTS: int = 150
DEFAULT_MIN_TOKENS: int = 64
DEFAULT_MAX_TOKENS: int = 128
DEFAULT_SEED: int = 42
DATA_DIR: Path = Path(__file__).parent.parent / "data"

# Ensure the data directory exists at import time.
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_wikitext2(split: str = "test") -> Any:
    """Load WikiText-2 dataset from HuggingFace.

    Parameters
    ----------
    split : str
        Dataset split to load. One of "train", "validation", or "test".

    Returns
    -------
    datasets.Dataset
        The loaded dataset split. Each row contains a "text" column.

    Notes
    -----
    Uses the datasets library:
    ``load_dataset("wikitext", "wikitext-2-raw-v1", split=split)``
    """
    print(f"[data] Loading WikiText-2 ({split} split) …")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    print(f"[data] Loaded {len(dataset):,} rows.")
    return dataset


def tokenize_and_chunk(
    dataset: Any,
    tokenizer: Any,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
) -> List[Dict]:
    """Tokenize dataset text and chunk into fixed-length windows.

    Parameters
    ----------
    dataset : datasets.Dataset
        HuggingFace dataset with a "text" column.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer to use. Special tokens are not added during encoding.
    min_tokens : int
        Minimum chunk length in tokens. Chunks shorter than this are
        discarded after tokenization.
    max_tokens : int
        Maximum chunk length in tokens. Longer texts are truncated to
        exactly this many tokens.
    seed : int
        Random seed for reproducibility. Reserved for future use (e.g.
        if shuffling is added); currently processing is deterministic.

    Returns
    -------
    list of dict
        Each dict contains the following keys:

        - ``"text"`` : str -- original (stripped) text string.
        - ``"tokens"`` : list of int -- token IDs (length <= max_tokens).
        - ``"n_tokens"`` : int -- number of tokens in this chunk.
        - ``"source_idx"`` : int -- row index in the original dataset.

    Notes
    -----
    - Empty lines and whitespace-only rows are skipped.
    - Rows that produce fewer than ``min_tokens`` tokens are discarded.
    - Rows that produce more than ``max_tokens`` tokens are truncated to
      exactly ``max_tokens`` tokens (first ``max_tokens`` ids retained).
    - No special tokens are added (``add_special_tokens=False``).
    """
    chunks: List[Dict] = []

    for idx, example in enumerate(dataset):
        text: str = example["text"].strip()

        # Skip empty or whitespace-only lines.
        if not text:
            continue

        tokens: List[int] = tokenizer.encode(text, add_special_tokens=False)

        # Discard chunks that are too short after tokenization.
        if len(tokens) < min_tokens:
            continue

        # Truncate to the maximum allowed length.
        tokens = tokens[:max_tokens]

        chunks.append(
            {
                "text": text,
                "tokens": tokens,
                "n_tokens": len(tokens),
                "source_idx": idx,
            }
        )

    print(f"[data] tokenize_and_chunk: produced {len(chunks):,} valid chunks "
          f"(min_tokens={min_tokens}, max_tokens={max_tokens}).")
    return chunks


def create_eval_set(
    chunks: List[Dict],
    n_prompts: int = DEFAULT_N_PROMPTS,
    seed: int = DEFAULT_SEED,
) -> List[Dict]:
    """Sample a fixed evaluation set from the chunked dataset.

    Parameters
    ----------
    chunks : list of dict
        All available chunks returned by :func:`tokenize_and_chunk`.
    n_prompts : int
        Target number of prompts in the evaluation set. If fewer chunks
        are available than requested, all chunks are returned.
    seed : int
        Random seed. MUST remain 42 to guarantee reproducibility across
        different call sites and Python sessions.

    Returns
    -------
    list of dict
        Sampled chunks. Each dict is the same as the input dicts but
        enriched with an additional ``"eval_idx"`` key (0-indexed integer)
        indicating position within the evaluation set.

    Notes
    -----
    Uses ``random.sample`` with a seeded ``random.Random`` instance.
    The module-level (global) random state is never touched, so results
    are reproducible regardless of call order or other random operations
    performed by the caller.
    """
    rng = random.Random(seed)
    n_available = len(chunks)

    if n_available == 0:
        print("[data] Warning: no chunks available; returning empty eval set.")
        return []

    if n_prompts > n_available:
        print(
            f"[data] Warning: requested {n_prompts} prompts but only "
            f"{n_available} chunks are available. Using all chunks."
        )

    selected: List[Dict] = rng.sample(chunks, min(n_prompts, n_available))

    for i, item in enumerate(selected):
        item["eval_idx"] = i

    print(f"[data] create_eval_set: selected {len(selected):,} prompts.")
    return selected


def create_debug_set(eval_set: List[Dict], n: int = 10) -> List[Dict]:
    """Return the first n prompts from the eval set for fast debugging.

    Parameters
    ----------
    eval_set : list of dict
        The full evaluation set produced by :func:`create_eval_set`.
    n : int
        Number of prompts to include in the debug subset.

    Returns
    -------
    list of dict
        The first ``n`` items from ``eval_set``. If ``eval_set`` has
        fewer than ``n`` items, all items are returned.
    """
    return eval_set[:n]


def save_eval_set(
    eval_set: List[Dict],
    path: Optional[Path] = None,
    filename: str = "eval_prompts.json",
) -> Path:
    """Save the evaluation set to a JSON file.

    Parameters
    ----------
    eval_set : list of dict
        The evaluation set to persist.
    path : Path, optional
        Directory in which to save the file. Defaults to :data:`DATA_DIR`.
    filename : str
        Output filename (including ``.json`` extension).

    Returns
    -------
    Path
        Absolute path to the file that was written.

    Notes
    -----
    Saves as a JSON array. Each entry contains at minimum the keys:
    ``eval_idx``, ``source_idx``, ``n_tokens``, ``tokens``, and ``text``.
    The file is written with UTF-8 encoding and 2-space indentation for
    human readability.
    """
    directory: Path = path if path is not None else DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)

    out_path: Path = directory / filename

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(eval_set, fh, indent=2, ensure_ascii=False)

    print(f"[data] Saved {len(eval_set):,} prompts to {out_path}.")
    return out_path


def load_eval_set(
    path: Optional[Path] = None,
    filename: str = "eval_prompts.json",
) -> List[Dict]:
    """Load a previously saved evaluation set from disk.

    Parameters
    ----------
    path : Path, optional
        Directory from which to load the file. Defaults to :data:`DATA_DIR`.
    filename : str
        Input filename (including ``.json`` extension).

    Returns
    -------
    list of dict
        The deserialized evaluation set. Each dict contains at minimum the
        keys ``eval_idx``, ``source_idx``, ``n_tokens``, ``tokens``, and
        ``text``.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    directory: Path = path if path is not None else DATA_DIR
    in_path: Path = directory / filename

    if not in_path.exists():
        raise FileNotFoundError(
            f"Eval set file not found: {in_path}. "
            "Run prepare_dataset() first to generate it."
        )

    with open(in_path, "r", encoding="utf-8") as fh:
        eval_set: List[Dict] = json.load(fh)

    print(f"[data] Loaded {len(eval_set):,} prompts from {in_path}.")
    return eval_set


def tokens_to_tensor(
    token_ids: List[int],
    device: str = "cpu",
) -> torch.Tensor:
    """Convert a list of token IDs to a 2-D tensor for model input.

    Parameters
    ----------
    token_ids : list of int
        Flat list of integer token IDs representing a single sequence.
    device : str
        PyTorch device string (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``).

    Returns
    -------
    torch.Tensor
        Shape ``[1, seq_len]``, dtype ``torch.long``, on the requested
        device. The leading dimension is the batch dimension (batch size 1).
    """
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def prepare_dataset(
    model_name: str = "google/gemma-2-2b",
    n_prompts: int = DEFAULT_N_PROMPTS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    save: bool = True,
    force_rebuild: bool = False,
) -> List[Dict]:
    """Full pipeline: load, tokenize, chunk, sample, and optionally save.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier used to instantiate the tokenizer.
        Common values:

        - ``"google/gemma-2-2b"`` for Gemma 2 2B.
        - ``"gpt2"`` for GPT-2.

    n_prompts : int
        Number of evaluation prompts to sample.
    min_tokens : int
        Minimum token length for a chunk to be retained.
    max_tokens : int
        Maximum token length; longer chunks are truncated.
    seed : int
        Global random seed used throughout the pipeline.
    save : bool
        If ``True``, write the evaluation set to
        ``DATA_DIR/eval_prompts.json`` and the debug set to
        ``DATA_DIR/debug_prompts.json``.
    force_rebuild : bool
        If ``False`` (default) and ``DATA_DIR/eval_prompts.json`` already
        exists, load and return the cached file without re-processing the
        raw dataset. Set to ``True`` to always rebuild from scratch.

    Returns
    -------
    list of dict
        The evaluation set (150 dicts by default).

    Notes
    -----
    - The tokenizer is loaded with
      ``AutoTokenizer.from_pretrained(model_name)``.
    - Progress is printed at each pipeline stage.
    - When ``save=True`` a 10-prompt debug subset is also written to
      ``DATA_DIR/debug_prompts.json`` for quick iteration.
    """
    eval_file: Path = DATA_DIR / "eval_prompts.json"

    # Fast path: return the cached file if it exists and rebuild is not forced.
    if not force_rebuild and eval_file.exists():
        print(f"[data] Found existing eval set at {eval_file}. "
              "Loading from disk (pass force_rebuild=True to regenerate).")
        return load_eval_set()

    # Step 1 – Load tokenizer.
    print(f"[data] Loading tokenizer for '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[data] Tokenizer vocabulary size: {tokenizer.vocab_size:,}.")

    # Step 2 – Load raw dataset.
    dataset = load_wikitext2(split="test")

    # Step 3 – Tokenize and chunk.
    chunks = tokenize_and_chunk(
        dataset=dataset,
        tokenizer=tokenizer,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seed=seed,
    )

    if not chunks:
        raise RuntimeError(
            "No valid chunks were produced. Check min_tokens / max_tokens "
            "settings and the dataset contents."
        )

    # Step 4 – Sample evaluation set.
    eval_set = create_eval_set(chunks, n_prompts=n_prompts, seed=seed)

    # Step 5 – Optionally persist to disk.
    if save:
        save_eval_set(eval_set, filename="eval_prompts.json")

        debug_set = create_debug_set(eval_set, n=10)
        save_eval_set(debug_set, filename="debug_prompts.json")

    return eval_set


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _print_summary(eval_set: List[Dict]) -> None:
    """Print summary statistics for an evaluation set.

    Parameters
    ----------
    eval_set : list of dict
        The evaluation set to summarise.
    """
    n = len(eval_set)
    if n == 0:
        print("[data] Eval set is empty — nothing to summarise.")
        return

    lengths = [item["n_tokens"] for item in eval_set]
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = sum(lengths) / n

    # Compute a simple percentile without numpy dependency.
    sorted_lengths = sorted(lengths)
    p25_idx = int(0.25 * (n - 1))
    p50_idx = int(0.50 * (n - 1))
    p75_idx = int(0.75 * (n - 1))

    print("\n" + "=" * 60)
    print("  Evaluation set summary")
    print("=" * 60)
    print(f"  Number of prompts : {n:>6,}")
    print(f"  Min token length  : {min_len:>6,}")
    print(f"  Max token length  : {max_len:>6,}")
    print(f"  Mean token length : {mean_len:>9.2f}")
    print(f"  25th percentile   : {sorted_lengths[p25_idx]:>6,}")
    print(f"  Median (50th)     : {sorted_lengths[p50_idx]:>6,}")
    print(f"  75th percentile   : {sorted_lengths[p75_idx]:>6,}")
    print("=" * 60)

    # Histogram buckets.
    bucket_size = 8
    print("\n  Token-length distribution (bucket width = 8 tokens):\n")
    buckets: Dict[int, int] = {}
    for l in lengths:
        bucket = (l // bucket_size) * bucket_size
        buckets[bucket] = buckets.get(bucket, 0) + 1
    bar_max = max(buckets.values())
    for bucket in sorted(buckets):
        count = buckets[bucket]
        bar_width = int(40 * count / bar_max)
        bar = "#" * bar_width
        print(f"  [{bucket:3d}-{bucket + bucket_size - 1:3d}] {bar:<40} {count:>4}")
    print()


if __name__ == "__main__":
    import sys

    # Allow overriding the model via a positional CLI argument, e.g.:
    #   python src/data.py gpt2
    model_arg: str = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-2-2b"

    print(f"[data] Running prepare_dataset with model='{model_arg}' …\n")
    result = prepare_dataset(model_name=model_arg, save=True, force_rebuild=False)
    _print_summary(result)
