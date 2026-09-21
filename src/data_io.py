"""Load and clean metabench per-item binary correctness matrices."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "benchmark-data"
PROCESSED_DIR = DATA_DIR / "processed"

BENCHMARKS = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]

MMLU_SUBJECTS = [
    p.stem for p in sorted(RAW_DIR.glob("mmlu_*.csv"))
    if "prompts" not in p.stem
]


def load_long_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"source": str, "item": int, "correct": float})


def load_benchmark_long(benchmark: str) -> pd.DataFrame:
    if benchmark == "mmlu":
        parts = []
        offset = 0
        for subj in MMLU_SUBJECTS:
            df = load_long_csv(RAW_DIR / f"{subj}.csv")
            df["item"] = df["item"] + offset
            df["subject"] = subj
            offset += df["item"].nunique()
            parts.append(df)
        return pd.concat(parts, ignore_index=True)
    return load_long_csv(RAW_DIR / f"{benchmark}.csv")


def long_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """Pivot long-format (source, item, correct) into a dense binary matrix.

    Returns (R, model_ids, item_ids) where R has shape [n_models, n_items].
    Models or items with any NaN entries are dropped.
    """
    pivot = df.pivot(index="source", columns="item", values="correct")
    pivot = pivot.dropna(axis=0, how="any").dropna(axis=1, how="any")
    R = pivot.values.astype(np.float32)
    model_ids = list(pivot.index)
    item_ids = list(pivot.columns)
    return R, model_ids, item_ids


def load_benchmark(benchmark: str, cache: bool = True) -> tuple[np.ndarray, list[str], list[int]]:
    """Load a benchmark and return (R, model_ids, item_ids).

    Caches the processed matrix as .npz + model index for speed.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PROCESSED_DIR / f"{benchmark}.npz"
    index_path = PROCESSED_DIR / f"{benchmark}_models.txt"

    if cache and cache_path.exists() and index_path.exists():
        data = np.load(cache_path)
        R = data["R"]
        item_ids = data["item_ids"].tolist()
        model_ids = index_path.read_text().strip().split("\n")
        return R, model_ids, item_ids

    df = load_benchmark_long(benchmark)
    R, model_ids, item_ids = long_to_matrix(df)

    if cache:
        np.savez_compressed(cache_path, R=R, item_ids=np.array(item_ids))
        index_path.write_text("\n".join(model_ids))

    return R, model_ids, item_ids


def load_all_benchmarks(cache: bool = True) -> dict:
    """Load all benchmarks. Returns {name: (R, model_ids, item_ids)}."""
    results = {}
    for bench in BENCHMARKS:
        print(f"Loading {bench}...")
        results[bench] = load_benchmark(bench, cache=cache)
        R, models, items = results[bench]
        print(f"  {bench}: {R.shape[0]} models × {R.shape[1]} items")
    return results
