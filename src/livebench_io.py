"""Load LiveBench model_judgment data into per-task binary R matrices."""

from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

LIVEBENCH_BINARY_TASKS = ["LCB_generation", "coding_completion", "typos"]


def load_livebench_task(task: str, cache: bool = True,
                        min_coverage: float = 0.8) -> tuple[np.ndarray, list[str], list[str]]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PROCESSED_DIR / f"livebench_{task}.npz"
    index_path = PROCESSED_DIR / f"livebench_{task}_models.txt"
    items_path = PROCESSED_DIR / f"livebench_{task}_items.txt"

    if cache and cache_path.exists() and index_path.exists() and items_path.exists():
        data = np.load(cache_path)
        R = data["R"]
        model_ids = index_path.read_text().strip().split("\n")
        item_ids = items_path.read_text().strip().split("\n")
        return R, model_ids, item_ids

    import pandas as pd
    from datasets import load_dataset
    ds = load_dataset("livebench/model_judgment", split="leaderboard")
    df = pd.DataFrame(ds)

    subset = df[df["task"] == task].copy()
    pivot = subset.pivot_table(index="model", columns="question_id",
                               values="score", aggfunc="first")

    n_q = pivot.shape[1]
    coverage = pivot.notna().sum(axis=1) / n_q
    pivot = pivot[coverage >= min_coverage]
    pivot = pivot.fillna(0)

    R = pivot.values.astype(np.float32)
    model_ids = list(pivot.index)
    item_ids = list(pivot.columns)

    if cache:
        np.savez_compressed(cache_path, R=R)
        index_path.write_text("\n".join(model_ids))
        items_path.write_text("\n".join(item_ids))

    return R, model_ids, item_ids


def load_all_livebench(cache: bool = True) -> dict:
    results = {}
    for task in LIVEBENCH_BINARY_TASKS:
        print(f"Loading livebench/{task}...")
        results[task] = load_livebench_task(task, cache=cache)
        R, models, items = results[task]
        print(f"  {task}: {R.shape[0]} models x {R.shape[1]} items")
    return results
