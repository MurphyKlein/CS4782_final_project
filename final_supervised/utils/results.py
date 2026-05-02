import json
from pathlib import Path


def save_results(
    results: dict,
    results_dir: Path,
    fname: str = "results_supervised.json",
) -> Path:
    """
    Persist the results dict as JSON in results_dir.

    Expected results format:
    {
        T: {
            "mse": float,
            "mae": float,
            "best_epoch": int,
            "best_val_mse": float,
        }
    }
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / fname
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved -> {path}")
    return path
