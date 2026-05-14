"""
Grid search para Squirrel.

Uso:
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep --workers 6
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep --n-splits 1
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep --dry-run
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep --resume results/grid_partial.json
"""

import argparse
import os

from src.experiments.bisheaf_diffusion.Node_Classification._sweep_lib import run_sweep
from src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.config import BASE_CONFIG, GRID

_DATASET     = "Squirrel"
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main(
    n_splits:          int   = 10,
    workers:           int   = None,
    max_gpu_mb:        float = 5120,
    mem_per_job_mb:    float = 512,
    ram_per_worker_mb: float = 1500,
    resume_path:       str   = None,
    dry_run:           bool  = False,
) -> dict:
    return run_sweep(
        dataset=_DATASET,
        base_config=BASE_CONFIG,
        grid=GRID,
        results_dir=_RESULTS_DIR,
        n_splits=n_splits,
        workers=workers,
        max_gpu_mb=max_gpu_mb,
        mem_per_job_mb=mem_per_job_mb,
        ram_per_worker_mb=ram_per_worker_mb,
        resume_path=resume_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{_DATASET} Grid Search")
    parser.add_argument("--n-splits",       type=int,   default=10)
    parser.add_argument("--workers",        type=int,   default=None)
    parser.add_argument("--max-gpu-mb",     type=float, default=5120)
    parser.add_argument("--mem-per-job",    type=float, default=512)
    parser.add_argument("--ram-per-worker", type=float, default=1500)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--dry-run",        action="store_true")
    args = parser.parse_args()
    main(
        n_splits=args.n_splits,
        workers=args.workers,
        max_gpu_mb=args.max_gpu_mb,
        mem_per_job_mb=args.mem_per_job,
        ram_per_worker_mb=args.ram_per_worker,
        resume_path=args.resume,
        dry_run=args.dry_run,
    )