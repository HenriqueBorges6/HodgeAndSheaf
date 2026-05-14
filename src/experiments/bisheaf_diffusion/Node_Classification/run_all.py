"""
Orquestrador sequencial de grid search para todos os datasets.

Chama o sweep de cada dataset em sequência — espera terminar antes de começar o próximo.
Resultados ficam em Dataset/results/ de cada um.

Uso:
    # Todos os datasets (ordem padrão)
    python -m src.experiments.bisheaf_diffusion.Node_Classification.run_all --workers 6

    # Subset de datasets
    python -m src.experiments.bisheaf_diffusion.Node_Classification.run_all --datasets Texas Wisconsin --workers 6

    # Smoke test (1 split por config)
    python -m src.experiments.bisheaf_diffusion.Node_Classification.run_all --n-splits 1 --workers 4

    # Ver jobs sem rodar
    python -m src.experiments.bisheaf_diffusion.Node_Classification.run_all --dry-run
"""

import argparse
import time

# Imports lazy para não carregar torch no processo principal
_SWEEP_MODULES = {
    "Texas":     "src.experiments.bisheaf_diffusion.Node_Classification.Texas.sweep",
    "Wisconsin": "src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep",
    "Film":      "src.experiments.bisheaf_diffusion.Node_Classification.Film.sweep",
    "Squirrel":  "src.experiments.bisheaf_diffusion.Node_Classification.Squirrel.sweep",
    "Chameleon": "src.experiments.bisheaf_diffusion.Node_Classification.Chameleon.sweep",
}

_DEFAULT_ORDER = ["Texas", "Wisconsin", "Film", "Squirrel", "Chameleon"]


def _load_sweep(dataset: str):
    """Importa o módulo do sweep de um dataset sob demanda."""
    import importlib
    mod = importlib.import_module(_SWEEP_MODULES[dataset])
    return mod.main


def run_all(
    datasets:          list  = None,
    n_splits:          int   = 10,
    workers:           int   = None,
    max_gpu_mb:        float = 5120,
    mem_per_job_mb:    float = 512,
    ram_per_worker_mb: float = 1500,
    dry_run:           bool  = False,
) -> dict[str, dict]:
    datasets = datasets or _DEFAULT_ORDER

    unknown = set(datasets) - set(_SWEEP_MODULES)
    if unknown:
        raise ValueError(f"Datasets desconhecidos: {unknown}. "
                         f"Opcoes: {list(_SWEEP_MODULES)}")

    print(f"\n{'#'*70}")
    print(f"  run_all — BiSheafDiffusion Node Classification")
    print(f"  Datasets : {datasets}")
    print(f"  Splits   : {n_splits}")
    print(f"  Workers  : {workers or 'auto'}")
    print(f"  Total    : {len(datasets)} sweeps x {n_splits} splits x 96 configs")
    print(f"{'#'*70}\n")

    all_results: dict[str, dict] = {}
    total_start = time.time()

    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(datasets)}] INICIANDO: {dataset}")
        print(f"{'='*70}")

        sweep_main = _load_sweep(dataset)
        t0 = time.time()

        try:
            result = sweep_main(
                n_splits=n_splits,
                workers=workers,
                max_gpu_mb=max_gpu_mb,
                mem_per_job_mb=mem_per_job_mb,
                ram_per_worker_mb=ram_per_worker_mb,
                dry_run=dry_run,
            )
            elapsed = time.time() - t0
            all_results[dataset] = result
            h, rem = divmod(int(elapsed), 3600)
            print(f"\n  [{i}/{len(datasets)}] CONCLUIDO: {dataset}  ({h}h {rem//60:02d}m)")

        except KeyboardInterrupt:
            elapsed = time.time() - t0
            print(f"\n  [{i}/{len(datasets)}] INTERROMPIDO: {dataset} "
                  f"(apos {elapsed:.0f}s)")
            print(f"  Resultados parciais salvos em Dataset/results/grid_partial.json")
            print(f"\n  Encerrando run_all. Use --resume no sweep individual para retomar.")
            break

        except Exception as e:
            print(f"\n  [{i}/{len(datasets)}] ERRO: {dataset}: {e!r}")
            print(f"  Continuando com o proximo dataset.\n")
            all_results[dataset] = {"error": str(e)}
            continue

    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)

    if not dry_run:
        print(f"\n{'#'*70}")
        print(f"  run_all concluido — {h}h {rem//60:02d}m")
        print(f"  Datasets processados: {list(all_results)}")
        print(f"{'#'*70}\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orquestrador sequencial de grid search — todos os datasets."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets a rodar (em ordem). Padrao: {_DEFAULT_ORDER}",
    )
    parser.add_argument("--n-splits",       type=int,   default=10,
                        help="Splits por config (max 10, default 10).")
    parser.add_argument("--workers",        type=int,   default=None,
                        help="Workers paralelos por sweep. Auto se omitido.")
    parser.add_argument("--max-gpu-mb",     type=float, default=5120)
    parser.add_argument("--mem-per-job",    type=float, default=512)
    parser.add_argument("--ram-per-worker", type=float, default=1500)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Mostra jobs sem executar nenhum treino.")
    args = parser.parse_args()

    run_all(
        datasets=args.datasets,
        n_splits=args.n_splits,
        workers=args.workers,
        max_gpu_mb=args.max_gpu_mb,
        mem_per_job_mb=args.mem_per_job,
        ram_per_worker_mb=args.ram_per_worker,
        dry_run=args.dry_run,
    )
