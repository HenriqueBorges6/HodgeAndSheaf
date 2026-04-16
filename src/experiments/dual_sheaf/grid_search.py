"""
Sweep do DualSheafDiffusion — salva resultados progressivamente.

Grade completa:
  laplacian_mode  : dual | mixed_self | mixed_nbr | symmetric
  cosheaf_b       : False | True
  train_on_trainval: False | True

Avaliacao:
  WebKB  (Texas, Cornell)    — 10 splits pre-definidos
  Planetoid (Cora, Citeseer) — split fixo, N_SEEDS seeds

Salva em runs/dual_sheaf_sweep/<run_id>/
  results.jsonl  — uma linha JSON por experimento concluido (append)
  summary.json   — tabela final (gerada ao terminar)
  progress.json  — estado atual do sweep (para retomar se interrompido)

Uso:
  # Sweep completo (deixar de noite)
  python -m src.experiments.dual_sheaf.grid_search

  # Apenas datasets heterófilos
  python -m src.experiments.dual_sheaf.grid_search --datasets Texas Cornell

  # Retomar sweep anterior
  python -m src.experiments.dual_sheaf.grid_search --resume runs/dual_sheaf_sweep/2024-01-15_22-30

  # Teste rapido (poucos epochs, 2 seeds)
  python -m src.experiments.dual_sheaf.grid_search --datasets Texas --epochs 200 --seeds 2
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

from src.experiments.dual_sheaf.train import main as train_main

# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

LAPLACIAN_MODES   = ["dual", "mixed_self", "mixed_nbr", "symmetric"]
WEBKB_DATASETS    = {"texas", "cornell", "wisconsin"}
PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}

BASE_MODEL_CFG = {
    "d": 2,
    "hidden_channels": 64,
    "num_layers": 2,
    "learner_hidden": 64,
    "learner_gnn_layers": 1,
    "dropout": 0.5,
    "left_weights": True,
    "right_weights": True,
    "use_eps": True,
    "single_learner": False,
    "orth_trans": "householder",
    "mlp_dims": [64],
}


def build_jobs(datasets, include_cosheaf, n_seeds, epochs):
    """
    Gera lista de jobs. Cada job e um dict com tudo necessario para rodar.

    Politica por dataset:
      WebKB  (Texas, Cornell)    — grade completa: laplacian_mode x cosheaf_b x train_on_trainval
      Planetoid (Cora, Citeseer) — apenas laplacian_mode, cosheaf_b=False, train_on_trainval=True
    """
    jobs = []

    for dataset in datasets:
        is_webkb   = dataset.lower() in WEBKB_DATASETS
        is_planetoid = dataset.lower() in PLANETOID_DATASETS
        splits     = list(range(10)) if is_webkb else list(range(n_seeds))

        if is_planetoid:
            # Configuracao reduzida: so laplacian_mode, sem cosheaf, so train_on_trainval=True
            for mode in LAPLACIAN_MODES:
                for split_or_seed in splits:
                    jobs.append({
                        "dataset": dataset,
                        "train_on_trainval": True,
                        "laplacian_mode": mode,
                        "cosheaf_b": False,
                        "split_or_seed": split_or_seed,
                        "is_webkb": False,
                        "epochs": epochs,
                    })
        else:
            # Grade completa para WebKB
            cosheaf_options = [False, True] if include_cosheaf else [False]
            for trainval in [False, True]:
                for mode in LAPLACIAN_MODES:
                    for cb in cosheaf_options:
                        for split_or_seed in splits:
                            jobs.append({
                                "dataset": dataset,
                                "train_on_trainval": trainval,
                                "laplacian_mode": mode,
                                "cosheaf_b": cb,
                                "split_or_seed": split_or_seed,
                                "is_webkb": True,
                                "epochs": epochs,
                            })
    return jobs


def job_key(job):
    """Chave unica para identificar um job (usado para retomar)."""
    return (
        job["dataset"],
        job["train_on_trainval"],
        job["laplacian_mode"],
        job["cosheaf_b"],
        job["split_or_seed"],
    )


# ---------------------------------------------------------------------------
# Execucao de um job
# ---------------------------------------------------------------------------

def run_job(job: dict) -> dict:
    """Executa um job e retorna o resultado."""
    cfg = {
        "dataset": job["dataset"],
        "epochs": job["epochs"],
        "lr": 0.01,
        "weight_decay": 5e-4,
        "train_on_trainval": job["train_on_trainval"],
        "model": {
            **BASE_MODEL_CFG,
            "laplacian_mode": job["laplacian_mode"],
            "cosheaf_b": job["cosheaf_b"],
        },
    }

    if job["is_webkb"]:
        cfg["split_idx"] = job["split_or_seed"]
        cfg["seed"] = 0
    else:
        cfg["seed"] = job["split_or_seed"]

    t0 = time.time()
    try:
        r = train_main(cfg_override=cfg)
        return {
            **job,
            "test_acc": r["test_at_best_val"] * 100,
            "val_acc":  r["best_val_acc"] * 100,
            "elapsed_s": round(time.time() - t0, 1),
            "status": "ok",
        }
    except Exception as e:
        return {
            **job,
            "test_acc": None,
            "val_acc":  None,
            "elapsed_s": round(time.time() - t0, 1),
            "status": f"error: {e}",
        }


# ---------------------------------------------------------------------------
# Agregacao e tabela
# ---------------------------------------------------------------------------

def aggregate(results: list) -> dict:
    """
    Agrupa por (dataset, train_on_trainval, laplacian_mode, cosheaf_b)
    e calcula media +- std dos test_acc.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["dataset"], r["train_on_trainval"], r["laplacian_mode"], r["cosheaf_b"])
        groups[key].append(r["test_acc"])

    agg = {}
    for key, accs in groups.items():
        arr = np.array(accs)
        agg[str(key)] = {
            "dataset": key[0],
            "train_on_trainval": key[1],
            "laplacian_mode": key[2],
            "cosheaf_b": key[3],
            "n": len(accs),
            "mean": round(float(arr.mean()), 2),
            "std":  round(float(arr.std()),  2),
            "accs": accs,
        }
    return agg


def print_summary(agg: dict, datasets: list):
    """Imprime tabela de resultados agrupada por train_on_trainval."""
    for trainval in [False, True]:
        tv_str = "train+val → test" if trainval else "train → val → test"
        print(f"\n{'='*72}")
        print(f"  {tv_str}")
        print(f"{'='*72}")

        col_w = 15
        ds_cols = [d for d in datasets]
        header = f"  {'Config':<26}" + "".join(f"  {d:>{col_w}}" for d in ds_cols)
        sep = "-" * len(header)
        print(header)
        print(sep)

        rows = [v for v in agg.values() if v["train_on_trainval"] == trainval]
        rows_sorted = sorted(rows, key=lambda r: (r["laplacian_mode"], r["cosheaf_b"]))

        # Reorganizar: uma linha por config, colunas = datasets
        configs_seen = {}
        for r in rows_sorted:
            key = (r["laplacian_mode"], r["cosheaf_b"])
            if key not in configs_seen:
                configs_seen[key] = {}
            configs_seen[key][r["dataset"]] = r

        for (mode, cb), ds_results in sorted(configs_seen.items()):
            cb_str = "+cosheaf" if cb else "        "
            cfg_label = f"{mode:<14} {cb_str}"
            row_str = ""
            for d in ds_cols:
                if d in ds_results:
                    r = ds_results[d]
                    cell = f"{r['mean']:>5.1f}±{r['std']:<4.1f}"
                else:
                    cell = "---"
                row_str += f"  {cell:>{col_w}}"
            print(f"  {cfg_label}  {row_str}")

        print(sep)
        # Melhor por dataset
        print(f"  {'Melhor:':<26}", end="")
        for d in ds_cols:
            ds_rows = [v for v in rows if v["dataset"] == d]
            if ds_rows:
                best = max(ds_rows, key=lambda r: r["mean"])
                cb_str = "C" if best["cosheaf_b"] else " "
                cell = f"{best['mean']:.1f}[{best['laplacian_mode'][:4]}{cb_str}]"
            else:
                cell = "---"
            print(f"  {cell:>{col_w}}", end="")
        print()


# ---------------------------------------------------------------------------
# Progresso e retomada
# ---------------------------------------------------------------------------

def load_done_keys(results_path: str) -> set:
    """Le resultados ja concluidos de um arquivo .jsonl."""
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(tuple(job_key(r)))
            except Exception:
                pass
    return done


def append_result(results_path: str, result: dict):
    """Salva um resultado em modo append (uma linha JSON)."""
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_all_results(results_path: str) -> list:
    results = []
    if not os.path.exists(results_path):
        return results
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["Texas", "Cornell", "Cora", "Citeseer"])
    parser.add_argument("--no-cosheaf", action="store_true",
                        help="Nao incluir cosheaf_b=True")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds para datasets Planetoid")
    parser.add_argument("--resume", type=str, default=None,
                        help="Caminho de um run anterior para retomar")
    args = parser.parse_args()

    # Diretorio de saida
    if args.resume:
        out_dir = args.resume
        print(f"\nRetomando sweep em: {out_dir}")
    else:
        run_id  = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = os.path.join("runs", "dual_sheaf_sweep", run_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nNovo sweep: {out_dir}")

    results_path = os.path.join(out_dir, "results.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    # Construir jobs
    jobs = build_jobs(
        datasets=args.datasets,
        include_cosheaf=not args.no_cosheaf,
        n_seeds=args.seeds,
        epochs=args.epochs,
    )
    total = len(jobs)

    # Carregar jobs ja feitos (para retomada)
    done_keys = load_done_keys(results_path)
    pending   = [j for j in jobs if tuple(job_key(j)) not in done_keys]

    print(f"  Total de jobs: {total}  |  Ja concluidos: {total - len(pending)}  |  Pendentes: {len(pending)}")

    if not pending:
        print("  Todos os jobs ja concluidos!")
    else:
        # Estimar tempo total (baseado em tempo medio por job se houver historico)
        done_results = load_all_results(results_path)
        elapsed_list = [r["elapsed_s"] for r in done_results if r.get("status") == "ok"]
        avg_s = np.mean(elapsed_list) if elapsed_list else None

        if avg_s:
            eta_s  = avg_s * len(pending)
            eta_h  = int(eta_s // 3600)
            eta_m  = int((eta_s % 3600) // 60)
            print(f"  Tempo medio/job: {avg_s:.0f}s  |  ETA: {eta_h}h{eta_m:02d}m")

        print(f"\n  Iniciando em 3s... (Ctrl+C para interromper e retomar depois)\n")
        time.sleep(3)

        sweep_start = time.time()
        completed   = total - len(pending)

        for i, job in enumerate(pending, 1):
            mode    = job["laplacian_mode"]
            cb      = job["cosheaf_b"]
            tv      = job["train_on_trainval"]
            ds      = job["dataset"]
            s       = job["split_or_seed"]
            split_type = "split" if job["is_webkb"] else "seed"

            # Barra de progresso simples
            done_frac = (completed + i - 1) / total
            bar_len   = 30
            filled    = int(bar_len * done_frac)
            bar       = "█" * filled + "░" * (bar_len - filled)

            # ETA dinamica
            elapsed_so_far = time.time() - sweep_start
            if i > 1:
                avg_job_s = elapsed_so_far / (i - 1)
                rem_s     = avg_job_s * (len(pending) - i + 1)
                eta_str   = f"ETA ~{int(rem_s//3600)}h{int((rem_s%3600)//60):02d}m"
            else:
                eta_str = "ETA ?"

            tv_str = "TV=T" if tv else "TV=F"
            cb_str = "+C"   if cb else "   "
            print(f"\n[{bar}] {completed+i}/{total}  {eta_str}")
            print(f"  {ds:<10} {tv_str}  {mode:<12} {cb_str}  {split_type}={s}")

            result = run_job(job)
            append_result(results_path, result)

            if result["status"] == "ok":
                print(f"  → test={result['test_acc']:.2f}%  ({result['elapsed_s']:.0f}s)")
            else:
                print(f"  → ERRO: {result['status']}")

        total_elapsed = time.time() - sweep_start
        print(f"\n  Sweep concluido em {total_elapsed/3600:.1f}h")

    # Gerar sumario final
    all_results = load_all_results(results_path)
    agg = aggregate(all_results)

    with open(summary_path, "w") as f:
        json.dump(list(agg.values()), f, indent=2)
    print(f"  Sumario salvo em: {summary_path}")

    print_summary(agg, args.datasets)


if __name__ == "__main__":
    main()
