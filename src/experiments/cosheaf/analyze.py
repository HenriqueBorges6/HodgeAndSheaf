"""
Análise e visualização dos resultados do sweep CoSheafNN.

Uso:
    python -m src.experiments.cosheaf.analyze --csv runs/cosheaf_sweep/summary.csv
    python -m src.experiments.cosheaf.analyze --csv runs/cosheaf_sweep/summary.csv --out-dir runs/cosheaf_sweep/plots
    python -m src.experiments.cosheaf.analyze --csv runs/cosheaf_sweep/summary.csv --dataset MUTAG
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BASELINES = {"MUTAG": 0.660, "PTC_MR": 0.598, "PROTEINS": 0.598, "ENZYMES": 0.167}

# Hiperparâmetros categóricos (armazenados como string no CSV)
CATEGORICAL_PARAMS = ["hidden_dims", "learner_hidden", "mlp_dims", "edge_feat_mode"]
# Hiperparâmetros numéricos
NUMERIC_PARAMS     = ["d", "lr", "weight_decay", "dropout"]


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load(csv_path, dataset=None):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["mean_acc"] = df["mean_acc"].astype(float)
    df["std_acc"]  = df["std_acc"].astype(float)
    df["d"]        = df["d"].astype(int)
    df["lr"]       = df["lr"].astype(float)
    df["weight_decay"] = df["weight_decay"].astype(float)
    df["dropout"]  = df["dropout"].astype(float)
    if dataset:
        df = df[df["dataset"] == dataset].copy()
    return df


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def print_top(df, n=5):
    print("\n" + "=" * 70)
    print("  TOP CONFIGURAÇÕES POR DATASET")
    print("=" * 70)
    for ds in sorted(df["dataset"].unique()):
        sub = (df[df["dataset"] == ds]
               .sort_values("mean_acc", ascending=False)
               .head(n))
        print(f"\n  {ds}  ({len(df[df['dataset']==ds])} runs concluídas)")
        print(f"  {'mean':>6}  {'std':>6}  {'d':>3}  "
              f"{'hd':>8}  {'lh':>5}  {'mlp':>5}  "
              f"{'lr':>8}  {'wd':>6}  {'drop':>5}")
        print(f"  {'-'*65}")
        for _, r in sub.iterrows():
            print(f"  {r['mean_acc']:.4f}  {r['std_acc']:.4f}  "
                  f"{int(r['d']):>3}  "
                  f"{r['hidden_dims']:>8}  {r['learner_hidden']:>5}  {r['mlp_dims']:>5}  "
                  f"{r['lr']:>8}  {r['weight_decay']:>6}  {r['dropout']:>5}")

    print("\n  Baselines (maioria de classe):")
    for ds, b in BASELINES.items():
        if ds in df["dataset"].values:
            best = df[df["dataset"] == ds]["mean_acc"].max()
            print(f"    {ds:10s}  baseline={b:.3f}  melhor={best:.4f}  "
                  f"ganho={best - b:+.4f}")


# ---------------------------------------------------------------------------
# Plot 1: melhor acc por dataset (barras com erro)
# ---------------------------------------------------------------------------

def plot_best_per_dataset(df, out_dir):
    datasets = sorted(df["dataset"].unique())
    means, stds = [], []
    for ds in datasets:
        row = df[df["dataset"] == ds].sort_values("mean_acc", ascending=False).iloc[0]
        means.append(row["mean_acc"])
        stds.append(row["std_acc"])

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(datasets))
    ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.85,
           error_kw={"elinewidth": 1.5})
    for i, ds in enumerate(datasets):
        if ds in BASELINES:
            ax.axhline(BASELINES[ds],
                       xmin=i / len(datasets) + 0.02,
                       xmax=(i + 1) / len(datasets) - 0.02,
                       color="tomato", linewidth=1.5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Acurácia 10-fold (média)")
    ax.set_title("Melhor configuração por dataset\n(linha vermelha = baseline majoritário)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    _save(fig, out_dir, "best_per_dataset.png")


# ---------------------------------------------------------------------------
# Plot 2: distribuição de acurácias (histograma por dataset)
# ---------------------------------------------------------------------------

def plot_acc_distribution(df, out_dir):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]["mean_acc"]
        ax.hist(sub, bins=15, color="#4C72B0", alpha=0.8, edgecolor="white")
        ax.axvline(sub.mean(), color="tomato", linestyle="--", linewidth=1.5,
                   label=f"média={sub.mean():.3f}")
        if ds in BASELINES:
            ax.axvline(BASELINES[ds], color="green", linestyle=":", linewidth=1.5,
                       label=f"baseline={BASELINES[ds]:.3f}")
        ax.set_title(ds)
        ax.set_xlabel("mean_acc")
        ax.set_ylabel("Nº de configs")
        ax.legend(fontsize=8)
    fig.suptitle("Distribuição de acurácias por dataset", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "acc_distribution.png")


# ---------------------------------------------------------------------------
# Plot 3: heatmap mean_acc médio para par de hiperparâmetros
# ---------------------------------------------------------------------------

def plot_heatmap(df, row_col, col_col, out_dir, title=None):
    pivot = df.groupby([row_col, col_col])["mean_acc"].mean().unstack()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.2),
                                    max(3, len(pivot.index) * 0.9)))
    vmin = df["mean_acc"].min()
    vmax = df["mean_acc"].max()
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(title or f"mean_acc médio: {row_col} × {col_col}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out_dir, f"heatmap_{row_col}_vs_{col_col}.png")


# ---------------------------------------------------------------------------
# Plot 4: efeito de cada hiperparâmetro (violin por dataset)
# ---------------------------------------------------------------------------

def plot_hyperparam_effect(df, param, out_dir):
    datasets = sorted(df["dataset"].unique())
    values   = sorted(df[param].unique(), key=lambda v: str(v))
    if len(values) < 2:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(values)))

    for ax, ds in zip(axes, datasets):
        sub  = df[df["dataset"] == ds]
        data = [sub[sub[param] == v]["mean_acc"].values for v in values]
        # Filtra grupos sem dados suficientes para violin
        valid = [(d, c) for d, c in zip(data, colors) if len(d) >= 2]
        if not valid:
            ax.set_title(ds)
            continue
        valid_data, valid_colors = zip(*valid)
        valid_labels = [str(v) for v, d in zip(values, data) if len(d) >= 2]

        parts = ax.violinplot(valid_data, positions=range(len(valid_data)),
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], valid_colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=15, ha="right")
        ax.set_title(ds)
        ax.set_xlabel(param)
        if ax == axes[0]:
            ax.set_ylabel("Acurácia 10-fold")

    fig.suptitle(f"Efeito de '{param}' por dataset", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"effect_{param}.png")


# ---------------------------------------------------------------------------
# Plot 5: top-k configs por dataset (barras horizontais)
# ---------------------------------------------------------------------------

def plot_top_configs(df, out_dir, k=10):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6 * len(datasets), max(4, k * 0.45)),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = (df[df["dataset"] == ds]
               .sort_values("mean_acc", ascending=False)
               .head(k))
        labels = [
            f"d={r.d} hd={r.hidden_dims}\nlh={r.learner_hidden} mlp={r.mlp_dims}\n"
            f"lr={r.lr} wd={r.weight_decay} do={r.dropout}"
            for _, r in sub.iterrows()
        ]
        y = np.arange(len(sub))
        ax.barh(y, sub["mean_acc"], xerr=sub["std_acc"],
                capsize=3, color="#4C72B0", alpha=0.8,
                error_kw={"elinewidth": 1.2})
        if ds in BASELINES:
            ax.axvline(BASELINES[ds], color="tomato", linestyle="--",
                       linewidth=1.2, label=f"baseline={BASELINES[ds]:.3f}")
            ax.legend(fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean_acc")
        ax.set_title(f"{ds} — Top {k}")

    fig.suptitle("Top configurações por dataset", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "top_configs.png")


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _save(fig, out_dir, fname):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str, required=True,
                        help="Caminho para summary.csv gerado pelo sweep")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Diretório para salvar os plots (default: exibe na tela)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filtrar por dataset (ex: MUTAG)")
    parser.add_argument("--top",     type=int, default=10,
                        help="Quantas configs mostrar no plot top_configs (default: 10)")
    args = parser.parse_args()

    df = load(args.csv, dataset=args.dataset)
    print(f"\nRuns carregadas: {len(df)}  |  Datasets: {sorted(df['dataset'].unique())}")

    if df.empty:
        print("Nenhum resultado 'ok' encontrado no CSV.")
        return

    print_top(df)

    print("\nGerando plots...")
    plot_best_per_dataset(df, args.out_dir)
    plot_acc_distribution(df, args.out_dir)
    plot_top_configs(df, args.out_dir, k=args.top)

    # Heatmaps entre pares de hiperparâmetros relevantes
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        if len(sub) < 4:
            continue
        ds_dir = os.path.join(args.out_dir, ds) if args.out_dir else None
        plot_heatmap(sub, "d", "hidden_dims",    ds_dir, f"[{ds}] mean_acc: d × hidden_dims")
        plot_heatmap(sub, "d", "learner_hidden", ds_dir, f"[{ds}] mean_acc: d × learner_hidden")
        plot_heatmap(sub, "lr", "weight_decay",  ds_dir, f"[{ds}] mean_acc: lr × weight_decay")
        plot_heatmap(sub, "dropout", "mlp_dims", ds_dir, f"[{ds}] mean_acc: dropout × mlp_dims")

    # Violin por hiperparâmetro
    for param in NUMERIC_PARAMS + CATEGORICAL_PARAMS:
        if df[param].nunique() > 1:
            plot_hyperparam_effect(df, param, args.out_dir)

    print("\nPronto.")


if __name__ == "__main__":
    main()
