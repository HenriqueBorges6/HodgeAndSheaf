"""
Análise e visualização dos resultados do sweep.

Uso:
    python -m src.experiments.pi_one.analyze --csv runs/sweep/summary.csv
    python -m src.experiments.pi_one.analyze --csv runs/sweep/summary.csv --out-dir runs/sweep/plots
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["mean_acc"] = df["mean_acc"].astype(float)
    df["std_acc"]  = df["std_acc"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def print_top(df, n=5):
    print("\n" + "="*65)
    print("  TOP CONFIGURAÇÕES POR DATASET")
    print("="*65)
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds].sort_values("mean_acc", ascending=False).head(n)
        print(f"\n  {ds}  ({len(df[df['dataset']==ds])} runs concluídas)")
        print(f"  {'mean':>6}  {'std':>6}  {'d':>3}  {'np':>3}  {'λ':>5}  {'lr':>8}  {'wd':>6}")
        print(f"  {'-'*52}")
        for _, r in sub.iterrows():
            print(f"  {r['mean_acc']:.4f}  {r['std_acc']:.4f}  "
                  f"{int(r['d']):>3}  {int(r['num_powers']):>3}  "
                  f"{r['lambda_hol']:>5}  {r['lr']:>8}  {r['weight_decay']:>6}")

    baselines = {"MUTAG": 0.660, "PTC_MR": 0.598, "PROTEINS": 0.598, "ENZYMES": 0.167}
    print("\n  Baselines (maioria de classe):")
    for ds, b in baselines.items():
        if ds in df["dataset"].values:
            best = df[df["dataset"] == ds]["mean_acc"].max()
            print(f"    {ds:10s}  baseline={b:.3f}  melhor={best:.4f}  "
                  f"ganho={best-b:+.4f}")


# ---------------------------------------------------------------------------
# Plot 1: melhor acc por dataset (barras com erro)
# ---------------------------------------------------------------------------

def plot_best_per_dataset(df, out_dir):
    datasets = sorted(df["dataset"].unique())
    means, stds = [], []
    for ds in datasets:
        sub = df[df["dataset"] == ds].sort_values("mean_acc", ascending=False).iloc[0]
        means.append(sub["mean_acc"])
        stds.append(sub["std_acc"])

    baselines = {"MUTAG": 0.660, "PTC_MR": 0.598, "PROTEINS": 0.598, "ENZYMES": 0.167}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(datasets))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    for i, ds in enumerate(datasets):
        if ds in baselines:
            ax.axhline(baselines[ds], xmin=(i/len(datasets)) + 0.02,
                       xmax=((i+1)/len(datasets)) - 0.02,
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
# Plot 2: heatmap mean_acc médio para cada par de hiperparâmetros
# ---------------------------------------------------------------------------

def plot_heatmap(df, row_col, col_col, out_dir, title=None):
    pivot = df.groupby([row_col, col_col])["mean_acc"].mean().unstack()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu",
                   vmin=df["mean_acc"].min(), vmax=df["mean_acc"].max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(title or f"mean_acc: {row_col} × {col_col}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fname = f"heatmap_{row_col}_vs_{col_col}.png"
    _save(fig, out_dir, fname)


# ---------------------------------------------------------------------------
# Plot 3: efeito de cada hiperparâmetro (violin / strip por dataset)
# ---------------------------------------------------------------------------

def plot_hyperparam_effect(df, param, out_dir):
    datasets = sorted(df["dataset"].unique())
    values = sorted(df[param].unique())
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(3.5 * n_ds, 4), sharey=True)
    if n_ds == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(values)))

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        data = [sub[sub[param] == v]["mean_acc"].values for v in values]
        parts = ax.violinplot(data, positions=range(len(values)),
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values])
        ax.set_title(ds)
        ax.set_xlabel(param)
        if ax == axes[0]:
            ax.set_ylabel("Acurácia 10-fold")

    fig.suptitle(f"Efeito de '{param}' por dataset", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, f"effect_{param}.png")


# ---------------------------------------------------------------------------
# Plot 4: distribuição de acurácias (histograma por dataset)
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
        ax.set_title(ds)
        ax.set_xlabel("mean_acc")
        ax.set_ylabel("Nº de configs")
        ax.legend(fontsize=8)
    fig.suptitle("Distribuição de acurácias por dataset", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "acc_distribution.png")


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
    parser.add_argument("--csv", type=str, required=True,
                        help="Caminho para summary.csv gerado pelo sweep")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Diretório para salvar os plots (default: exibe na tela)")
    args = parser.parse_args()

    df = load(args.csv)
    print(f"\nRuns carregadas: {len(df)}  |  Datasets: {sorted(df['dataset'].unique())}")

    print_top(df)

    print("\nGerando plots...")
    plot_best_per_dataset(df, args.out_dir)
    plot_acc_distribution(df, args.out_dir)
    plot_heatmap(df, "d", "num_powers", args.out_dir, "mean_acc médio: d × num_powers")
    plot_heatmap(df, "lambda_hol", "lr", args.out_dir, "mean_acc médio: λ_hol × lr")
    for param in ["d", "num_powers", "lambda_hol", "lr", "weight_decay"]:
        if df[param].nunique() > 1:
            plot_hyperparam_effect(df, param, args.out_dir)

    print("\nPronto.")


if __name__ == "__main__":
    main()
