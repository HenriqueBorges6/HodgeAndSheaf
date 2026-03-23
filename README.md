# HodgeAndSheaf

Implementacao de GNNs topologicas para classificacao de grafos baseadas em **Teoria de Hodge**, **Feixes (Sheaves)** e **Fibrados Principais**. Tres modelos operam sobre estruturas algebricas distintas — Laplaciano de Hodge escalar, co-feixes planos com mapas ortogonais, e conexoes com holonomia — todos avaliados em datasets moleculares via 10-fold stratified cross-validation.

## Modelos

### 1. HodgeGNN

GNN sobre o **espaco de arestas** via Laplaciano de Hodge de ordem 1.

**Ideia central:** A matriz de incidencia $B_1$ define o Laplaciano $L_1 = B_1^\top B_1 \in \mathbb{R}^{E \times E}$, que captura adjacencia entre arestas via nos compartilhados. A convolucao opera diretamente nas arestas:

$$H^{(l+1)} = \sigma(L_{1,\text{norm}} \cdot H^{(l)} \cdot W^{(l)})$$

**Pipeline:**
```
edge_attr (E, in_dim)
    -> edge_embed: Linear + ReLU
    -> HodgeConv x N + ReLU (normalizacao simetrica opcional)
    -> Dropout
    -> global_pool (sum | mean)
    -> MLP classificador
    -> logits (B, num_classes)
```

**Referencia:** Park et al., "Convolving Directed Graph Edges via Hodge Laplacian" (MICCAI 2023)

### 2. PiOneGNN

GNN baseada em **fibrados principais** e **grupo fundamental** $\pi_1$.

**Ideia central:** Cada aresta carrega uma matriz de transporte paralelo $O_{uv} \in SO(d)$, parametrizada via algebra de Lie ($A_{\text{raw}} \to A_{\text{antissim}} \to \exp(A) \in SO(d)$). Features de nos vizinhos sao transportadas antes da agregacao. O readout extrai invariantes de gauge via **momentos espectrais de Wilson** $W_c^{(k)} = \text{tr}(H_c^k)$ sobre a base de ciclos fundamentais do grafo.

**Pipeline:**
```
x_node (N, in_node_dim)  -> node_embed  -> h (N, d)
edge_attr (E, in_e_dim)  -> EdgeEncoder -> O_edges (E, d, d) in SO(d)
                                |
                        ConnectionConv x num_layers + ReLU
                                |
                        WilsonReadout (vetorizado)
                          |-- normas:  sum ||h_v||^2         por grafo -> (B, 1)
                          |-- wilson:  sum tr(H_c^k), k=1..K por grafo -> (B, K)
                                |
                          concat -> (B, 1+K)
                                |
                          MLP classificador -> logits (B, num_classes)
```

**Funcao de perda composta:** $\mathcal{L} = \mathcal{L}_{CE} + \lambda \sum_c \|I - H_c\|_F^2$ (regularizacao topologica que penaliza holonomias nao-triviais).

### 3. CoSheafNN

GNN sobre o **espaco de arestas** via **co-feixe plano** com mapas ortogonais aprendiveis.

**Ideia central:** Cada aresta recebe um mapa de restricao $\hat{F}_e \in O(d)$ aprendido por um GNN interno sobre o grafo de linhas. O Laplaciano do co-feixe $\Delta_1 = B^\top B$ (operador de bordo em bloco) difunde informacao entre arestas respeitando a geometria local do co-feixe:

$$x'_e = D_e \cdot x_e - \sum_{e' \sim e} \text{sign}(e, e') \cdot \hat{F}_e^\top (\hat{F}_{e'} \cdot x_{e'})$$

**Pipeline:**
```
x_node (N, node_feat) --+
                         +-> node_agg = (x_node[src] + x_node[tgt]) / 2
x_edge (m, edge_feat) --+
                         cat -> Linear -> ReLU                 (m, hidden)
                              |
                         CoSheafConv x N
                           |-- Learner: GNN no line graph -> Orthogonal -> maps (m, d, d)
                           |-- Stalk embed: d Linear independentes       -> x (m, d, out_ch)
                           |-- Transport: F_i^T @ F_j @ x_j com signs   -> msg
                           |-- Aggregate: scatter_add -> D*x - agg       -> (m, d, out_ch)
                           |-- Stalk projection: reshape(m, d*h) -> Linear -> (m, h) -> ReLU
                              |
                         global_add_pool (edge_batch)          (B, h)
                              |
                         MLP classificador                     (B, num_classes)
```

**Parametrizacao ortogonal:** Mapa de Cayley $Q = (I+S)^{-1}(I-S)$ ou reflexoes de Householder, selecionavel via `orth_method`.

**Referencia:** Curry, "Sheaves, Cosheaves and Applications" (2014); Bodnar et al., "Bundle Neural Networks" (2024)

## Estrutura do Projeto

```
src/
|-- __init__.py
|-- models/
|   |-- __init__.py
|   |-- Hodge/
|   |   |-- __init__.py
|   |   |-- HodgeGNN.py              # HodgeConv + HodgeGNN
|   |   |-- teste_hodge_utils.py     # Testes das funcoes auxiliares
|   |   +-- test_mutag.py
|   |-- PiOneGNN/
|   |   |-- preprocessing.py         # Extracao de ciclos fundamentais (NetworkX)
|   |   |-- edge_encoder.py          # edge_attr -> SO(d) via Algebra de Lie
|   |   |-- connection_conv.py       # Message passing com transporte paralelo
|   |   |-- wilson_readout.py        # Readout vetorizado: momentos espectrais de Wilson
|   |   |-- collate.py               # Batching customizado com padding de ciclos
|   |   +-- pi_one_GNN.py            # Modelo completo
|   |-- CoSheafNN/
|   |   |-- __init__.py
|   |   |-- utils.py                 # build_incidence, build_line_graph_edge_index
|   |   |-- cosheaf_learner.py       # SAGELayer, GINLayer, Orthogonal, HouseholderOrthogonal
|   |   |-- cosheaf_conv.py          # CoSheafConv (Laplaciano do co-feixe)
|   |   |-- cosheaf_network.py       # CoSheafNetwork (modelo completo)
|   |   +-- sanity_check.py          # 5 verificacoes automaticas
|   +-- CoSheafGNN/                  # Implementacao alternativa (legacy, nao usar)
|-- experiments/
|   |-- hodge/
|   |   |-- config.py                # Hiperparametros
|   |   +-- train.py                 # 10-fold CV
|   |-- pi_one/
|   |   |-- config.py
|   |   |-- train.py                 # 10-fold CV multi-dataset
|   |   |-- sweep.py                 # Grid search com subprocessos isolados
|   |   |-- analyze.py               # Analise dos resultados
|   |   +-- diagnose.py              # Diagnostico de datasets
|   +-- cosheaf/
|       |-- config.py
|       |-- train.py                 # 10-fold CV multi-dataset
|       |-- sweep.py                 # Grid search (768 configs x 4 datasets)
|       |-- train_ogb.py             # Treino com splits oficiais OGB
|       |-- sweep_ogb.py             # Sweep para OGB
|       |-- edge_features.py         # Features artificiais para datasets sem edge_attr
|       |-- analyze.py               # Visualizacao e plots
|       +-- training_datasets/       # Configs e treinos por dataset (MUTAG, ENZYMES)
```

## Datasets

| Dataset | Grafos | Nos/grafo (media) | Classes | Fonte |
|---------|--------|--------------------|---------|-------|
| MUTAG | 188 | ~17 | 2 | TUDataset |
| PTC_MR | 344 | ~14 | 2 | TUDataset |
| PROTEINS | 1113 | ~39 | 2 | TUDataset |
| ENZYMES | 600 | ~33 | 6 | TUDataset |
| ogbg-molhiv | 41,127 | ~26 | 2 | OGB |

## Instalacao

```bash
pip install -e .
```

Dependencias principais: `torch`, `torch_geometric`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `matplotlib`, `pandas`.

## Como Usar

### HodgeGNN

```bash
python -m src.experiments.hodge.train
```

### PiOneGNN

```bash
python -m src.experiments.pi_one.train
python -m src.experiments.pi_one.sweep --out-dir runs/sweep          # grid search
python -m src.experiments.pi_one.sweep --out-dir runs/sweep --dry-run
```

### CoSheafNN

```bash
python -m src.experiments.cosheaf.train --dataset MUTAG
python -m src.experiments.cosheaf.sweep                              # grid search (768 runs)
python -m src.experiments.cosheaf.sweep --datasets MUTAG PTC_MR --dry-run
python -m src.experiments.cosheaf.analyze --csv runs/cosheaf_sweep/summary.csv
python -m src.models.CoSheafNN.sanity_check                          # 5 verificacoes
```

## Resultados (MUTAG, 10-fold CV, seed=1)

| Modelo | Acuracia Media |
|--------|---------------|
| Baseline majoritario | 0.66 |
| PiOneGNN (v1, 100 epochs) | 0.77 +/- 0.065 |
| HodgeGNN (sum pooling) | ~0.80-0.82 |
| CoSheafNN (d=2, Cayley corrigido) | ~0.88 +/- 0.074 |

## Matematica

Os tres modelos compartilham a construcao da **matriz de incidencia** $B_1 \in \mathbb{R}^{N \times E}$:

```
B_1[v, e] = -1   se v e fonte de e
B_1[v, e] = +1   se v e destino de e
```

A partir dela:

- **HodgeGNN:** Usa o Laplaciano escalar $L_1 = B_1^\top B_1$ para convolucionar no espaco de arestas.
- **CoSheafNN:** Substitui $B_1$ por um operador de bordo em blocos $B \in \mathbb{R}^{Nd \times Ed}$ com mapas ortogonais aprendiveis, obtendo o Laplaciano de co-feixe $\Delta_1 = B^\top B$.
- **PiOneGNN:** Aprende matrizes de transporte $O_{uv} \in SO(d)$ nas arestas e extrai invariantes topologicos (holonomia) dos ciclos fundamentais $\pi_1(G)$.

## Referencias

1. Park et al. (2023). "Convolving Directed Graph Edges via Hodge Laplacian for Brain Network Analysis." MICCAI 2023.
2. Bodnar et al. (2022). "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs." NeurIPS 2022.
3. Bodnar et al. (2024). "Bundle Neural Networks." ICML 2024.
4. Curry (2014). "Sheaves, Cosheaves and Applications." PhD Thesis.
5. Hansen & Ghrist (2019). "Toward a spectral theory of cellular sheaves."
6. Lim (2015). "Hodge Laplacians on graphs."
7. Barbero et al. (2022). "Sheaf Neural Networks with Connection Laplacians." ICML 2022.
