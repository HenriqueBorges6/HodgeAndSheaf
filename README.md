# Hodge Laplacian & Sheaf Neural Networks

Implementação de três arquiteturas de Graph Neural Networks para classificação de grafos baseadas em Hodge Laplacian e teoria de Sheaves.

## Estrutura do Projeto

```
Sheaves/
├── models/
│   ├── __init__.py
│   ├── hodge_gnn.py                    # Hodge-GNN (Park et al., MICCAI 2023)
│   ├── sheaf_nn.py                     # Sheaf Neural Network (Bodnar et al., NeurIPS 2022)
│   └── edge_sheaf_laplacian.py         # Híbrido: Hodge + Sheaf
├── utils/
│   ├── __init__.py
│   ├── hodge_utils.py                  # Cálculo de Hodge Laplacian L1
│   └── data_loader.py                  # Carregamento do dataset MUTAG
├── Experiments/
│   ├── __init__.py
│   ├── train_hodge.py                  # Treinar HodgeGNN
│   ├── train_sheaf.py                  # Treinar SheafNN
│   └── train_edge_sheaf.py             # Treinar EdgeSheafLaplacian
└── README.md
```

## Modelos Implementados

### 1. HodgeGNN (Hodge Graph Neural Network)

**Referência**: Park et al., MICCAI 2023 - "Convolving Directed Graph Edges via Hodge Laplacian"

**Conceitos-chave**:
- Opera diretamente nas **arestas** do grafo (não nos nós)
- Usa **Hodge 1-Laplacian** L₁ = B₁ᵀB₁ para capturar conectividade entre arestas
- L₁(i,j) indica se arestas eᵢ e eⱼ compartilham um nó
- Convolução edge-wise: H⁽ˡ⁺¹⁾ = σ(L₁ H⁽ˡ⁾ W⁽ˡ⁾)

**Vantagens**:
- Não requer features nos nós
- Captura informação topológica de ordem superior
- Preserva direção das arestas

**Arquivo**: `models/hodge_gnn.py`

### 2. Sheaf Neural Network

**Referência**: Bodnar et al., NeurIPS 2022 - "Neural Sheaf Diffusion"

**Conceitos-chave**:
- Associa **stalks** (espaços vetoriais d-dimensionais) a cada aresta
- Aprende **restriction maps** F_{v⊆e}: F(v) → F(e) entre stalks adjacentes
- Constrói **Sheaf Laplacian** ΔF que mede "discordância" entre stalks vizinhos
- Difusão sheaf: X(t+1) = X(t) - σ(ΔF (I ⊗ W₁) X(t) W₂)

**Tipos de Sheaf**:
- `diagonal`: d parâmetros independentes
- `orthogonal`: Matrizes ortogonais via reflexões de Householder
- `general`: Matrizes d×d completas

**Vantagens**:
- Evita oversmoothing via geometria sheaf
- Lida com heterofilia
- Aprende estrutura geométrica dos dados

**Otimizações**:
- **Vectorização completa**: processa todos os pares de arestas em uma única chamada MLP
- **Eigenvalue decomposition**: mais estável que SVD para D^(-1/2)

**Arquivo**: `models/sheaf_nn.py`

### 3. Edge Sheaf Laplacian Network (Híbrido)

**Conceito**: Combina os dois paradigmas anteriores

**Inovações**:
- Usa L₁ para determinar adjacência entre arestas
- Aprende restriction maps **entre arestas** (não nós)
- Camada de difusão híbrida:
  - Componente Hodge: -α L₁ X W_hodge
  - Componente Sheaf: -β ΔF (I ⊗ W_sheaf) X
- Parâmetros α, β aprendidos automaticamente

**Vantagens**:
- Combina força de ambos os métodos
- Balance adaptativo via α, β
- Maior expressividade

**Arquivo**: `models/edge_sheaf_laplacian.py`

## Dataset

**MUTAG**: Dataset de moléculas para classificação de mutagenicidade
- 188 grafos (moléculas)
- 2 classes: mutagênico (1) vs não-mutagênico (0)
- 4 tipos de arestas (ligações químicas)
- Carregado via `kagglehub`

## Como Usar

### Instalar Dependências

```bash
pip install torch numpy scipy scikit-learn kagglehub
```

### Treinar HodgeGNN

```bash
cd Sheaves/Experiments
python train_hodge.py
```

**Hiperparâmetros principais**:
- `hidden_dims = [32, 64]`: Dimensões das camadas ocultas
- `pooling_type = 'mean'`: Tipo de pooling ('mean', 'sum', 'max')
- `learning_rate = 0.001`

### Treinar Sheaf Neural Network

```bash
python train_sheaf.py
```

**Hiperparâmetros principais**:
- `stalk_dim = 8`: Dimensão dos stalks
- `sheaf_type = 'diagonal'`: Tipo de sheaf ('diagonal', 'orthogonal', 'general')
- `hidden_dims = [32, 64]`

### Treinar Edge Sheaf Laplacian Network

```bash
python train_edge_sheaf.py
```

**Hiperparâmetros principais**:
- `stalk_dim = 8`
- `sheaf_type = 'diagonal'`
- Parâmetros α, β aprendidos automaticamente

## Métricas de Avaliação

Todos os scripts reportam:
- **Accuracy**: Acurácia geral
- **Precision**: Precisão
- **Recall**: Revocação
- **F1-Score**: Média harmônica de precision e recall
- **Confusion Matrix**: Matriz de confusão

## Detalhes Técnicos

### Hodge 1-Laplacian

Para um grafo com arestas E = {e₁, ..., eₘ}, L₁ ∈ ℝ^(m×m) é definido como:

```
L₁(i,j) = {
    2      se i = j (self-loop)
    -2     se eᵢ = (u,v) e eⱼ = (v,u) (aresta reversa)
    1      se eᵢ e eⱼ compartilham nó com mesma orientação
    -1     se eᵢ e eⱼ compartilham nó com orientação oposta
    0      caso contrário
}
```

Computado via: L₁ = B₁ᵀB₁, onde B₁ é a matriz de incidência.

### Sheaf Laplacian

Para stalks de dimensão d:

1. **Diagonal blocks**: LF_{vv} = Σ F^T_{v⊆e} F_{v⊆e}
2. **Off-diagonal blocks**: LF_{vu} = -F^T_{v⊆e} F_{u⊆e}
3. **Normalização**: ΔF = D^(-1/2) LF D^(-1/2)

### Otimizações de Performance

Comparado com implementações ingênuas:

1. **SheafBuilder vectorizado**:
   - Antes: Loop O(E²) sobre pares de arestas
   - Agora: Batch MLP único
   - **Speedup esperado**: 7-10x

2. **Eigenvalue decomposition**:
   - Mais estável que SVD
   - Regularização via 1e-6

3. **Batch processing**:
   - Processa múltiplos grafos em paralelo

## Referências

1. **Hodge-GNN**: Park, J., et al. (2023). "Convolving Directed Graph Edges via Hodge Laplacian for Brain Network Analysis." MICCAI 2023.

2. **Sheaf Neural Networks**: Bodnar, C., et al. (2022). "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs." NeurIPS 2022.

3. **Simplicial Complexes**: Lim, L.H. (2015). "Hodge Laplacians on graphs."

## Autores

Implementação otimizada para classificação de grafos usando MUTAG dataset.
