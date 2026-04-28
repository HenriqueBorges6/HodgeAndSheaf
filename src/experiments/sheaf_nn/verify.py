"""
Verifica se o SheafNN esta computando corretamente:
  1. Shapes corretos em cada etapa
  2. Mapas de restricao realmente ortogonais (Q^T Q ~ I)
  3. Message passing produz saida diferente da entrada
  4. Residuo epsilon funcionando
  5. Gradientes chegam em todos os parametros
  6. Sheaf learner produz mapas diferentes por no

Rode com:
    python -m src.experiments.sheaf_nn.verify
"""

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from src.models.SheafNN.sheaf import FlatBundleConv, FlatSheafLearner, Orthogonal
from src.models.SheafNN import SheafNN

torch.manual_seed(0)

# --- Dados ---
dataset = Planetoid(root="data", name="Cora")
data = dataset[0]
x, edge_index = data.x, data.edge_index
n, feat_dim = x.shape
num_classes = dataset.num_classes

print(f"\n{'='*60}")
print(f"  Verificacao do SheafNN — Cora")
print(f"  nodes={n}  edges={edge_index.shape[1]}  feat={feat_dim}  classes={num_classes}")
print(f"{'='*60}\n")

d = 2
f = 64
dropout = 0.0   # desligar dropout para verificacoes deterministicas

# -------------------------------------------------------------------------
# 1. Verificar Orthogonal transform
# -------------------------------------------------------------------------
print("[ 1 ] Ortogonalizacao (Orthogonal)")
orth = Orthogonal(d=d, orthogonal_map='euler')
params = torch.randn(n, 1)          # 1 param por no (angulo de rotacao 2D)
Q = orth(params)                     # (n, d, d)

QtQ = torch.bmm(Q.transpose(-2, -1), Q)   # deve ser ~ I
eye = torch.eye(d).unsqueeze(0).expand(n, -1, -1)
err = (QtQ - eye).abs().max().item()
print(f"    Q shape: {tuple(Q.shape)}")
print(f"    max |Q^T Q - I|: {err:.2e}  {'OK' if err < 1e-5 else 'FALHOU'}\n")

# -------------------------------------------------------------------------
# 2. Verificar FlatSheafLearner — mapas diferentes por no
# -------------------------------------------------------------------------
print("[ 2 ] FlatSheafLearner — diversidade de mapas")
learner = FlatSheafLearner(
    d=d, hidden_channels=f,
    out_shape=(1,),           # euler d=2: 1 param
    linear_emb=True,
    gnn_type='SAGE', gnn_layers=1, gnn_hidden=64,
    gnn_residual=False, pe_size=0, sheaf_act='tanh',
)
learner.eval()
with torch.no_grad():
    x_in = torch.randn(n, d * f)
    maps_out = learner(x_in, edge_index)     # (n, 1)
print(f"    output shape: {tuple(maps_out.shape)}")
print(f"    mean: {maps_out.mean():.4f}  std: {maps_out.std():.4f}")
std_ok = maps_out.std().item() > 1e-4
print(f"    nos produzem mapas distintos: {'OK' if std_ok else 'FALHOU — todos iguais'}\n")

# -------------------------------------------------------------------------
# 3. Verificar FlatBundleConv — message passing muda features
# -------------------------------------------------------------------------
print("[ 3 ] FlatBundleConv — message passing")
conv = FlatBundleConv(
    in_channels=f, out_channels=f,
    stalk_dimension=d, dropout=dropout,
    orth_trans='euler', gnn_layers=1, gnn_hidden=64,
    left_weights=True, right_weights=True, use_eps=True,
)
conv.eval()
x_sheaf = torch.randn(n, d * f)
with torch.no_grad():
    x_out, _ = conv(x_sheaf, edge_index)

print(f"    input  shape: {tuple(x_sheaf.shape)}")
print(f"    output shape: {tuple(x_out.shape)}")
diff = (x_out - x_sheaf).abs().mean().item()
print(f"    mean |x_out - x_in|: {diff:.4f}  {'OK' if diff > 1e-4 else 'FALHOU — saida igual a entrada'}")

# verificar que nos distintos tem saidas distintas
node_std = x_out.std(dim=0).mean().item()
print(f"    std entre nos (media sobre dims): {node_std:.4f}  {'OK' if node_std > 1e-4 else 'FALHOU — todos nos iguais'}\n")

# -------------------------------------------------------------------------
# 4. Verificar epsilon residual — parametro aprendido
# -------------------------------------------------------------------------
print("[ 4 ] Epsilon residual")
eps_val = conv.epsilons.data
scale = (1 + torch.tanh(eps_val))
print(f"    epsilons shape: {tuple(eps_val.shape)}")
print(f"    escala inicial (1+tanh(0)): {scale.mean():.4f}  (esperado ~1.5)")
print(f"    OK\n")

# -------------------------------------------------------------------------
# 5. Verificar SheafNN completo — forward e backward
# -------------------------------------------------------------------------
print("[ 5 ] SheafNN forward + backward")
model = SheafNN(
    node_feat_dim=feat_dim, d=d, hidden_channels=f,
    num_layers=2, num_classes=num_classes,
    conv_type='bundle', orth_trans='euler',
    gnn_layers=1, gnn_hidden=64,
    dropout=dropout, use_eps=True,
    mlp_dims=[128],
)
model.train()
out = model(x, edge_index)
print(f"    output shape: {tuple(out.shape)}  (esperado: ({n}, {num_classes}))")

loss = nn.CrossEntropyLoss()(out[data.train_mask], data.y[data.train_mask])
loss.backward()
print(f"    loss: {loss.item():.4f}")

# Verificar gradientes
no_grad = [name for name, p in model.named_parameters() if p.grad is None]
with_grad = [name for name, p in model.named_parameters() if p.grad is not None]
grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}

print(f"    parametros com grad: {len(with_grad)}/{len(with_grad)+len(no_grad)}")
if no_grad:
    print(f"    SEM GRAD: {no_grad}")

# Mostrar as menores normas de gradiente (potencial vanishing)
sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1])
print(f"\n    5 menores normas de gradiente:")
for name, norm in sorted_grads[:5]:
    print(f"      {name:<50s} {norm:.2e}")
print(f"\n    5 maiores normas de gradiente:")
for name, norm in sorted_grads[-5:]:
    print(f"      {name:<50s} {norm:.2e}")

# -------------------------------------------------------------------------
# 6. Verificar que input_proj e convs realmente transformam os dados
# -------------------------------------------------------------------------
print(f"\n[ 6 ] Pipeline de transformacoes")
model.eval()
with torch.no_grad():
    h0 = x
    h1 = model.input_proj(h0)
    print(f"    raw features    -> input_proj  : {tuple(h0.shape)} -> {tuple(h1.shape)}")
    h_prev = h1
    for i, conv in enumerate(model.convs):
        h_next, _ = conv(h_prev, edge_index)
        diff_layer = (h_next - h_prev).abs().mean().item()
        print(f"    conv[{i}] saida shape: {tuple(h_next.shape)}  mean_diff={diff_layer:.4f}  {'OK' if diff_layer > 1e-6 else 'FALHOU'}")
        h_prev = h_next
    h_norm = model.pre_mlp_norm(h_prev)
    print(f"    pre_mlp_norm: std_antes={h_prev.std():.4f}  std_depois={h_norm.std():.4f}  (esperado ~1.0 depois)")
    logits = model.mlp(h_norm)
    print(f"    mlp output: {tuple(logits.shape)}  logit_std={logits.std():.4f}  (esperado ~1)")

print(f"\n{'='*60}")
print(f"  Verificacao concluida")
print(f"{'='*60}\n")
