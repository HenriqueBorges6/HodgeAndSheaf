"""
Sanity checks para CoSheafGNN.
Baseado na secao 10 do cosheafguide.md.

Rode com:
    python -m src.models.CoSheafGNN.sanity_check
"""

import torch
from .cosheaf_network import CoSheafNetwork
from .utils import build_incidence, build_line_graph_edge_index


# Grafo de teste:
#   v0 --e0-- v1 --e1-- v2
#              |
#             e2
#              |
#             v3

N, M, D = 4, 3, 2
EDGE_INDEX = torch.tensor([[0, 1, 1], [1, 2, 3]])
NODE_FEAT_DIM, EDGE_FEAT_DIM = 7, 4
HIDDEN_DIMS = [16, 16]
LEARNER_HIDDEN = [32]
NUM_CLASSES = 2


def build_block_boundary(B1, maps):
    """
    Constroi o operador de bordo por blocos B em R^{n*d x m*d}.
    B[v*d:(v+1)*d, e*d:(e+1)*d] = B1[v,e] * maps[e]
    """
    n, m = B1.shape
    d = maps.shape[1]
    B = torch.zeros(n * d, m * d)
    for v in range(n):
        for e in range(m):
            sign = B1[v, e].item()
            if sign != 0.0:
                B[v*d:(v+1)*d, e*d:(e+1)*d] = sign * maps[e]
    return B


def check_delta1_sdp(maps, verbose=True):
    """Check 1: Delta_1 = B^T B e simetrico e semidefinido positivo."""
    B1 = build_incidence(N, EDGE_INDEX)
    B  = build_block_boundary(B1, maps)
    Delta1 = B.T @ B

    sym  = torch.allclose(Delta1, Delta1.T, atol=1e-5)
    eigs = torch.linalg.eigvalsh(Delta1)
    sdp  = eigs.min().item() >= -1e-5

    if verbose:
        print(f"  [Check 1] Delta_1 simetrico: {'OK' if sym else 'FALHOU'}  |  "
              f"SDP (min eigval={eigs.min():.2e}): {'OK' if sdp else 'FALHOU'}")
    return sym and sdp


def check_maps_orthogonal(maps, verbose=True):
    """Check 2: F^T F = I para cada mapa."""
    QtQ = maps.transpose(-2, -1) @ maps
    eye = torch.eye(D).unsqueeze(0).expand(M, -1, -1)
    ok  = torch.allclose(QtQ, eye, atol=1e-4)
    if verbose:
        max_err = (QtQ - eye).abs().max().item()
        print(f"  [Check 2] Mapas ortogonais (max |Q^T Q - I|={max_err:.2e}): {'OK' if ok else 'FALHOU'}")
    return ok


def check_gradients(model, verbose=True):
    """Check 3: gradientes fluem para todos os parametros."""
    x_node = torch.randn(N, NODE_FEAT_DIM)
    x_edge = torch.randn(M, EDGE_FEAT_DIM)

    model.train()
    out  = model(x_node, x_edge, EDGE_INDEX)
    loss = out.sum()
    loss.backward()

    nan_params  = []
    none_params = []
    for name, p in model.named_parameters():
        if p.grad is None:
            none_params.append(name)
        elif torch.isnan(p.grad).any():
            nan_params.append(name)

    ok = (len(nan_params) == 0 and len(none_params) == 0)
    if verbose:
        total = sum(1 for _ in model.parameters())
        if ok:
            print(f"  [Check 3] Gradientes: OK ({total} parametros)")
        else:
            print(f"  [Check 3] Gradientes: FALHOU")
            for n in none_params:
                print(f"    grad=None : {n}")
            for n in nan_params:
                print(f"    grad=NaN  : {n}")
    return ok


def check_output_sensitivity(model, verbose=True):
    """Check 4: outputs diferentes para inputs diferentes."""
    model.eval()
    with torch.no_grad():
        out1 = model(torch.randn(N, NODE_FEAT_DIM), torch.randn(M, EDGE_FEAT_DIM), EDGE_INDEX)
        out2 = model(torch.randn(N, NODE_FEAT_DIM), torch.randn(M, EDGE_FEAT_DIM), EDGE_INDEX)
    ok = not torch.allclose(out1, out2)
    if verbose:
        print(f"  [Check 4] Sensibilidade ao input: {'OK' if ok else 'FALHOU (modelo colapsado!)'}")
    return ok


def run_all():
    torch.manual_seed(42)
    print("=" * 60)
    print("  SANITY CHECKS -- CoSheafGNN")
    print("=" * 60)

    model = CoSheafNetwork(
        node_feat_dim  = NODE_FEAT_DIM,
        edge_feat_dim  = EDGE_FEAT_DIM,
        d              = D,
        hidden_dims    = HIDDEN_DIMS,
        num_classes    = NUM_CLASSES,
        learner_hidden = LEARNER_HIDDEN,
    )

    # Extrai mapas da primeira camada via learner
    print("[DEBUG] construindo line_edge_index...")
    line_edge_index, _ = build_line_graph_edge_index(N, EDGE_INDEX)
    print(f"[DEBUG] line_edge_index shape: {line_edge_index.shape}")
    x_edge_dummy = torch.randn(M, HIDDEN_DIMS[0])
    print("[DEBUG] rodando learner...")
    with torch.no_grad():
        maps = model.convs[0].learner(x_edge_dummy, line_edge_index)  # (m, d, d)
    print(f"[DEBUG] maps shape: {maps.shape}, has nan: {maps.isnan().any()}")

    import traceback
    results = []
    for name, fn, args in [
        ("check_delta1_sdp",        check_delta1_sdp,        (maps,)),
        ("check_maps_orthogonal",   check_maps_orthogonal,   (maps,)),
        ("check_gradients",         check_gradients,         (model,)),
        ("check_output_sensitivity",check_output_sensitivity,(model,)),
    ]:
        print(f"[DEBUG] iniciando {name}...")
        try:
            results.append(fn(*args))
        except Exception:
            print(f"  ERRO em {name}:")
            traceback.print_exc()
            results.append(False)
        print(f"[DEBUG] {name} concluido")

    print("=" * 60)
    passed = sum(results)
    print(f"  {passed}/{len(results)} checks passaram")
    if passed == len(results):
        print("  TODOS OS CHECKS PASSARAM")
    else:
        print("  ATENCAO: ha checks falhando")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
