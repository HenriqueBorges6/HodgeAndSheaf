"""
Análise pós-treino para BiSheafNodeClassifier.

Funções:
    extract_internals   — forward com return_internals=True
    _build_laplacians   — materializa L_F, L_C, L(F,C) densos (n*d × n*d)
    compute_spectra     — autovalores / valores singulares dos três Laplacianos
    map_statistics      — normas, SVs, assimetria F×C, correlação com homofilia
    diffusion_dynamics  — energia e mudança por camada
    node_analysis       — acurácia, energia, confiança por nó
    analyze_config      — orquestra tudo e salva JSON
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F_nn


# ---------------------------------------------------------------------------
# 1. Extrair internals
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_internals(model, data, device=None):
    """
    Roda forward com return_internals=True e move resultados para CPU.

    Retorna (logits, pred, internals) onde internals tem:
      reps   : list[(n, d*f)]  após input_proj e cada camada de difusão
      maps_F : list[(m, d, d)] mapas F por camada
      maps_C : list[(m, d, d)] mapas C por camada
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x   = data.x.to(device)
    ei  = data.edge_index.to(device)
    rev = data.reverse_idx.to(device)
    logits, internals = model(x, ei, rev, return_internals=True)
    internals["reps"]   = [r.cpu() for r in internals["reps"]]
    internals["maps_F"] = [m.cpu() for m in internals["maps_F"]]
    internals["maps_C"] = [m.cpu() for m in internals["maps_C"]]
    pred = logits.argmax(dim=1).cpu()
    return logits.cpu(), pred, internals


# ---------------------------------------------------------------------------
# 2. Materializar Laplacianos densos (n*d × n*d)
# ---------------------------------------------------------------------------

def _build_laplacians(maps_F, maps_C, reverse_idx, src, tgt, n, d):
    """
    Constrói três Laplacianos densos a partir dos mapas de uma camada.

    maps_F[i] = F_{src[i]⊴e},  maps_C[i] = C_{src[i]⊴e}
    (indexados pelo nó de origem, conforme saída do forward)

    Retorna (L_F, L_C, L_bi), cada (n*d, n*d):
      L_F  : Laplaciano simétrico PSD usando somente mapas F
      L_C  : Laplaciano simétrico PSD usando somente mapas C
      L_bi : Laplaciano BiSheaf não-simétrico
    """
    rev = reverse_idx.cpu()
    src = src.cpu()
    tgt = tgt.cpu()
    maps_F = maps_F.cpu().float()
    maps_C = maps_C.cpu().float()

    maps_F_tgt = maps_F[rev]   # F_{tgt[i]⊴e}
    maps_C_tgt = maps_C[rev]   # C_{tgt[i]⊴e}

    m = src.shape[0]

    # Pré-computa todos os blocos em batch (GPU ou BLAS)
    diag_F  = torch.bmm(maps_F_tgt.mT, maps_F_tgt)   # F_v^T F_v  (m, d, d)
    off_F   = torch.bmm(maps_F_tgt.mT, maps_F)        # F_v^T F_u  (m, d, d)
    diag_C  = torch.bmm(maps_C_tgt, maps_C_tgt.mT)    # C_v C_v^T  (m, d, d)
    off_C   = torch.bmm(maps_C_tgt, maps_C.mT)        # C_v C_u^T  (m, d, d)
    diag_bi = torch.bmm(maps_C_tgt, maps_F_tgt)       # C_v F_v    (m, d, d)
    off_bi  = torch.bmm(maps_C_tgt, maps_F)           # C_v F_u    (m, d, d)

    nd  = n * d
    L_F  = torch.zeros(nd, nd)
    L_C  = torch.zeros(nd, nd)
    L_bi = torch.zeros(nd, nd)

    for i in range(m):
        u = src[i].item()
        v = tgt[i].item()
        vs, ve = v * d, (v + 1) * d
        us, ue = u * d, (u + 1) * d
        L_F [vs:ve, vs:ve] += diag_F [i]
        L_F [vs:ve, us:ue] -= off_F  [i]
        L_C [vs:ve, vs:ve] += diag_C [i]
        L_C [vs:ve, us:ue] -= off_C  [i]
        L_bi[vs:ve, vs:ve] += diag_bi[i]
        L_bi[vs:ve, us:ue] -= off_bi [i]

    return L_F, L_C, L_bi


# ---------------------------------------------------------------------------
# 3. Análise espectral
# ---------------------------------------------------------------------------

def compute_spectra(L_F, L_C, L_bi, full_spectrum=True):
    """
    Retorna dicionário com espectros dos três Laplacianos.

    L_F, L_C : simétricos → eigvalsh (autovalores reais, ordenados)
    L_bi     : não-simétrico → eigvals (complexos) e svdvals (valores singulares)

    full_spectrum: se False, armazena só estatísticas resumidas (economiza espaço).
    """
    lam_F = torch.linalg.eigvalsh(L_F).numpy()
    lam_C = torch.linalg.eigvalsh(L_C).numpy()
    sv_bi = torch.linalg.svdvals(L_bi).numpy()
    eig_bi = torch.linalg.eigvals(L_bi)
    eig_bi_re = eig_bi.real.numpy()
    eig_bi_im = eig_bi.imag.numpy()

    def _gap(lam):
        pos = lam[lam > 1e-6]
        return float(pos[0]) if len(pos) > 0 else 0.0

    def _sym_stats(lam):
        st = {
            "spectral_gap": _gap(lam),
            "lambda_max":   float(lam.max()),
            "lambda_mean":  float(lam.mean()),
            "n_zero":       int((lam < 1e-6).sum()),
        }
        if full_spectrum:
            st["eigenvalues"] = lam.tolist()
        return st

    return {
        "L_F":  _sym_stats(lam_F),
        "L_C":  _sym_stats(lam_C),
        "L_bi": {
            "sv_max":          float(sv_bi.max()),
            "sv_min_nonzero":  float(sv_bi[sv_bi > 1e-6].min()) if (sv_bi > 1e-6).any() else 0.0,
            "sv_mean":         float(sv_bi.mean()),
            "n_near_zero_sv":  int((sv_bi < 1e-6).sum()),
            "max_imag":        float(np.abs(eig_bi_im).max()),
            "eig_re_min":      float(eig_bi_re.min()),
            "eig_re_max":      float(eig_bi_re.max()),
            **({"singular_values":     sv_bi.tolist(),
                "eigenvalues_real":    eig_bi_re.tolist(),
                "eigenvalues_imag":    eig_bi_im.tolist()} if full_spectrum else {}),
        },
    }


# ---------------------------------------------------------------------------
# 4. Estatísticas dos mapas
# ---------------------------------------------------------------------------

def map_statistics(maps_F, maps_C, src, tgt, data):
    """
    Estatísticas por aresta e globais dos mapas F e C de uma camada.

    Retorna dicionário com normas, SVs, assimetria F-C e correlação com homofilia.
    """
    m = maps_F.shape[0]
    d = maps_F.shape[1]

    # Norma de Frobenius por aresta
    norms_F = maps_F.reshape(m, -1).norm(dim=1).numpy()
    norms_C = maps_C.reshape(m, -1).norm(dim=1).numpy()

    # Assimetria ||F_e - C_e||_F
    asym = (maps_F - maps_C).reshape(m, -1).norm(dim=1).numpy()

    # Valores singulares médios por aresta
    sv_F = torch.linalg.svdvals(maps_F).numpy()   # (m, d)
    sv_C = torch.linalg.svdvals(maps_C).numpy()

    # ||F^T F - I||_F — mede afastamento da ortogonalidade
    I = torch.eye(d).unsqueeze(0).expand(m, -1, -1)
    orth_err_F = (torch.bmm(maps_F.mT, maps_F) - I).reshape(m, -1).norm(dim=1).numpy()
    orth_err_C = (torch.bmm(maps_C.mT, maps_C) - I).reshape(m, -1).norm(dim=1).numpy()

    # Correlação assimetria F×C com same-class (homofilia)
    y          = data.y.cpu().numpy()
    src_np     = src.cpu().numpy()
    tgt_np     = tgt.cpu().numpy()
    same_class = (y[src_np] == y[tgt_np]).astype(float)
    if same_class.std() > 0 and asym.std() > 0:
        corr = float(np.corrcoef(asym, same_class)[0, 1])
    else:
        corr = 0.0

    def _s(arr):
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
        }

    return {
        "n_edges":                     m,
        "d":                           d,
        "norm_F":                      _s(norms_F),
        "norm_C":                      _s(norms_C),
        "asymmetry_FC":                _s(asym),
        "sv_mean_per_edge_F":          _s(sv_F.mean(axis=1)),
        "sv_mean_per_edge_C":          _s(sv_C.mean(axis=1)),
        "orthogonality_error_F":       _s(orth_err_F),
        "orthogonality_error_C":       _s(orth_err_C),
        "homophily_asym_correlation":  corr,
    }


# ---------------------------------------------------------------------------
# 5. Dinâmica de difusão
# ---------------------------------------------------------------------------

def diffusion_dynamics(reps):
    """
    reps: list[(n, d*f)] — representações após input_proj (índice 0) e cada camada.
    Retorna lista de dicionários com estatísticas por camada de difusão.
    """
    layers = []
    for l in range(1, len(reps)):
        x_prev  = reps[l - 1]
        x_curr  = reps[l]
        delta   = (x_curr - x_prev).norm(dim=1)
        energy  = x_curr.norm(dim=1)
        e_prev  = x_prev.norm(dim=1)
        rel_chg = float(delta.mean() / (e_prev.mean() + 1e-8))
        layers.append({
            "layer":           l,
            "mean_change":     float(delta.mean()),
            "std_change":      float(delta.std()),
            "max_change":      float(delta.max()),
            "mean_energy":     float(energy.mean()),
            "std_energy":      float(energy.std()),
            "relative_change": rel_chg,
        })
    return layers


# ---------------------------------------------------------------------------
# 6. Análise por nó
# ---------------------------------------------------------------------------

def node_analysis(logits, pred, internals, data):
    """
    Acurácia, energia de representação e confiança por nó (correto vs errado).
    """
    y       = data.y.cpu()
    correct = (pred == y).numpy()

    last_rep = internals["reps"][-1]           # (n, d*f)
    energy   = last_rep.norm(dim=1).numpy()

    probs      = F_nn.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values.numpy()

    e_correct   = energy[correct]
    e_incorrect = energy[~correct]
    c_correct   = confidence[correct]
    c_incorrect = confidence[~correct]

    result = {
        "accuracy": {"overall": float(correct.mean())},
        "energy":   {
            "correct_mean":   float(e_correct.mean())   if len(e_correct)   > 0 else None,
            "incorrect_mean": float(e_incorrect.mean()) if len(e_incorrect) > 0 else None,
        },
        "confidence": {
            "correct_mean":   float(c_correct.mean())   if len(c_correct)   > 0 else None,
            "incorrect_mean": float(c_incorrect.mean()) if len(c_incorrect) > 0 else None,
        },
    }

    for split_name, mask in [
        ("train", data.train_mask),
        ("val",   data.val_mask),
        ("test",  data.test_mask),
    ]:
        m = mask.cpu().numpy()
        result["accuracy"][split_name] = float(correct[m].mean())  if m.sum() > 0 else None
        result["energy"  ][split_name] = float(energy[m].mean())   if m.sum() > 0 else None

    return result


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

def analyze_config(model, data, cfg_name: str, save_dir: str,
                   device=None, full_spectrum=True):
    """
    Roda todas as análises no modelo treinado e salva JSON em save_dir.

    Args:
        model:         BiSheafNodeClassifier treinado
        data:          PyG Data com edge_index, reverse_idx, x, y, *_mask
        cfg_name:      identificador da configuração
        save_dir:      diretório onde salvar {cfg_name}.json
        device:        dispositivo (default: mesmo do modelo)
        full_spectrum: se False, não armazena listas completas de autovalores

    Retorna o dicionário de resultados.
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = next(model.parameters()).device

    src = data.edge_index[0].cpu()
    tgt = data.edge_index[1].cpu()
    rev = data.reverse_idx.cpu()
    n   = data.x.shape[0]
    d   = model.d

    logits, pred, internals = extract_internals(model, data, device=device)

    layers_analysis = []
    for layer_idx, (mF, mC) in enumerate(zip(internals["maps_F"], internals["maps_C"])):
        map_stats = map_statistics(mF, mC, src, tgt, data)
        L_F, L_C, L_bi = _build_laplacians(mF, mC, rev, src, tgt, n, d)
        spectra = compute_spectra(L_F, L_C, L_bi, full_spectrum=full_spectrum)
        layers_analysis.append({
            "layer":          layer_idx,
            "map_statistics": map_stats,
            "spectra":        spectra,
        })

    dynamics   = diffusion_dynamics(internals["reps"])
    node_stats = node_analysis(logits, pred, internals, data)

    result = {
        "config":             cfg_name,
        "n_nodes":            n,
        "d":                  d,
        "n_layers":           len(internals["maps_F"]),
        "layers":             layers_analysis,
        "diffusion_dynamics": dynamics,
        "node_analysis":      node_stats,
    }

    out_path = os.path.join(save_dir, f"{cfg_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
