"""
emg_network/network_metrics.py
================================
EMG相関ネットワークの指標（次数・コミュニティ・効率・安定性）を計算する。

CONFIG.MUSCLE_NAMES = ['R_GM', ..., 'R_TA', 'L_GM', ..., 'L_TA'] 順に対応。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from emg_network.correlation import EMGCorrelationResult


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class EMGNetworkMetrics:
    task_key      : str
    phase         : int
    speed         : str
    muscle_names  : list[str]

    weighted_degree   : np.ndarray   # shape (16,)
    betweenness       : np.ndarray   # shape (16,)
    clustering_coef   : np.ndarray   # shape (16,)
    community_labels  : np.ndarray   # shape (16,)

    global_efficiency : float
    modularity_Q      : float
    temporal_stability: float
    left_efficiency   : float
    right_efficiency  : float
    inter_efficiency  : float

    top_hub_muscles   : list[str] = field(default_factory=list)


# =============================================================================
# グラフ指標計算（外部ライブラリなしの実装）
# =============================================================================

def _weighted_degree(adj: np.ndarray) -> np.ndarray:
    return np.abs(adj).sum(axis=1)


def _clustering(adj: np.ndarray) -> np.ndarray:
    A = np.abs(adj)
    N = A.shape[0]
    k = (A > 0).sum(axis=1).astype(float)
    cc = np.zeros(N)
    for i in range(N):
        if k[i] < 2:
            continue
        nb = np.where(A[i] > 0)[0]
        tri = sum(
            (A[i, j] * A[i, h] * A[j, h]) ** (1 / 3)
            for j in nb for h in nb if j != h
        )
        cc[i] = tri / (k[i] * (k[i] - 1))
    return cc


def _floyd_warshall_distance(adj: np.ndarray) -> np.ndarray:
    """重み付き距離行列（重み=相関値→距離=1/相関）。"""
    N = adj.shape[0]
    A = np.abs(adj)
    with np.errstate(divide='ignore'):
        D = np.where(A > 0, 1.0 / A, np.inf)
    np.fill_diagonal(D, 0)
    for k in range(N):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def _betweenness(adj: np.ndarray) -> np.ndarray:
    N = adj.shape[0]
    D = _floyd_warshall_distance(adj)
    bc = np.zeros(N)
    for i in range(N):
        for s in range(N):
            for t_ in range(N):
                if s == t_ or s == i or t_ == i:
                    continue
                dst = D[s, t_]
                if np.isfinite(dst) and dst > 0:
                    if abs(D[s, i] + D[i, t_] - dst) < 1e-9:
                        bc[i] += 1.0
    denom = (N - 1) * (N - 2)
    if denom > 0:
        bc /= denom
    return bc


def _global_efficiency(adj: np.ndarray) -> float:
    N = adj.shape[0]
    if N <= 1:
        return 0.0
    D = _floyd_warshall_distance(adj)
    with np.errstate(divide='ignore'):
        inv_D = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
    return float(inv_D.sum() / (N * (N - 1)))


def _spectral_community(adj: np.ndarray) -> np.ndarray:
    """Fiedler ベクトル符号で2分割。"""
    N = adj.shape[0]
    A = np.abs(adj)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    evals, evecs = np.linalg.eigh(L)
    fiedler = evecs[:, 1]
    return (fiedler >= 0).astype(int)


def _modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    A = np.abs(adj)
    m = A.sum() / 2
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    N = len(labels)
    Q = sum(
        (A[i, j] - k[i] * k[j] / (2 * m))
        for i in range(N) for j in range(N)
        if labels[i] == labels[j]
    )
    return Q / (2 * m)


def _temporal_stability(corr_sliding: np.ndarray) -> float:
    """隣接フレーム間のフロベニウス距離から安定性を算出する。"""
    T = corr_sliding.shape[0]
    if T < 2:
        return 1.0
    diffs = [
        np.linalg.norm(corr_sliding[t + 1] - corr_sliding[t], 'fro')
        for t in range(T - 1)
    ]
    max_diff = max(diffs) if diffs else 1.0
    return float(1.0 - np.mean(diffs) / (max_diff + 1e-12))


# =============================================================================
# 公開 API
# =============================================================================

def compute_emg_network_metrics(
    cr: EMGCorrelationResult,
    top_k: int = 3,
) -> EMGNetworkMetrics:
    """
    EMGCorrelationResult からネットワーク指標を計算する。
    """
    adj  = cr.adj_matrix   # (16, 16)
    ch   = cr.muscle_names

    wd   = _weighted_degree(adj)
    cc   = _clustering(adj)
    bc   = _betweenness(adj)
    comm = _spectral_community(adj)
    Q    = _modularity(adj, comm)
    E    = _global_efficiency(adj)
    stab = _temporal_stability(cr.corr_sliding)

    # 左(8〜15) / 右(0〜7) サブグラフ
    left  = list(range(8, 16))
    right = list(range(0, 8))
    E_L   = _global_efficiency(adj[np.ix_(left, left)])
    E_R   = _global_efficiency(adj[np.ix_(right, right)])

    adj_LR = np.zeros_like(adj)
    for l in left:
        for r in right:
            adj_LR[l, r] = adj[l, r]
            adj_LR[r, l] = adj[r, l]
    E_inter = _global_efficiency(adj_LR)

    top_hubs = [ch[i] for i in np.argsort(wd)[::-1][:top_k]]

    return EMGNetworkMetrics(
        task_key          = cr.task_key,
        phase             = cr.phase,
        speed             = cr.speed,
        muscle_names      = ch,
        weighted_degree   = wd,
        betweenness       = bc,
        clustering_coef   = cc,
        community_labels  = comm,
        global_efficiency = E,
        modularity_Q      = Q,
        temporal_stability= stab,
        left_efficiency   = E_L,
        right_efficiency  = E_R,
        inter_efficiency  = E_inter,
        top_hub_muscles   = top_hubs,
    )


def compare_contralateral_effect(
    with_futto: EMGNetworkMetrics,
    without_futto: EMGNetworkMetrics,
) -> dict:
    """
    Futto装着 vs 非装着の筋協調ネットワーク変化をまとめる。
    """
    def _delta(a, b):
        return {'abs': float(a - b), 'ratio': float(a / b) if b != 0 else float('inf')}

    return {
        'global_efficiency' : _delta(with_futto.global_efficiency,  without_futto.global_efficiency),
        'modularity_Q'      : _delta(with_futto.modularity_Q,       without_futto.modularity_Q),
        'inter_efficiency'  : _delta(with_futto.inter_efficiency,    without_futto.inter_efficiency),
        'left_efficiency'   : _delta(with_futto.left_efficiency,     without_futto.left_efficiency),
        'right_efficiency'  : _delta(with_futto.right_efficiency,    without_futto.right_efficiency),
        'temporal_stability': _delta(with_futto.temporal_stability,  without_futto.temporal_stability),
        'hub_muscles_with'  : with_futto.top_hub_muscles,
        'hub_muscles_without': without_futto.top_hub_muscles,
    }


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    from emg_network.correlation import compute_emg_correlation, build_emg_csv_path

    print("=== EMG Network Metrics テスト ===\n")
    all_m = {}
    for tk in ['task01', 'task02', 'task03']:
        path = build_emg_csv_path(tk, phase=3)
        cr   = compute_emg_correlation(path, tk, phase=3, speed='1.1')
        m    = compute_emg_network_metrics(cr)
        all_m[tk] = m
        print(f"{tk}:")
        print(f"  Global Eff   : {m.global_efficiency:.4f}")
        print(f"  Modularity Q : {m.modularity_Q:.4f}")
        print(f"  Stability    : {m.temporal_stability:.4f}")
        print(f"  Inter Eff    : {m.inter_efficiency:.4f}")
        print(f"  Top Hubs     : {m.top_hub_muscles}")
        print()

    if 'task01' in all_m and 'task03' in all_m:
        print("=== Contralateral Effect (task01 vs task03) ===")
        ce = compare_contralateral_effect(all_m['task01'], all_m['task03'])
        for k, v in ce.items():
            print(f"  {k}: {v}")
