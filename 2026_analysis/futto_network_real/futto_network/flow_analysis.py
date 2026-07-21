"""
futto_network/flow_analysis.py
================================
タスクC: 物理層（Futto）の力伝達経路・ボトルネック解析

実装内容:
  C-1: Weighted Betweenness Centrality（ボトルネック特定）
       距離 d_ij = 1 / F_ij で定義し、力が伝わりやすいほど近い
  C-2: Flow Analysis（分岐点での張力分配比率）
       P_ij(t) = F_ij(t) / Σ F_i_out(t)
  C-3: Physical Community Detection（Louvain 法ベースのコミュニティ）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "futto_common"))
import CONFIG as CFG
from futto_network.build_graph import FuttoGraph


# =============================================================================
# C-1: Weighted Betweenness Centrality
# =============================================================================

@dataclass
class BetweennessResult:
    task_key   : str
    phase      : int
    speed      : str
    node_ids   : list[int]

    # shape (T, N)  各時刻・各ノードの媒介中心性
    betweenness_t : np.ndarray

    # shape (N,)  歩行周期平均
    betweenness_mean : np.ndarray

    # shape (T,)  最大媒介中心性ノードの時系列インデックス
    bottleneck_idx   : np.ndarray
    bottleneck_node_id: np.ndarray


def _dijkstra_all_pairs(W_t: np.ndarray) -> np.ndarray:
    """
    単一時刻の張力行列から全ペア最短経路距離行列を計算する。
    距離 d_ij = 1 / F_ij  (F_ij=0 のエッジは ∞)

    Returns
    -------
    D : np.ndarray shape (N, N)
    """
    N = W_t.shape[0]
    # 距離行列の初期化
    with np.errstate(divide='ignore'):
        D = np.where(W_t > 0, 1.0 / W_t, np.inf)
    np.fill_diagonal(D, 0.0)

    # Floyd-Warshall（N≤15なので十分高速）
    for k in range(N):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def _betweenness_at(W_t: np.ndarray) -> np.ndarray:
    """
    単一時刻の媒介中心性を計算する（Brandes アルゴリズム簡略版）。

    距離が等しい場合の最短路数を考慮した正規化版。

    Returns
    -------
    bc : np.ndarray shape (N,)
    """
    N = W_t.shape[0]
    if N <= 2:
        return np.zeros(N)

    D = _dijkstra_all_pairs(W_t)
    bc = np.zeros(N)

    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            d_st = D[s, t]
            if not np.isfinite(d_st) or d_st <= 0:
                continue
            for v in range(N):
                if v == s or v == t:
                    continue
                # v が s→t の最短路上にあるか
                if (np.isfinite(D[s, v]) and np.isfinite(D[v, t])
                        and abs(D[s, v] + D[v, t] - d_st) < 1e-9):
                    bc[v] += 1.0

    # 正規化: (N-1)(N-2) で割る
    denom = (N - 1) * (N - 2)
    if denom > 0:
        bc /= denom
    return bc


def compute_betweenness(
    graph : FuttoGraph,
    phase : int = 0,
    speed : str = "",
) -> BetweennessResult:
    """
    歩行周期全体の Weighted Betweenness Centrality を計算する。

    計算量が O(T * N^3) なので、N≤15 の task01 でも 101 * 15^3 ≈ 34万 演算で実用範囲。
    """
    W   = graph.W       # (T, N, N)
    T, N, _ = W.shape

    bc_t = np.zeros((T, N))
    for t in range(T):
        bc_t[t] = _betweenness_at(W[t])

    bc_mean       = bc_t.mean(axis=0)
    bottleneck_idx = np.argmax(bc_t, axis=1)   # 各時刻で最大
    if N > 0:
        btn_node_id = np.array([graph.node_ids[bottleneck_idx[t]] for t in range(T)])
    else:
        btn_node_id = np.zeros(T, dtype=int)

    return BetweennessResult(
        task_key          = graph.task_key,
        phase             = phase,
        speed             = speed,
        node_ids          = graph.node_ids,
        betweenness_t     = bc_t,
        betweenness_mean  = bc_mean,
        bottleneck_idx    = bottleneck_idx,
        bottleneck_node_id= btn_node_id,
    )


# =============================================================================
# C-2: Flow Analysis（張力分配比率）
# =============================================================================

@dataclass
class FlowResult:
    task_key  : str
    phase     : int
    speed     : str
    node_ids  : list[int]

    # {node_id: {seg_name: np.ndarray shape(T,)}}  各ノードの出力エッジへの分配比率
    flow_ratio : dict[int, dict[str, np.ndarray]]

    # 主要分岐ノードのサマリー
    # {node_id: {'dominant_seg': str, 'mean_ratio': float}}
    branch_summary : dict[int, dict]


def compute_flow_analysis(
    graph : FuttoGraph,
    phase : int = 0,
    speed : str = "",
) -> FlowResult:
    """
    各ノードを分岐点とみなし、出力エッジへの張力分配比率を計算する。

    P_ij(t) = F_ij(t) / Σ_k F_ik(t)

    「上流→下流」の向きは解剖学的な定義が難しいため、
    ここでは各ノードの接続エッジ全体に対する分配比率を算出する。
    特定ノードを「分岐点」として指定することで、
    指定ノードから出るエッジへの力の分配比率を取得できる。
    """
    lines    = CFG.TASK_CONFIGS.get(graph.task_key, {}).get('LINES_TO_DRAW', {})
    node_ids = graph.node_ids

    # ノードごとに接続セグメントをリストアップ
    node_segs: dict[int, list[str]] = {nid: [] for nid in node_ids}
    for seg_name, (id1, id2) in lines.items():
        if id1 in node_segs:
            node_segs[id1].append(seg_name)
        if id2 in node_segs:
            node_segs[id2].append(seg_name)

    flow_ratio: dict[int, dict[str, np.ndarray]] = {}
    branch_summary: dict[int, dict] = {}

    for nid in node_ids:
        segs = node_segs[nid]
        if len(segs) < 2:
            continue   # 分岐なし（端点）はスキップ

        # 各セグメントの張力時系列
        seg_tensions: dict[str, np.ndarray] = {}
        for seg in segs:
            arr = graph.segment_tension(seg)
            if arr is not None:
                seg_tensions[seg] = arr
            else:
                seg_tensions[seg] = np.zeros(graph.T)

        # 総張力（分母）
        total = np.zeros(graph.T)
        for arr in seg_tensions.values():
            total += arr

        # 比率計算（ゼロ除算防止）
        ratios: dict[str, np.ndarray] = {}
        for seg, arr in seg_tensions.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(total > 1e-9, arr / total, 0.0)
            ratios[seg] = r

        flow_ratio[nid] = ratios

        # サマリー: 平均的に最も多くの力を受け取るエッジ
        mean_ratios = {seg: float(r.mean()) for seg, r in ratios.items()}
        dominant    = max(mean_ratios, key=mean_ratios.get)
        branch_summary[nid] = {
            'dominant_seg'       : dominant,
            'dominant_mean_ratio': mean_ratios[dominant],
            'n_branches'         : len(segs),
            'mean_ratios'        : mean_ratios,
        }

    return FlowResult(
        task_key       = graph.task_key,
        phase          = phase,
        speed          = speed,
        node_ids       = node_ids,
        flow_ratio     = flow_ratio,
        branch_summary = branch_summary,
    )


# =============================================================================
# C-3: Physical Community Detection（Louvain 近似）
# =============================================================================

@dataclass
class CommunityResult:
    task_key   : str
    phase      : int
    speed      : str
    node_ids   : list[int]

    # shape (T, N) 各時刻のコミュニティラベル
    labels_t    : np.ndarray

    # shape (N,)  歩行周期で最頻出のコミュニティラベル
    labels_mode : np.ndarray

    # shape (T,)  モジュラリティ Q の時系列
    modularity_t : np.ndarray

    # shape (3, N): 代表時刻（IC=0, MSt=31, PSw=62）のラベル
    labels_events : np.ndarray

    # 解剖学的コミュニティの命名（自動割り当て）
    community_names : dict[int, str]


def _louvain_approximate(W_t: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """
    Louvain 法の近似実装（ランダム初期化 + 貪欲なラベル最適化）。
    外部ライブラリなしで動作する軽量版。

    Returns
    -------
    labels : np.ndarray shape (N,)
    """
    N      = W_t.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=int)

    A      = np.abs(W_t).copy()
    np.fill_diagonal(A, 0)
    m      = A.sum() / 2.0
    if m == 0:
        return np.zeros(N, dtype=int)

    k      = A.sum(axis=1)
    labels = np.arange(N, dtype=int)   # 各ノードが独立したコミュニティ

    improved = True
    itr      = 0
    while improved and itr < n_iter:
        improved = False
        itr += 1
        for i in np.random.permutation(N):
            current_label = labels[i]
            neighbor_labels = set(labels[np.where(A[i] > 0)[0]])
            neighbor_labels.add(current_label)

            best_label = current_label
            best_dQ    = 0.0

            for new_label in neighbor_labels:
                # コミュニティ内エッジ和の変化量 ΔQ を近似計算
                in_new   = A[i, labels == new_label].sum()
                in_cur   = A[i, labels == current_label].sum() - A[i, i]
                k_new    = k[labels == new_label].sum()
                k_cur    = k[labels == current_label].sum() - k[i]

                dQ = (in_new - in_cur) / m - k[i] * (k_new - k_cur) / (2 * m ** 2)
                if dQ > best_dQ:
                    best_dQ    = dQ
                    best_label = new_label

            if best_label != current_label:
                labels[i] = best_label
                improved   = True

    # ラベルを 0 から連番に振り直す
    unique_labels = sorted(set(labels))
    remap         = {old: new for new, old in enumerate(unique_labels)}
    return np.array([remap[l] for l in labels], dtype=int)


def _modularity_score(W_t: np.ndarray, labels: np.ndarray) -> float:
    """Newman-Girvan モジュラリティ Q を計算する。"""
    A = np.abs(W_t)
    m = A.sum() / 2.0
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    N = len(labels)
    Q = sum(
        (A[i, j] - k[i] * k[j] / (2 * m))
        for i in range(N) for j in range(N)
        if labels[i] == labels[j]
    ) / (2 * m)
    return float(Q)


def _assign_community_names(
    labels_mode : np.ndarray,
    node_ids    : list[int],
    task_key    : str,
) -> dict[int, str]:
    """
    各コミュニティに解剖学的名称を自動割り当てする。
    SEGMENTS（Hip/Thigh/Knee/Shank/Foot）との対応で命名する。
    """
    seg_def   = CFG.TASK_CONFIGS.get(task_key, {}).get('SEGMENTS', {})
    seg_names = list(seg_def.keys())   # ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']

    # 各コミュニティに最も多く含まれるセグメントの名称を割り当て
    n_comm = int(labels_mode.max()) + 1
    names  = {}
    for comm_id in range(n_comm):
        member_ids = [node_ids[i] for i, l in enumerate(labels_mode) if l == comm_id]
        best_seg   = 'Unknown'
        best_count = 0
        for seg, members in seg_def.items():
            count = len(set(member_ids) & set(members))
            if count > best_count:
                best_count = count
                best_seg   = seg
        names[comm_id] = best_seg

    return names


def compute_community(
    graph     : FuttoGraph,
    phase     : int = 0,
    speed     : str = "",
    n_iter    : int = 20,
) -> CommunityResult:
    """
    歩行周期全体に対してコミュニティ検出を実行する。

    代表イベント時刻（IC=0, MSt=31, PSw=62）のラベルも保存する。
    """
    W   = graph.W
    T, N, _ = W.shape

    labels_t     = np.zeros((T, N), dtype=int)
    modularity_t = np.zeros(T)

    np.random.seed(42)   # 再現性確保
    for t in range(T):
        lab            = _louvain_approximate(W[t], n_iter)
        labels_t[t]    = lab
        modularity_t[t]= _modularity_score(W[t], lab)

    # 各ノードの最頻出ラベル
    from scipy.stats import mode as scipy_mode
    if N > 0:
        labels_mode = scipy_mode(labels_t, axis=0, keepdims=False).mode
    else:
        labels_mode = np.zeros(N, dtype=int)

    # 代表イベント
    event_times   = [0, 31, 62]   # IC, MSt, PSw
    labels_events = labels_t[event_times, :]   # (3, N)

    community_names = _assign_community_names(labels_mode, graph.node_ids, graph.task_key)

    return CommunityResult(
        task_key        = graph.task_key,
        phase           = phase,
        speed           = speed,
        node_ids        = graph.node_ids,
        labels_t        = labels_t,
        labels_mode     = labels_mode,
        modularity_t    = modularity_t,
        labels_events   = labels_events,
        community_names = community_names,
    )


# =============================================================================
# CSV 保存ヘルパー
# =============================================================================

def save_flow_results(
    bt_result  : BetweennessResult,
    fl_result  : FlowResult,
    cm_result  : CommunityResult,
    out_dir    : Path,
) -> None:
    """C-1, C-2, C-3 の結果を CSV に保存する。"""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    # C-1: 媒介中心性
    bc_df = pd.DataFrame(
        bt_result.betweenness_t,
        columns=[str(n) for n in bt_result.node_ids],
    )
    bc_df.index.name = 'gait_cycle_%'
    bc_df.to_csv(out_dir / "betweenness_centrality.csv", float_format='%.6f')

    # C-2: フロー比率（各分岐ノードごとに1ファイル）
    for nid, ratios in fl_result.flow_ratio.items():
        fl_df = pd.DataFrame(ratios)
        fl_df.index.name = 'gait_cycle_%'
        fl_df.to_csv(out_dir / f"flow_ratio_node{nid}.csv", float_format='%.4f')

    # C-3: コミュニティラベル
    cm_df = pd.DataFrame(
        cm_result.labels_t,
        columns=[str(n) for n in cm_result.node_ids],
    )
    cm_df.index.name = 'gait_cycle_%'
    cm_df.to_csv(out_dir / "community_labels.csv")

    pd.DataFrame({
        'gait_cycle_%': range(101),
        'modularity_Q': cm_result.modularity_t,
    }).to_csv(out_dir / "modularity_timeseries.csv", index=False, float_format='%.6f')

    print(f"  [Flow/Community] 保存 → {out_dir}")
