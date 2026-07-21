"""
multilayer_network/fusion.py
=============================
物理層（Futto張力）・関節層（仮想関節角度相関）・生体層（EMG相関）の
3層を統合した多層ネットワークを構築する。

インタ層エッジ設計:
  Physical → Joint  : Futto ゴムの端点ノードを JOINT_CENTER_DEFS の
                       マーカーIDリストで解剖学的関節にマッピング
  Joint    → EMG    : MUSCLE_INDICATORS の markers を使い、
                       関節に最も近い筋群にエッジを張る
  Physical → EMG    : 同セグメントグループ内の直接結合（弱重み）

主要出力:
  - スープラ隣接行列（N_total × N_total）
  - 多層ページランク
  - 歩行効率スコア（3層統合）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from futto_network.build_graph    import FuttoGraph
from futto_network.efficiency     import EfficiencyResult
from futto_network.centrality     import CentralityResult
from emg_network.correlation      import EMGCorrelationResult
from emg_network.network_metrics  import EMGNetworkMetrics
from multilayer_network.joint_layer import JointNetworkResult


# =============================================================================
# インタ層マッピング（TASK_CONFIGS から動的生成）
# =============================================================================

def _build_futto_to_joint_map(task_key: str) -> dict[int, list[str]]:
    """
    Futto マーカーID → 解剖学的関節名リスト のマッピングを
    JOINT_CENTER_DEFS から動的に生成する。

    各関節の markers リストに含まれるマーカーIDを
    その関節にマッピングする。

    Returns
    -------
    dict: marker_id → [joint_name, ...]
    """
    cfg  = CFG.TASK_CONFIGS.get(task_key, {})
    defs = cfg.get('JOINT_CENTER_DEFS', {})
    mapping: dict[int, list[str]] = {}

    for jname, jdef in defs.items():
        for mid in jdef.get('markers', []):
            mapping.setdefault(mid, []).append(jname)

    return mapping


def _build_joint_to_emg_map(task_key: str) -> dict[str, list[int]]:
    """
    関節名 → EMG チャンネルインデックスリスト のマッピングを
    MUSCLE_INDICATORS から動的に生成する。

    MUSCLE_INDICATORS の markers リストから最も近いマーカーを持つ筋を選択。
    ここでは emg_col（例: "L_TA_mean"）から筋名を特定し、
    MUSCLE_NAMES 内のインデックスを返す。

    Returns
    -------
    dict: joint_name → [emg_channel_idx, ...]
    """
    cfg        = CFG.TASK_CONFIGS.get(task_key, {})
    mi_def     = cfg.get('MUSCLE_INDICATORS', {})
    jc_defs    = cfg.get('JOINT_CENTER_DEFS', {})
    all_muscles= CFG.MUSCLE_NAMES   # ['R_GM', ..., 'L_TA']

    # 関節ごとの「所属マーカーIDセット」
    joint_marker_sets: dict[str, set[int]] = {}
    for jname, jdef in jc_defs.items():
        joint_marker_sets[jname] = set(jdef.get('markers', []))

    result: dict[str, list[int]] = {jn: [] for jn in jc_defs}

    for m_name, m_info in mi_def.items():
        # emg_col: "L_TA_mean" → L_TA
        col = m_info.get('emg_col', '')
        ch_name = col.replace('_mean', '')   # "L_TA"
        if ch_name not in all_muscles:
            continue
        ch_idx = all_muscles.index(ch_name)

        # この筋のマーカーIDがどの関節セットと重なるか
        m_markers = set(m_info.get('markers', []))
        for jname, jset in joint_marker_sets.items():
            if m_markers & jset:   # 共通マーカーがあれば紐付け
                result[jname].append(ch_idx)

    return result


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class MultilayerResult:
    task_key   : str
    phase      : int
    speed      : str

    N_physical : int
    N_joint    : int   # = 3
    N_emg      : int   # = 16
    N_total    : int

    supra_adjacency      : np.ndarray   # (N_total, N_total)
    multilayer_pagerank  : np.ndarray   # (N_total,)
    multilayer_community : np.ndarray   # (N_total,)

    coupling_PJ  : float
    coupling_JE  : float
    coupling_PE  : float

    gait_efficiency_score : float
    contribution_physical : float
    contribution_joint    : float
    contribution_emg      : float

    node_labels : list[str] = field(default_factory=list)


# =============================================================================
# スープラ隣接行列の構築
# =============================================================================

def _build_supra(
    graph     : FuttoGraph,
    jn        : JointNetworkResult,
    emg_cr    : EMGCorrelationResult,
    task_key  : str,
    gait_t    : int = 50,
    w_PJ      : float = 1.0,
    w_JE      : float = 1.0,
    w_PE      : float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    """
    3層のスープラ隣接行列（N_total × N_total）を構築する。
    """
    N_P = graph.N
    N_J = 3             # Hip, Knee, Ankle
    N_E = len(CFG.MUSCLE_NAMES)
    N   = N_P + N_J + N_E

    supra  = np.zeros((N, N))
    labels = []

    # ── ノードラベル ─────────────────────────────────────────
    for nid in graph.node_ids:
        labels.append(str(nid))
    labels.extend(['Hip', 'Knee', 'Ankle'])
    labels.extend(CFG.MUSCLE_NAMES)

    # ── PP ブロック（物理層内部）────────────────────────────
    if N_P > 0:
        supra[:N_P, :N_P] = graph.adjacency_at(gait_t)

    # ── JJ ブロック（関節層内部）────────────────────────────
    supra[N_P:N_P+N_J, N_P:N_P+N_J] = jn.coupling_matrix[:N_J, :N_J]

    # ── EE ブロック（生体層内部）────────────────────────────
    supra[N_P+N_J:, N_P+N_J:] = emg_cr.adj_matrix

    # ── PJ ブロック（物理 ↔ 関節）───────────────────────────
    if N_P > 0:
        futto_to_joint = _build_futto_to_joint_map(task_key)
        joint_idx_map  = {'Hip': 0, 'Knee': 1, 'Ankle': 2}
        # 各 Futto ノードの平均張力（全時刻平均）
        node_mean_tension = graph.W.mean(axis=(0, 2))  # shape (N_P,)

        for local_i, nid in enumerate(graph.node_ids):
            jnames = futto_to_joint.get(nid, [])
            for jn_name in jnames:
                ji = joint_idx_map.get(jn_name)
                if ji is None:
                    continue
                w = node_mean_tension[local_i] * w_PJ
                supra[local_i, N_P + ji] = w
                supra[N_P + ji, local_i] = w

    # ── JE ブロック（関節 ↔ 筋）────────────────────────────
    joint_to_emg = _build_joint_to_emg_map(task_key)
    joint_idx_map = {'Hip': 0, 'Knee': 1, 'Ankle': 2}
    for jn_name, emg_idxs in joint_to_emg.items():
        ji = joint_idx_map.get(jn_name)
        if ji is None:
            continue
        ci = jn.coord_index[ji]
        for ei in emg_idxs:
            w = ci * w_JE
            supra[N_P + ji, N_P + N_J + ei] = w
            supra[N_P + N_J + ei, N_P + ji] = w

    # ── PE ブロック（物理 ↔ 筋、弱結合）───────────────────
    if N_P > 0:
        # SEGMENT_GROUPS の FK/FH_BK_BS/BH_BT_FA に対応する EMG インデックス
        seg_emg_map = {
            'FK'     : [CFG.MUSCLE_NAMES.index(m) for m in ['L_RF', 'L_VL'] if m in CFG.MUSCLE_NAMES],
            'FH_BK_BS': [CFG.MUSCLE_NAMES.index(m) for m in ['L_GM', 'L_ILIO', 'L_SOL'] if m in CFG.MUSCLE_NAMES],
            'BH_BT_FA': [CFG.MUSCLE_NAMES.index(m) for m in ['L_BF', 'L_ST', 'L_TA'] if m in CFG.MUSCLE_NAMES],
        }
        lines = CFG.TASK_CONFIGS.get(task_key, {}).get('LINES_TO_DRAW', {})
        seg_groups = CFG.SEGMENT_GROUPS

        for grp_name, seg_list in seg_groups.items():
            emg_idxs = seg_emg_map.get(grp_name, [])
            if not emg_idxs:
                continue
            for seg_name in seg_list:
                edge = lines.get(seg_name)
                if edge is None:
                    continue
                id1, id2 = edge
                for pid, node_id in enumerate(graph.node_ids):
                    if node_id in (id1, id2):
                        t_arr = graph.segment_tension(seg_name)
                        w_base = float(t_arr.mean()) * w_PE if t_arr is not None else 0.0
                        for ei in emg_idxs:
                            supra[pid, N_P + N_J + ei] = max(supra[pid, N_P + N_J + ei], w_base)
                            supra[N_P + N_J + ei, pid] = max(supra[N_P + N_J + ei, pid], w_base)

    return supra, labels


# =============================================================================
# 多層ページランク
# =============================================================================

def _pagerank(supra: np.ndarray, d: float = 0.85, max_iter: int = 300) -> np.ndarray:
    N   = supra.shape[0]
    A   = np.abs(supra)
    col = A.sum(axis=0)
    col[col == 0] = 1.0
    P   = A / col
    pr  = np.ones(N) / N
    for _ in range(max_iter):
        pr_new = d * P @ pr + (1 - d) / N
        pr_new /= pr_new.sum()
        if np.linalg.norm(pr_new - pr) < 1e-9:
            break
        pr = pr_new
    return pr_new


# =============================================================================
# 簡易 k-means
# =============================================================================

def _kmeans(X: np.ndarray, k: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N   = X.shape[0]
    if N <= k:
        return np.arange(N)
    centers = X[rng.choice(N, k, replace=False)]
    labels  = np.zeros(N, dtype=int)
    for _ in range(100):
        dists  = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_l  = np.argmin(dists, axis=1)
        if np.all(new_l == labels):
            break
        labels = new_l
        for ki in range(k):
            mask = labels == ki
            if mask.any():
                centers[ki] = X[mask].mean(axis=0)
    return labels


def _spectral_embed(supra: np.ndarray, n_dim: int = 3) -> np.ndarray:
    N = supra.shape[0]
    if N < 2:
        return np.zeros((N, n_dim))
    A = np.abs(supra)
    D = np.diag(A.sum(axis=1))
    L = D - A
    _, evecs = np.linalg.eigh(L)
    return evecs[:, 1:n_dim + 1]


# =============================================================================
# 歩行効率スコア
# =============================================================================

def _gait_score(
    er   : EfficiencyResult,
    jn   : JointNetworkResult,
    em   : EMGNetworkMetrics,
    w    : tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> tuple[float, float, float, float]:
    w_P, w_J, w_E = w
    c_P = min(max(float(er.efficiency_norm.mean()), 0), 1)
    c_J = min(max(float(jn.interlimb_proxy), 0), 1)
    c_E = min(max(float(em.global_efficiency), 0), 1)
    return w_P * c_P + w_J * c_J + w_E * c_E, w_P * c_P, w_J * c_J, w_E * c_E


# =============================================================================
# 公開 API
# =============================================================================

def build_multilayer_network(
    graph   : FuttoGraph,
    jn      : JointNetworkResult,
    emg_cr  : EMGCorrelationResult,
    emg_m   : EMGNetworkMetrics,
    er      : EfficiencyResult,
    gait_t  : int = 50,
) -> MultilayerResult:
    """3層を統合した多層ネットワークを構築する。"""
    task_key = graph.task_key
    N_P      = graph.N
    N_J      = 3
    N_E      = len(CFG.MUSCLE_NAMES)
    N        = N_P + N_J + N_E

    supra, labels = _build_supra(graph, jn, emg_cr, task_key, gait_t)

    # ページランク
    pr = _pagerank(supra)

    # コミュニティ
    if N > 3:
        comm = _kmeans(_spectral_embed(supra, 3), k=3, seed=0)
    else:
        comm = np.zeros(N, dtype=int)

    # インタ層結合強度（ブロック平均）
    def _block_mean(r1, c1, r2, c2):
        block = supra[r1:r2, c1:c2]
        nz    = block[block > 0]
        return float(nz.mean()) if len(nz) > 0 else 0.0

    c_PJ = _block_mean(0, N_P, N_P, N_P + N_J)
    c_JE = _block_mean(N_P, N_P + N_J, N_P + N_J, N)
    c_PE = _block_mean(0, N_P + N_J, N_P, N)

    score, cp, cj, ce = _gait_score(er, jn, emg_m)

    return MultilayerResult(
        task_key              = task_key,
        phase                 = er.phase,
        speed                 = er.speed,
        N_physical            = N_P,
        N_joint               = N_J,
        N_emg                 = N_E,
        N_total               = N,
        supra_adjacency       = supra,
        multilayer_pagerank   = pr,
        multilayer_community  = comm,
        coupling_PJ           = c_PJ,
        coupling_JE           = c_JE,
        coupling_PE           = c_PE,
        gait_efficiency_score = score,
        contribution_physical = cp,
        contribution_joint    = cj,
        contribution_emg      = ce,
        node_labels           = labels,
    )


def compare_multilayer(results: dict[str, MultilayerResult]) -> dict:
    task_keys = sorted(results.keys())
    return {
        'task_keys'              : task_keys,
        'gait_efficiency_score'  : [results[t].gait_efficiency_score for t in task_keys],
        'coupling_PJ'            : [results[t].coupling_PJ            for t in task_keys],
        'coupling_JE'            : [results[t].coupling_JE            for t in task_keys],
        'coupling_PE'            : [results[t].coupling_PE            for t in task_keys],
        'contribution_physical'  : [results[t].contribution_physical  for t in task_keys],
        'contribution_joint'     : [results[t].contribution_joint      for t in task_keys],
        'contribution_emg'       : [results[t].contribution_emg        for t in task_keys],
    }


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    from futto_network.build_graph    import FuttoGraph
    from futto_network.efficiency     import compute_efficiency
    from futto_network.centrality     import compute_strength_centrality
    from emg_network.correlation      import compute_emg_correlation, build_emg_csv_path
    from emg_network.network_metrics  import compute_emg_network_metrics
    from multilayer_network.joint_layer import compute_joint_network
    import numpy as np

    print("=== Multilayer Network テスト ===\n")
    ml_results = {}
    for tk in ['task01', 'task02']:
        g    = FuttoGraph(tk)._fill_simulated()
        er   = compute_efficiency(g, phase=3, speed='1.1')
        cr_e = compute_emg_correlation(build_emg_csv_path(tk, 3), tk, 3, '1.1')
        em   = compute_emg_network_metrics(cr_e)

        t = np.linspace(0, 2 * np.pi, 101)
        ja = {'Hip': 20*np.sin(t), 'Knee': 40*np.abs(np.sin(t)), 'Ankle': 15*np.sin(t+0.5)}
        jn = compute_joint_network(ja, tk, 3, '1.1')

        ml = build_multilayer_network(g, jn, cr_e, em, er)
        ml_results[tk] = ml

        top_nodes = [ml.node_labels[i] for i in np.argsort(ml.multilayer_pagerank)[-3:][::-1]]
        print(f"{tk}:")
        print(f"  N_total={ml.N_total}  GES={ml.gait_efficiency_score:.4f}")
        print(f"  PJ={ml.coupling_PJ:.4f}  JE={ml.coupling_JE:.4f}  PE={ml.coupling_PE:.4f}")
        print(f"  Top PageRank nodes: {top_nodes}")
        print()

    print("=== 比較 ===")
    comp = compare_multilayer(ml_results)
    for k, v in comp.items():
        print(f"  {k}: {v}")
