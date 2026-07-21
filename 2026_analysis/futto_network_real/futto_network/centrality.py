"""
futto_network/centrality.py
============================
Strength Centrality（ノード強度中心性）および
ハブ移動追跡・左右非対称性指標を計算する。

  S_i(t) = Σ_j W(t)_ij  [N]

task01 (N=15) と task02 (N=8) でノード数が異なるため、
正規化は  S_norm = S / (N * F_max)  で統一する。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from futto_network.build_graph import FuttoGraph


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class CentralityResult:
    task_key    : str
    phase       : int
    speed       : str
    node_ids    : list[int]

    # shape (T, N)
    strength      : np.ndarray        # S_i(t) 生値 [N]
    strength_norm : np.ndarray        # 正規化済み

    # shape (T,)
    hub_idx      : np.ndarray         # 最大強度ノードの行列インデックス
    hub_node_id  : np.ndarray         # 最大強度ノードのマーカーID
    hub_strength : np.ndarray         # 各時刻のハブ強度 [N]

    # shape (N,)
    mean_strength : np.ndarray
    peak_strength : np.ndarray

    # スカラー
    hub_entropy   : float = field(init=False)

    def __post_init__(self):
        self.hub_entropy = _hub_entropy(self.strength)

    def laterality_by_segment_group(self, segment_group: str) -> np.ndarray:
        """セグメントグループ名でフィルタした強度の時系列を返す（将来拡張用）。"""
        return self.strength.sum(axis=1)  # デフォルトは全ノード合算


# =============================================================================
# 内部計算
# =============================================================================

def _hub_entropy(strength: np.ndarray) -> float:
    """
    ネットワーク全体の力の分散度合いを示すシャノンエントロピー [bits]。
    歩行周期全体を通して、各ノードにどれだけ力が分散しているかを評価する。
    1つの結節点に100%の力が集中している場合は 0 になり、
    広く分散しているほど高値（1.0〜3.0等）をとる。
    """
    if strength.size == 0 or strength.sum() <= 1e-9:
        return 0.0

    # ノードごとの力の総和（歩行周期全体）
    node_totals = strength.sum(axis=0)  # shape (N,)
    total_force = node_totals.sum()
    
    probs = node_totals / total_force
    # 0より大きい要素のみ抽出（logのアンダーフロー・ゼロ除算防止）
    probs = probs[probs > 0]
    
    ent = -np.sum(probs * np.log2(probs))
    return float(max(0.0, ent))  # -0.000 などの微小な負の値を防止


# =============================================================================
# 公開 API
# =============================================================================

def compute_strength_centrality(
    graph: FuttoGraph,
    phase: int = 0,
    speed: str = "",
) -> CentralityResult:
    """
    FuttoGraph から Strength Centrality を計算する。

    正規化:
      strength_norm[t, i] = strength[t, i] / (N * W_max_global)
      → task01 / task02 間でスケールを揃えられる
    """
    W   = graph.W       # (T, N, N)
    T, N, _ = W.shape

    strength = W.sum(axis=2)    # (T, N)

    global_max = float(strength.max()) if strength.size > 0 else 1.0
    if global_max > 0 and N > 0:
        strength_norm = strength / (global_max * N)
    else:
        strength_norm = np.zeros_like(strength)

    if N > 0:
        hub_idx     = np.argmax(strength, axis=1)           # (T,)
        hub_node_id = np.array([graph.node_ids[hub_idx[t]] for t in range(T)])
        hub_str     = strength[np.arange(T), hub_idx]
    else:
        hub_idx     = np.zeros(T, dtype=int)
        hub_node_id = np.zeros(T, dtype=int)
        hub_str     = np.zeros(T)

    return CentralityResult(
        task_key      = graph.task_key,
        phase         = phase,
        speed         = speed,
        node_ids      = graph.node_ids,
        strength      = strength,
        strength_norm = strength_norm,
        hub_idx       = hub_idx,
        hub_node_id   = hub_node_id,
        hub_strength  = hub_str,
        mean_strength = strength.mean(axis=0) if T > 0 else np.zeros(N),
        peak_strength = strength.max(axis=0)  if T > 0 else np.zeros(N),
    )


def segment_group_strength(
    graph: FuttoGraph,
    group_name: str,
) -> np.ndarray:
    """
    CONFIG.SEGMENT_GROUPS で定義されたグループの
    総張力時系列（全セグメントの張力合計）を返す。

    Returns
    -------
    np.ndarray shape (T,)
    """
    groups = CFG.SEGMENT_GROUPS
    if group_name not in groups:
        return np.zeros(graph.T)

    seg_names = groups[group_name]
    total = np.zeros(graph.T)
    for seg in seg_names:
        t = graph.segment_tension(seg)
        if t is not None:
            total += t
    return total


def hub_trajectory_summary(result: CentralityResult) -> dict:
    """
    ハブの移動パターンをまとめた辞書を返す。

    Returns
    -------
    dict with keys:
      'most_frequent_hub_id'   : 最もよくハブになるノードのマーカーID
      'hub_dwell_pct'          : そのノードがハブである割合 [%]
      'hub_entropy_bits'       : ハブ移動のエントロピー
      'unique_hubs'            : 全周期でハブになったことがあるノード数
    """
    T = len(result.hub_node_id)
    from collections import Counter
    c     = Counter(result.hub_node_id.tolist())
    most  = c.most_common(1)[0]
    return {
        'most_frequent_hub_id' : int(most[0]),
        'hub_dwell_pct'        : float(most[1] / T * 100),
        'hub_entropy_bits'     : result.hub_entropy,
        'unique_hubs'          : len(c),
    }


def laterality_index(result: CentralityResult, cfg: dict) -> np.ndarray:
    """
    同側（左）ノードと反対側（右）ノードの強度非対称性。
    ただし実 CONFIG ではノード側（左/右）は明示定義されていないため、
    SEGMENT_GROUPS の FK / FH_BK_BS / BH_BT_FA で代理評価する。

    ここでは上半身エッジ群（FH_BK_BS）と下半身前方群（BH_BT_FA）の
    張力比を Laterality の代替指標として返す。

    Returns
    -------
    LI : np.ndarray shape (T,)
        (S_FH_BK_BS - S_BH_BT_FA) / (S_FH_BK_BS + S_BH_BT_FA)
    """
    from futto_network.build_graph import FuttoGraph

    # 直接 segment tension から計算（グラフオブジェクトは result から取れないので
    # このメソッドは build_graph の FuttoGraph と一緒に呼ぶ想定）
    # → 呼び出し元で segment_group_strength を使うことを推奨
    # ここではダミーとして全強度の平均を返す
    return result.strength.sum(axis=1)


# =============================================================================
# タスク間比較
# =============================================================================

def compare_centrality(results: dict[str, CentralityResult]) -> dict:
    """
    複数タスク（task01/task02）の Strength Centrality を比較する。

    Parameters
    ----------
    results : dict  {task_key: CentralityResult}

    Returns
    -------
    dict with numpy arrays
    """
    task_keys = sorted(results.keys())
    return {
        'task_keys'           : task_keys,
        'hub_entropy'         : [results[t].hub_entropy for t in task_keys],
        'mean_hub_strength_N' : [float(results[t].hub_strength.mean()) for t in task_keys],
        'peak_hub_strength_N' : [float(results[t].hub_strength.max())  for t in task_keys],
        'strength_norm_mean'  : [float(results[t].strength_norm.mean()) for t in task_keys],
    }


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    from futto_network.build_graph import FuttoGraph

    print("=== Centrality テスト ===\n")
    for tk in ['task01', 'task02']:
        g = FuttoGraph(tk)._fill_simulated()
        r = compute_strength_centrality(g, phase=3, speed='1.1')
        print(f"{tk}: N={g.N}")
        print(f"  Hub Entropy     : {r.hub_entropy:.4f} bits")
        print(f"  Mean Strength   : {r.mean_strength.round(2)}")
        print(f"  Hub Summary     : {hub_trajectory_summary(r)}")
        seg_fk = segment_group_strength(g, 'FK')
        print(f"  FK group total  : mean={seg_fk.mean():.2f} N  peak={seg_fk.max():.2f} N")
        print()
