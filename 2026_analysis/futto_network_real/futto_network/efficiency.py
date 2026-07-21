"""
futto_network/efficiency.py
============================
Futtoネットワークの力伝達効率（E）と最大固有値（λ_max）を計算する。

  E(t) = 1/(N(N-1)) * Σ_{i≠j} F_ij(t)
       ※ 距離 d_ij = 1/F_ij → 効率 = Σ F_ij / N(N-1)

  λ_max(t) : W(t) の最大固有値（システムの連成強度）

タスク間正規化:
  E_norm(t)        = E(t) / F_max_global
  λ_max_norm(t)    = λ_max(t) / (N * F_max_global)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from futto_network.build_graph import FuttoGraph

FORCE_FLOOR = 1e-6   # ゼロ除算保護


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class EfficiencyResult:
    task_key  : str
    phase     : int
    speed     : str
    N         : int
    T         : int

    efficiency        : np.ndarray   # E(t)         shape (T,)
    efficiency_norm   : np.ndarray   # E_norm(t)    shape (T,)
    lambda_max        : np.ndarray   # λ_max(t)     shape (T,)
    lambda_max_norm   : np.ndarray   # 正規化        shape (T,)
    all_eigenvalues   : np.ndarray   # 全固有値      shape (T, N)

    # 代表時刻（立脚期）の統計
    efficiency_mean : float = field(init=False)
    efficiency_peak : float = field(init=False)
    lambda_max_mean : float = field(init=False)
    lambda_max_peak : float = field(init=False)

    def __post_init__(self):
        self.efficiency_mean = float(self.efficiency.mean())
        self.efficiency_peak = float(self.efficiency.max())
        self.lambda_max_mean = float(self.lambda_max.mean())
        self.lambda_max_peak = float(self.lambda_max.max())


# =============================================================================
# コア計算
# =============================================================================

def _efficiency_at(W_t: np.ndarray) -> float:
    """
    単一時刻 t の張力行列から効率 E を計算する。
    E = Σ_{i≠j} F_ij / N(N-1)   (= off-diagonal 平均)
    """
    N = W_t.shape[0]
    if N <= 1:
        return 0.0
    mask   = ~np.eye(N, dtype=bool)
    F_vals = W_t[mask]
    return float(F_vals.sum() / (N * (N - 1)))


def _eigenvalues_at(W_t: np.ndarray) -> np.ndarray:
    """
    実対称行列 W_t の固有値を降順で返す。
    """
    evals = np.linalg.eigvalsh(W_t)
    return evals[::-1]


# =============================================================================
# 公開 API
# =============================================================================

def compute_efficiency(
    graph: FuttoGraph,
    phase: int = 0,
    speed: str = "",
) -> EfficiencyResult:
    """
    FuttoGraph からネットワーク効率・最大固有値を計算する。

    正規化基準:
      F_max_global = 全時刻・全エッジの最大張力
      E_norm       = E / F_max_global
      λ_max_norm   = λ_max / (N * F_max_global)
    """
    W   = graph.W       # (T, N, N)
    T, N, _ = W.shape

    efficiency    = np.zeros(T)
    lambda_max    = np.zeros(T)
    all_evals     = np.zeros((T, max(N, 1)))

    F_max = float(W.max()) if W.size > 0 and W.max() > 0 else 1.0

    for t in range(T):
        W_t = W[t]
        efficiency[t] = _efficiency_at(W_t)
        if N >= 1:
            evals          = _eigenvalues_at(W_t)
            lambda_max[t]  = evals[0]
            all_evals[t]   = evals

    efficiency_norm = efficiency / F_max
    lambda_norm     = lambda_max / (N * F_max) if N > 0 else lambda_max.copy()

    return EfficiencyResult(
        task_key         = graph.task_key,
        phase            = phase,
        speed            = speed,
        N                = N,
        T                = T,
        efficiency       = efficiency,
        efficiency_norm  = efficiency_norm,
        lambda_max       = lambda_max,
        lambda_max_norm  = lambda_norm,
        all_eigenvalues  = all_evals,
    )


def spectral_gap(result: EfficiencyResult) -> np.ndarray:
    """
    スペクトルギャップ Δλ(t) = λ_1 - λ_2。
    高値 = 単一ハブ支配、低値 = 分散型力伝達。
    """
    if result.N < 2:
        return np.zeros(result.T)
    return result.all_eigenvalues[:, 0] - result.all_eigenvalues[:, 1]


def phase_split_stats(result: EfficiencyResult) -> dict[str, float]:
    """
    立脚期 (0–60%) / 遊脚期 (60–100%) でネットワーク指標を分離する。
    """
    T        = result.T
    t_stance = slice(0, int(T * 0.60))
    t_swing  = slice(int(T * 0.60), T)

    e_st = float(result.efficiency[t_stance].mean())
    e_sw = float(result.efficiency[t_swing].mean())
    l_st = float(result.lambda_max[t_stance].mean())
    l_sw = float(result.lambda_max[t_swing].mean())

    return {
        'stance_E'        : e_st,
        'swing_E'         : e_sw,
        'E_ratio_st_sw'   : e_st / e_sw if e_sw > 0 else float('inf'),
        'stance_lambda'   : l_st,
        'swing_lambda'    : l_sw,
        'lambda_ratio_st_sw': l_st / l_sw if l_sw > 0 else float('inf'),
    }


def compare_efficiency(results: dict[str, EfficiencyResult]) -> dict:
    """
    複数タスクの正規化済み効率・固有値を比較する。
    """
    task_keys = sorted(results.keys())
    return {
        'task_keys'              : task_keys,
        'efficiency_mean_norm'   : [float(results[t].efficiency_norm.mean()) for t in task_keys],
        'efficiency_peak_norm'   : [float(results[t].efficiency_norm.max())  for t in task_keys],
        'lambda_max_mean_norm'   : [float(results[t].lambda_max_norm.mean()) for t in task_keys],
        'lambda_max_peak_norm'   : [float(results[t].lambda_max_norm.max())  for t in task_keys],
        'spectral_gap_mean'      : [float(spectral_gap(results[t]).mean())   for t in task_keys],
    }


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    from futto_network.build_graph import FuttoGraph

    print("=== Efficiency テスト ===\n")
    for tk in ['task01', 'task02']:
        g = FuttoGraph(tk)._fill_simulated()
        r = compute_efficiency(g, phase=3, speed='1.1')
        pd = phase_split_stats(r)
        print(f"{tk}: N={r.N}")
        print(f"  E mean (raw)  : {r.efficiency_mean:.4f} N")
        print(f"  E mean (norm) : {r.efficiency_norm.mean():.4f}")
        print(f"  λ_max mean    : {r.lambda_max_mean:.4f}")
        print(f"  λ_max (norm)  : {r.lambda_max_norm.mean():.4f}")
        print(f"  Spectral gap  : {spectral_gap(r).mean():.4f}")
        print(f"  Stance/Swing E ratio: {pd['E_ratio_st_sw']:.3f}")
        print()
