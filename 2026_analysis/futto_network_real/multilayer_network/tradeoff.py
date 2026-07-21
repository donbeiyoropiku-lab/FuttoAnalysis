"""
multilayer_network/tradeoff.py
================================
タスクE: 多層ネットワークの「肩代わり（Trade-off）」評価

コア仮説:
  「Futtoが力伝達を代替することで、神経筋システムの負担が軽減・単純化する」
  → E_Futto(t) が高いとき E_EMG(t) が低下するはずである

評価指標:
  - 相互相関係数（Cross-correlation）
  - E_Futto / E_EMG の比率（肩代わり効率）
  - 立脚期・遊脚期での負担移行量
  - タスク間比較（task01 vs task02 vs task03）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
from scipy.signal import correlate, correlation_lags

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "futto_common"))
import CONFIG as CFG
from futto_network.efficiency import EfficiencyResult
from emg_network.network_metrics import EMGNetworkMetrics


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class TradeoffResult:
    task_key    : str
    phase       : int
    speed       : str

    # 入力データ（時系列）
    E_futto     : np.ndarray   # shape (T,)  物理層効率（正規化済み）
    E_emg       : np.ndarray   # shape (T,)  EMG層効率（時変・スライディング）

    # ─── 相互相関 ───────────────────────────────────────────────
    # 全周期
    xcorr_full      : float    # lag=0 での相互相関係数
    xcorr_lag_opt   : int      # 最大相互相関を示す lag（正 = E_futto が先行）
    xcorr_max       : float    # 最大相互相関係数

    # ─── 比率・差分 ──────────────────────────────────────────────
    ratio_t         : np.ndarray   # E_Futto(t) / E_EMG(t)   shape (T,)
    ratio_mean      : float
    ratio_stance    : float        # 立脚期 (0–60%) 平均
    ratio_swing     : float        # 遊脚期 (60–100%) 平均

    # ─── 負担移行量 ──────────────────────────────────────────────
    # 立脚期に Futto が高く EMG が低い → 正値
    burden_shift_stance : float
    burden_shift_swing  : float

    # ─── Pearson 相関（逆相関の検定） ───────────────────────────
    pearson_r    : float
    pearson_p    : float
    is_tradeoff  : bool    # p<0.05 かつ r<0 なら True（逆相関 = 肩代わり成立）


@dataclass
class TradeoffComparison:
    """複数タスク間のトレードオフ比較結果。"""
    task_keys       : list[str]
    tradeoff_results: dict[str, TradeoffResult]

    # タスク間差分
    delta_xcorr     : dict[str, float]   # task01/02 - task03
    delta_ratio     : dict[str, float]
    tradeoff_verdict: dict[str, str]     # "成立" / "不明" / "不成立"


# =============================================================================
# EMG 効率の時変推定
# =============================================================================

def _emg_efficiency_timeseries(
    em_metrics : EMGNetworkMetrics,
    cr_sliding : np.ndarray,   # shape (T, 16, 16)
    threshold  : float = 0.3,
) -> np.ndarray:
    """
    スライディング相関行列から EMG ネットワーク効率の時系列を計算する。

    各時刻 t の相関行列に対して全域効率を計算する（計算コスト削減のため
    固有値ベース近似を使用）。

    E_approx(t) = λ_max(t) / trace(A(t))
    ※ 正確な Floyd-Warshall は101回繰り返すと重いため近似を使用
    """
    T = cr_sliding.shape[0]
    E_t = np.zeros(T)

    for t in range(T):
        A = np.abs(cr_sliding[t])
        np.fill_diagonal(A, 0)
        A = np.where(A >= threshold, A, 0.0)
        tr = np.trace(A @ A)
        if tr > 0:
            evals = np.linalg.eigvalsh(A)
            E_t[t] = float(evals.max() / (np.sqrt(tr) + 1e-12))
        else:
            E_t[t] = 0.0

    # 0-1 正規化
    e_max = E_t.max()
    if e_max > 0:
        E_t = E_t / e_max
    return E_t


# =============================================================================
# トレードオフ計算
# =============================================================================

def compute_tradeoff(
    er_phys    : EfficiencyResult,
    em_metrics : EMGNetworkMetrics,
    cr_sliding : np.ndarray,   # EMGCorrelationResult.corr_sliding
) -> TradeoffResult:
    """
    物理層と生体層の効率トレードオフを計算する。

    Parameters
    ----------
    er_phys    : 物理層（Futto）の効率結果
    em_metrics : EMG ネットワーク指標
    cr_sliding : shape (T, 16, 16) 時変 EMG 相関行列
    """
    T = er_phys.T

    # E_Futto(t): 既計算の正規化済み効率
    E_f = er_phys.efficiency_norm.copy()

    # E_EMG(t): スライディング窓から時変計算
    E_e = _emg_efficiency_timeseries(em_metrics, cr_sliding)

    # ─── 相互相関 ────────────────────────────────────────────────
    # lag=0 での Pearson r
    if E_f.std() > 1e-10 and E_e.std() > 1e-10:
        r0, p0 = stats.pearsonr(E_f, E_e)
    else:
        r0, p0 = 0.0, 1.0

    # 遅れを考慮した最大相互相関
    xcorr    = correlate(E_f - E_f.mean(), E_e - E_e.mean(), mode='full')
    lags     = correlation_lags(len(E_f), len(E_e), mode='full')
    lag_opt  = int(lags[np.argmax(xcorr)])
    xcorr_max = float(xcorr.max() / (np.sqrt(np.sum(E_f**2) * np.sum(E_e**2)) + 1e-12))

    # ─── 比率 ─────────────────────────────────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_t = np.where(E_e > 1e-9, E_f / E_e, 0.0)

    t_stance = slice(0, int(T * 0.60))
    t_swing  = slice(int(T * 0.60), T)

    r_mean    = float(ratio_t.mean())
    r_stance  = float(ratio_t[t_stance].mean())
    r_swing   = float(ratio_t[t_swing].mean())

    # ─── 負担移行量 ──────────────────────────────────────────────
    # Δ = E_futto_stance - E_emg_stance  (正 = Futto が立脚期に多く働く)
    bs_stance = float(E_f[t_stance].mean() - E_e[t_stance].mean())
    bs_swing  = float(E_f[t_swing].mean()  - E_e[t_swing].mean())

    is_tradeoff = bool(p0 < 0.05 and r0 < -0.1)

    return TradeoffResult(
        task_key             = er_phys.task_key,
        phase                = er_phys.phase,
        speed                = er_phys.speed,
        E_futto              = E_f,
        E_emg                = E_e,
        xcorr_full           = float(r0),
        xcorr_lag_opt        = lag_opt,
        xcorr_max            = xcorr_max,
        ratio_t              = ratio_t,
        ratio_mean           = r_mean,
        ratio_stance         = r_stance,
        ratio_swing          = r_swing,
        burden_shift_stance  = bs_stance,
        burden_shift_swing   = bs_swing,
        pearson_r            = float(r0),
        pearson_p            = float(p0),
        is_tradeoff          = is_tradeoff,
    )


def compare_tradeoff(
    results : dict[str, TradeoffResult]
) -> TradeoffComparison:
    """
    複数タスクのトレードオフ結果を比較する。
    task03（非装着）を基準に task01/02 の肩代わり効果を定量化。
    """
    task_keys = sorted(results.keys())

    baseline_r  = results.get('task03', None)
    delta_xcorr = {}
    delta_ratio = {}
    verdict     = {}

    for tk in task_keys:
        r = results[tk]
        if baseline_r is not None and tk != 'task03':
            delta_xcorr[tk] = round(r.xcorr_full - baseline_r.xcorr_full, 4)
            delta_ratio[tk] = round(r.ratio_mean  - baseline_r.ratio_mean,  4)
        else:
            delta_xcorr[tk] = 0.0
            delta_ratio[tk] = 0.0

        if r.is_tradeoff:
            verdict[tk] = "成立（逆相関有意）"
        elif r.pearson_p < 0.05 and r.pearson_r > 0.1:
            verdict[tk] = "不成立（正相関）"
        else:
            verdict[tk] = "不明（有意差なし）"

    return TradeoffComparison(
        task_keys        = task_keys,
        tradeoff_results = results,
        delta_xcorr      = delta_xcorr,
        delta_ratio      = delta_ratio,
        tradeoff_verdict = verdict,
    )


# =============================================================================
# CSV 保存ヘルパー
# =============================================================================

def save_tradeoff_results(result: TradeoffResult, out_dir: Path) -> None:
    """トレードオフ時系列データを CSV に保存する。"""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'gait_cycle_%'  : range(101),
        'E_futto_norm'  : result.E_futto,
        'E_emg_norm'    : result.E_emg,
        'ratio_Ef_Ee'   : result.ratio_t,
    }).to_csv(out_dir / "tradeoff_timeseries.csv",
              index=False, float_format='%.6f')

    summary = {
        'task_key'           : result.task_key,
        'phase'              : result.phase,
        'speed'              : result.speed,
        'pearson_r'          : round(result.pearson_r, 4),
        'pearson_p'          : round(result.pearson_p, 4),
        'is_tradeoff'        : result.is_tradeoff,
        'xcorr_full'         : round(result.xcorr_full, 4),
        'xcorr_lag_opt'      : result.xcorr_lag_opt,
        'ratio_mean'         : round(result.ratio_mean, 4),
        'ratio_stance'       : round(result.ratio_stance, 4),
        'ratio_swing'        : round(result.ratio_swing, 4),
        'burden_shift_stance': round(result.burden_shift_stance, 4),
        'burden_shift_swing' : round(result.burden_shift_swing, 4),
    }
    import json
    with open(out_dir / "tradeoff_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  [Tradeoff] r={result.pearson_r:.3f} p={result.pearson_p:.3f} "
          f"{'✓肩代わり成立' if result.is_tradeoff else '×不明'} → {out_dir}")
