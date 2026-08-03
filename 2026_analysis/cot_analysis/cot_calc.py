# =============================================================================
# cot_analysis/cot_calc.py
#
# 役割:
#   CPETデータからCOT(Cost of Transport)とフルード数を算出する。
#
# 計算式は COT.ipynb (単純平均版) に準拠する:
#   各速度段階の後半 (STAGE_TAIL_WINDOW_SEC 秒) のVO2平均値を用いて
#       COT = VO2_avg[ml/min] / weight[kg] / (speed[m/s] * 60) / g
#   フルード数:
#       Fr = speed[m/s] / sqrt(g * leg_length[m])
#   COT-Fr フィッティング関数 (Minetti型):
#       COT(Fr) = a/Fr + b*Fr
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


def compute_stage_vo2_means(df, walking_start_sec: float, stage_duration_sec: float,
                             tail_window_sec: float, n_stages: int) -> np.ndarray:
    """
    各速度段階の「後半 tail_window_sec 秒」のVO2平均値を段階ごとに算出する。

    段階 i (i=0..n_stages-1) の全体区間:
        [walking_start_sec + i*stage_duration_sec, walking_start_sec + (i+1)*stage_duration_sec)
    解析対象(後半)区間:
        [段階終了 - tail_window_sec, 段階終了)

    Parameters
    ----------
    df : pd.DataFrame  cot_loader.load_cpet_txt() の出力 ('Time_sec', 'VO2' 列を持つ)
    walking_start_sec : float  歩行開始時刻(計測開始からの秒数)
    stage_duration_sec : float  各速度段階の長さ(秒)
    tail_window_sec : float  解析に使う後半区間の長さ(秒)
    n_stages : int  速度段階数

    Returns
    -------
    np.ndarray, shape (n_stages,)  各段階のVO2平均値 [ml/min] (データが無い場合 NaN)
    """
    means = []
    for i in range(n_stages):
        stage_end = walking_start_sec + (i + 1) * stage_duration_sec
        tail_start = stage_end - tail_window_sec
        seg = df[(df['Time_sec'] >= tail_start) & (df['Time_sec'] < stage_end)]
        means.append(seg['VO2'].mean() if not seg.empty else np.nan)
    return np.array(means, dtype=float)


def compute_rest_vo2_mean(df, rest_window_sec: tuple[float, float]) -> float:
    """
    安静時VO2平均値を算出する (QC参照用。COT計算式には使用しない)。

    Parameters
    ----------
    df : pd.DataFrame  'Time_sec', 'VO2' 列を持つ
    rest_window_sec : (start, end)  安静区間 [秒]
    """
    start, end = rest_window_sec
    seg = df[(df['Time_sec'] >= start) & (df['Time_sec'] < end)]
    return float(seg['VO2'].mean()) if not seg.empty else float('nan')


def compute_cot(vo2_ml_min, weight_kg: float, speed_m_s, g: float = 9.8) -> np.ndarray:
    """COT = VO2 / weight / (speed*60) / g (速度ごとに算出、配列対応)。"""
    vo2 = np.asarray(vo2_ml_min, dtype=float)
    speed = np.asarray(speed_m_s, dtype=float)
    return vo2 / weight_kg / (speed * 60.0) / g


def compute_froude(speed_m_s, leg_length_m: float, g: float = 9.8) -> np.ndarray:
    """Fr = speed / sqrt(g * leg_length) (配列対応)。"""
    speed = np.asarray(speed_m_s, dtype=float)
    return speed / np.sqrt(g * leg_length_m)


def _fit_func(fr, a, b):
    fr = np.asarray(fr, dtype=float)
    return a / fr + b * fr


def fit_cot_vs_froude(froude, cot, fr_search_bounds: tuple[float, float] = (1e-3, 2.0)) -> dict:
    """
    COT-Fr曲線に COT(Fr) = a/Fr + b*Fr をフィッティングし、
    フィット係数と曲線上の最小点(最も経済的な歩行速度に相当)を返す。

    Parameters
    ----------
    froude, cot : 配列  同じ長さ、NaNを含んでいてもよい(自動除外)
    fr_search_bounds : 最小値探索の範囲 (Fr)

    Returns
    -------
    dict: {'a', 'b', 'fr_min', 'cot_min'}

    Raises
    ------
    RuntimeError  フィッティングに使える有効な点が3点未満、または収束しない場合
    """
    froude = np.asarray(froude, dtype=float)
    cot = np.asarray(cot, dtype=float)
    mask = ~(np.isnan(froude) | np.isnan(cot))
    if mask.sum() < 3:
        raise RuntimeError(f"フィッティングに使える有効な点が不足しています ({mask.sum()}点)。")

    popt, _ = curve_fit(_fit_func, froude[mask], cot[mask])
    a, b = float(popt[0]), float(popt[1])
    res = minimize_scalar(lambda x: _fit_func(x, a, b), bounds=fr_search_bounds, method='bounded')
    return {'a': a, 'b': b, 'fr_min': float(res.x), 'cot_min': float(res.fun)}
