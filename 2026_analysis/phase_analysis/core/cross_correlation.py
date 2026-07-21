# =============================================================================
# phase_analysis/core/cross_correlation.py
#
# 役割:
#   歩行周期内の2つの周期信号 (例: 関節角度とゴム張力) の
#   クロスコリレーションを計算し、位相遅れ (ms, %) を求める。
#
# 手法:
#   歩行周期の信号は周期的 (0%と100%が同じ状態) であるため、
#   線形シフトではなく循環シフト (np.roll) によるクロスコリレーションを用いる。
#   FFT を用いて高速に計算する。
#
# 解釈:
#   x を先行信号 (例: 関節角度)、y を追従信号 (例: ゴム張力) として、
#   corr(lag) が正の lag で最大になるとき、
#   「y は x より lag 分だけ遅れている」と解釈する。
# =============================================================================

from __future__ import annotations

import numpy as np

from ..config.phase_config import cycle_pct_to_ms


def _z_normalize(x: np.ndarray) -> np.ndarray:
    """平均0・標準偏差1に正規化する。"""
    std = x.std()
    std = std if std > 1e-9 else 1e-9
    return (x - x.mean()) / std


def circular_cross_correlation(x: np.ndarray, y: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    周期信号 x, y の循環クロスコリレーションを計算する。

    corr[lag] = mean( x(t) * y(t + lag) )  (循環シフト)

    lag > 0 で相関が最大になるとき、y を lag だけ「過去」に戻すと x と
    一致する、すなわち y は x より lag 分「遅れて」同じ変化を示す
    (y(t) ≈ x(t - lag)) と解釈できる。

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)  同じ長さの周期信号 (振幅は問わない)

    Returns
    -------
    lags : np.ndarray  ラグ (サンプル単位, -n/2 〜 n/2 の範囲に整列済み)
    corr : np.ndarray  正規化相関係数 (-1〜1 程度)
    """
    n = len(x)
    assert len(y) == n, "x と y は同じ長さである必要があります。"

    x_n = _z_normalize(np.asarray(x, dtype=float))
    y_n = _z_normalize(np.asarray(y, dtype=float))

    # FFT を用いた循環相互相関
    # ifft(conj(FFT(x)) * FFT(y)) は corr[lag] = mean(x(t) * y(t+lag)) に対応する
    X = np.fft.fft(x_n)
    Y = np.fft.fft(y_n)
    corr_raw = np.fft.ifft(np.conj(X) * Y).real / n

    # ラグを 0..n-1 → -n/2..n/2 に並べ替え (中心が lag=0)
    lags = np.arange(n)
    lags_centered = np.where(lags > n // 2, lags - n, lags)
    order = np.argsort(lags_centered)

    return lags_centered[order], corr_raw[order]


def find_phase_lag(x: np.ndarray, y: np.ndarray,
                   n_cycle_points: 'int | None' = None,
                   max_lag_pct: 'float | None' = 50.0,
                   base_cycle_s: 'float | None' = None
                   ) -> dict:
    """
    クロスコリレーションのピークから位相遅れを求める。

    Parameters
    ----------
    x, y           : np.ndarray  歩行周期1周期分の信号 (先行=x, 追従=y)
    n_cycle_points : int または None  1周期のサンプル数 (None なら len(x))
    max_lag_pct    : float または None  探索するラグの上限 (歩行周期の%)
        None の場合は全ラグ範囲を探索する。
    base_cycle_s   : float または None  1歩行周期の時間[秒] (ms 変換用)

    Returns
    -------
    dict:
        'lag_pct'    : float  ピーク位置のラグ (歩行周期の%, 正=yがxより遅れ)
        'lag_ms'     : float  ピーク位置のラグ (ms)
        'peak_corr'  : float  ピーク位置の相関係数
        'lags_pct'   : np.ndarray  全ラグ (歩行周期の%)
        'corr'       : np.ndarray  全ラグに対する相関係数
    """
    n = n_cycle_points if n_cycle_points is not None else len(x)
    lags_samples, corr = circular_cross_correlation(x, y)
    lags_pct = lags_samples / n * 100.0

    if max_lag_pct is not None:
        mask = np.abs(lags_pct) <= max_lag_pct
        lags_pct_search = lags_pct[mask]
        corr_search = corr[mask]
    else:
        lags_pct_search = lags_pct
        corr_search = corr

    peak_idx = np.argmax(corr_search)
    lag_pct = float(lags_pct_search[peak_idx])
    peak_corr = float(corr_search[peak_idx])
    lag_ms = cycle_pct_to_ms(lag_pct, base_cycle_s=base_cycle_s)

    return {
        'lag_pct':   lag_pct,
        'lag_ms':    lag_ms,
        'peak_corr': peak_corr,
        'lags_pct':  lags_pct,
        'corr':      corr,
    }