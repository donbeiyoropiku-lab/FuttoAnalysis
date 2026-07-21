# =============================================================================
# phase_analysis/core/crp.py
#
# 役割:
#   Continuous Relative Phase (CRP) を計算する。
#   歩行研究で標準的に使われる、2つの周期信号間の連続的な位相差の指標。
#
# 手順:
#   1. 各信号を振幅正規化する ( [-1, 1] の範囲に収める)
#   2. 正規化信号の時間微分 (角速度・張力速度に相当) を計算し、同様に正規化する
#   3. 位相角 φ(t) = atan2(正規化速度, 正規化位置) を計算する (位相平面上の角度)
#   4. CRP(t) = φ_x(t) - φ_y(t)  (信号 x に対する信号 y の相対位相)
#
# CRP の解釈:
#   CRP ≈ 0°   : 2信号が同位相 (in-phase) で変化している
#   CRP ≈ 180° : 2信号が逆位相 (anti-phase) で変化している
#   |CRP| の変化: 位相関係の時間的な安定性・協調パターンを表す
# =============================================================================

from __future__ import annotations

import numpy as np


def normalize_amplitude(x: np.ndarray) -> np.ndarray:
    """
    信号を振幅 [-1, 1] に正規化する ( (x - center) / amplitude )。

    center    = (max + min) / 2
    amplitude = (max - min) / 2
    """
    x = np.asarray(x, dtype=float)
    center    = (x.max() + x.min()) / 2.0
    amplitude = (x.max() - x.min()) / 2.0
    amplitude = amplitude if amplitude > 1e-9 else 1e-9
    return (x - center) / amplitude


def compute_phase_angle(x: np.ndarray, dt: float = 1.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    信号 x の位相角 (位相平面上の角度) を計算する。

    Parameters
    ----------
    x  : np.ndarray  周期信号 (振幅は問わない)
    dt : float       サンプリング間隔 (速度計算の勾配に使用)

    Returns
    -------
    phase   : np.ndarray  位相角 [rad]  (-π 〜 π)
    x_norm  : np.ndarray  正規化した位置信号 [-1, 1]
    v_norm  : np.ndarray  正規化した速度信号 [-1, 1] 程度
    """
    x_norm = normalize_amplitude(x)

    # 周期信号として循環勾配を計算 (歩行周期の 0% と 100% を隣接点とみなす)
    v = np.gradient(np.concatenate([x_norm[-1:], x_norm, x_norm[:1]]), dt)[1:-1]
    v_norm = normalize_amplitude(v)

    phase = np.arctan2(v_norm, x_norm)
    return phase, x_norm, v_norm


def compute_crp(x: np.ndarray, y: np.ndarray, dt: float = 1.0
                ) -> dict:
    """
    信号 x に対する信号 y の Continuous Relative Phase を計算する。

    Parameters
    ----------
    x, y : np.ndarray  同じ長さの周期信号 (例: 関節角度, ゴム張力)
    dt   : float       サンプリング間隔

    Returns
    -------
    dict:
        'crp_deg'            : np.ndarray  CRP [deg]  (-180 〜 180 にラップ)
        'crp_deg_unwrapped'  : np.ndarray  CRP [deg]  (アンラップ済み、連続値)
            ラップ版で ±180° 付近に見える瞬間的なジャンプは、多くの場合
            角度表現上の「巻き戻り」であって物理現象ではない。
            アンラップ版は np.unwrap (period=360) でこの巻き戻りを解消し、
            真の連続的な位相差の変化を復元したもの。
            ジャンプがアンラップ後も残っていれば、それは表示上の問題ではなく
            実際に位相関係が大きく変化したことを意味する。
        'phase_x'   : np.ndarray  x の位相角 [rad]
        'phase_y'   : np.ndarray  y の位相角 [rad]
        'x_norm'    : np.ndarray  正規化した x
        'v_x_norm'  : np.ndarray  正規化した x の速度
        'y_norm'    : np.ndarray  正規化した y
        'v_y_norm'  : np.ndarray  正規化した y の速度
        'mean_abs_crp' : float  |CRP| の周期平均 (ラップ版, 協調の目安)
        'unwrap_jump_detected' : bool
            ラップ版とアンラップ版で ±180° 付近のジャンプの有無に差がある場合 True。
            True の場合、①本当に位相関係が大きく回転した可能性と
            ②位相平面が原点付近を通過して位相角が不安定化した可能性の
            両方を疑う必要がある (後者は x_norm, y_norm が同時に 0 に
            近いタイミングと重なるかどうかで判別できる)。
    """
    assert len(x) == len(y), "x と y は同じ長さである必要があります。"

    phase_x, x_norm, v_x_norm = compute_phase_angle(x, dt)
    phase_y, y_norm, v_y_norm = compute_phase_angle(y, dt)

    crp_rad = phase_x - phase_y
    # -180 〜 180 度に正規化 (ラップ版)
    crp_deg = np.degrees(crp_rad)
    crp_deg = (crp_deg + 180.0) % 360.0 - 180.0

    # アンラップ版: ±180°境界での見かけ上のジャンプを解消した連続値
    # period=360 を指定し、度単位のまま扱う (numpy >= 1.21 が必要)
    try:
        crp_deg_unwrapped = np.unwrap(crp_deg, period=360.0)
    except TypeError:
        # numpy < 1.21: period引数が無いため、ラジアンで unwrap してから度に戻す
        crp_deg_unwrapped = np.degrees(np.unwrap(np.radians(crp_deg)))

    # ジャンプ検出: ラップ版の隣接差分が ±180°に近いのに
    # アンラップ版の隣接差分は小さい箇所があれば「見かけ上のジャンプ」ありと判定
    wrapped_diff = np.abs(np.diff(crp_deg))
    unwrapped_diff = np.abs(np.diff(crp_deg_unwrapped))
    jump_detected = bool(np.any((wrapped_diff > 150) & (unwrapped_diff < 30)))

    return {
        'crp_deg':              crp_deg,
        'crp_deg_unwrapped':    crp_deg_unwrapped,
        'phase_x':              phase_x,
        'phase_y':              phase_y,
        'x_norm':               x_norm,
        'v_x_norm':              v_x_norm,
        'y_norm':               y_norm,
        'v_y_norm':              v_y_norm,
        'mean_abs_crp':          float(np.mean(np.abs(crp_deg))),
        'unwrap_jump_detected':  jump_detected,
    }