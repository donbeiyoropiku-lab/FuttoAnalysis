# =============================================================================
# phase_analysis/viz/phase_plot.py
#
# 役割:
#   クロスコリレーション・CRP の計算結果を可視化する。
# =============================================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..core.cross_correlation import find_phase_lag
from ..core.crp import compute_crp


def plot_grf_phase_average(grf_result: dict,
                           leg: str = 'L', component: str = 'Fz',
                           label: str = '',
                           show_all_cycles: bool = False,
                           save_path: 'str | None' = None,
                           show_plot: bool = True) -> plt.Figure:
    """
    get_grf_phase_average_series() の結果 (平均±標準偏差) を描画する。

    Parameters
    ----------
    grf_result      : dict  get_grf_phase_average_series() の戻り値
    leg, component  : ラベル表示用
    label           : タイトルに付加するラベル
    show_all_cycles : True の場合、個別周期を薄く重ねて表示する
    save_path, show_plot : 保存・表示制御
    """
    cycles = grf_result['cycles']
    mean   = grf_result['mean']
    std    = grf_result['std']
    n      = grf_result['n_cycles']
    n_total = grf_result['n_total']

    fig, ax = plt.subplots(figsize=(10, 4.5))

    if show_all_cycles:
        for c in grf_result['all_cycles']:
            ax.plot(cycles, c, color='gray', linewidth=0.6, alpha=0.25)

    ax.plot(cycles, mean, color='#1f77b4', linewidth=2.2, label=f'Mean (n={n})', zorder=5)
    ax.fill_between(cycles, mean - std, mean + std,
                    alpha=0.25, color='#1f77b4', label='±1 SD', zorder=4)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axvspan(0, 60, alpha=0.04, color='green')
    ax.axvspan(60, 100, alpha=0.04, color='orange')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel(f'{component} (%BW)')

    title = f'{leg} GRF {component} — Phase Average ({n}/{n_total} cycles)'
    if label:
        title += f'  [{label}]'
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig


def plot_signals_overlay(cycles: np.ndarray,
                         x: np.ndarray, y: np.ndarray,
                         x_label: str = 'Signal X',
                         y_label: str = 'Signal Y',
                         label: str = '',
                         save_path: 'str | None' = None,
                         show_plot: bool = True) -> plt.Figure:
    """
    2つの信号を正規化して重ね描きする (位相関係を視覚的に把握する)。

    Parameters
    ----------
    cycles  : np.ndarray  歩行周期(%) (T,)
    x, y    : np.ndarray  比較する2信号 (T,)
    x_label, y_label : 凡例ラベル
    label   : タイトルに付加するラベル
    save_path, show_plot : 保存・表示制御
    """
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cycles, x_norm, color='#1f77b4', linewidth=2.0, label=x_label)
    ax.plot(cycles, y_norm, color='#d62728', linewidth=2.0, label=y_label)
    ax.axvspan(0, 60, alpha=0.04, color='green')
    ax.axvspan(60, 100, alpha=0.04, color='orange')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel('Normalized amplitude [0-1]')
    title = f'{x_label} vs {y_label} (normalized)'
    if label:
        title += f'  [{label}]'
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig


def plot_cross_correlation(x: np.ndarray, y: np.ndarray,
                           x_label: str = 'Signal X',
                           y_label: str = 'Signal Y',
                           label: str = '',
                           max_lag_pct: float = 50.0,
                           base_cycle_s: 'float | None' = None,
                           save_path: 'str | None' = None,
                           show_plot: bool = True) -> plt.Figure:
    """
    クロスコリレーション曲線とピーク位置を描画する。

    Parameters
    ----------
    x, y        : np.ndarray  比較する2信号 (先行=x, 追従=y)
    x_label, y_label : 信号名 (タイトル・注釈に使用)
    label       : タイトルに付加するラベル
    max_lag_pct : 探索するラグの上限 (%)
    base_cycle_s: 1歩行周期の時間[秒] (ms換算用)
    save_path, show_plot : 保存・表示制御
    """
    result = find_phase_lag(x, y, max_lag_pct=max_lag_pct,
                            base_cycle_s=base_cycle_s)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result['lags_pct'], result['corr'], color='#534AB7', linewidth=1.8)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')

    # ピーク位置をマーク
    ax.scatter([result['lag_pct']], [result['peak_corr']],
              color='#D85A30', s=80, zorder=5, edgecolors='black', linewidths=0.8)

    direction = f'{y_label} lags {x_label}' if result['lag_pct'] > 0 else \
                f'{y_label} leads {x_label}' if result['lag_pct'] < 0 else \
                'in phase'
    ax.annotate(
        f"lag = {result['lag_pct']:.1f}%\n"
        f"({result['lag_ms']:.0f} ms)\n"
        f"r = {result['peak_corr']:.3f}\n"
        f"({direction})",
        xy=(result['lag_pct'], result['peak_corr']),
        xytext=(15, 15), textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='gray')
    )

    ax.set_xlim(-max_lag_pct, max_lag_pct)
    ax.set_xlabel(f'Lag (% of gait cycle)  [positive = {y_label} lags {x_label}]')
    ax.set_ylabel('Cross-correlation coefficient')
    title = f'Cross-Correlation: {x_label} vs {y_label}'
    if label:
        title += f'  [{label}]'
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig, result


def plot_crp_analysis(cycles: np.ndarray,
                      x: np.ndarray, y: np.ndarray,
                      x_label: str = 'Signal X',
                      y_label: str = 'Signal Y',
                      label: str = '',
                      dt: float = 1.0,
                      save_path: 'str | None' = None,
                      show_plot: bool = True) -> tuple[plt.Figure, dict]:
    """
    Continuous Relative Phase (CRP) の解析結果を描画する。

    上段: CRP の時系列 (歩行周期%)
    下段左: x の位相平面 (正規化位置 vs 正規化速度)
    下段右: y の位相平面

    Parameters
    ----------
    cycles  : np.ndarray  歩行周期(%) (T,)
    x, y    : np.ndarray  比較する2信号
    x_label, y_label : 信号名
    label   : タイトルに付加するラベル
    dt      : サンプリング間隔 (勾配計算用)
    save_path, show_plot : 保存・表示制御

    Returns
    -------
    fig, result_dict (compute_crp の戻り値)
    """
    result = compute_crp(x, y, dt=dt)

    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.5], hspace=0.55, wspace=0.3)

    ax_crp   = fig.add_subplot(gs[0, :])
    ax_unwrp = fig.add_subplot(gs[1, :])
    ax_px    = fig.add_subplot(gs[2, 0])
    ax_py    = fig.add_subplot(gs[2, 1])

    title = f'Continuous Relative Phase: {x_label} vs {y_label}'
    if label:
        title += f'  [{label}]'
    fig.suptitle(title, fontsize=12)

    # --- 上段: CRP 時系列 (ラップ版, -180〜180) ---
    ax_crp.plot(cycles, result['crp_deg'], color='#534AB7', linewidth=2.0)
    ax_crp.axhline(0, color='green', linewidth=0.8, linestyle=':', alpha=0.7)
    ax_crp.axhline(180, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    ax_crp.axhline(-180, color='red', linewidth=0.8, linestyle=':', alpha=0.5)
    ax_crp.axvspan(0, 60, alpha=0.04, color='green')
    ax_crp.axvspan(60, 100, alpha=0.04, color='orange')
    ax_crp.set_xlim(0, 100)
    ax_crp.set_ylim(-190, 190)
    ax_crp.set_xlabel('Gait Cycle (%)')
    ax_crp.set_ylabel('CRP (deg)')
    jump_note = '  [wrap jump detected]' if result['unwrap_jump_detected'] else ''
    ax_crp.set_title(
        f"Wrapped CRP (-180 to 180 deg)   mean|CRP| = {result['mean_abs_crp']:.1f} deg{jump_note}",
        fontsize=10
    )
    ax_crp.grid(True, linestyle='--', alpha=0.5)

    # --- 2段目: CRP 時系列 (アンラップ版, 連続値) ---
    ax_unwrp.plot(cycles, result['crp_deg_unwrapped'], color='#993C1D', linewidth=2.0)
    ax_unwrp.axhline(0, color='green', linewidth=0.8, linestyle=':', alpha=0.7)
    # ±180の倍数に薄く参照線を引く (アンラップ後どこまで回転したか分かるように)
    y_min = result['crp_deg_unwrapped'].min()
    y_max = result['crp_deg_unwrapped'].max()
    ref_start = int(np.floor(y_min / 180.0)) * 180
    ref_end   = int(np.ceil(y_max / 180.0)) * 180
    for ref in range(ref_start, ref_end + 180, 180):
        if ref == 0:
            continue
        ax_unwrp.axhline(ref, color='red', linewidth=0.6, linestyle=':', alpha=0.3)
    ax_unwrp.axvspan(0, 60, alpha=0.04, color='green')
    ax_unwrp.axvspan(60, 100, alpha=0.04, color='orange')
    ax_unwrp.set_xlim(0, 100)
    ax_unwrp.set_xlabel('Gait Cycle (%)')
    ax_unwrp.set_ylabel('CRP (deg, unwrapped)')
    ax_unwrp.set_title(
        'Unwrapped CRP (continuous - reveals whether the jump above is real)',
        fontsize=10
    )
    ax_unwrp.grid(True, linestyle='--', alpha=0.5)

    # --- 下段左: x の位相平面 ---
    ax_px.plot(result['x_norm'], result['v_x_norm'], color='#1f77b4', linewidth=1.5)
    ax_px.scatter([result['x_norm'][0]], [result['v_x_norm'][0]],
                  color='green', s=40, zorder=5, label='start (0%)')
    ax_px.set_xlabel(f'{x_label} (normalized)')
    ax_px.set_ylabel('Velocity (normalized)')
    ax_px.set_title(f'{x_label} phase plane', fontsize=10)
    ax_px.axhline(0, color='gray', linewidth=0.5)
    ax_px.axvline(0, color='gray', linewidth=0.5)
    ax_px.set_aspect('equal')
    ax_px.legend(fontsize=8)
    ax_px.grid(True, linestyle='--', alpha=0.3)

    # --- 下段右: y の位相平面 ---
    ax_py.plot(result['y_norm'], result['v_y_norm'], color='#d62728', linewidth=1.5)
    ax_py.scatter([result['y_norm'][0]], [result['v_y_norm'][0]],
                  color='green', s=40, zorder=5, label='start (0%)')
    ax_py.set_xlabel(f'{y_label} (normalized)')
    ax_py.set_ylabel('Velocity (normalized)')
    ax_py.set_title(f'{y_label} phase plane', fontsize=10)
    ax_py.axhline(0, color='gray', linewidth=0.5)
    ax_py.axvline(0, color='gray', linewidth=0.5)
    ax_py.set_aspect('equal')
    ax_py.legend(fontsize=8)
    ax_py.grid(True, linestyle='--', alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig, result


def plot_task_lag_comparison(lag_results: dict,
                             x_label: str = 'Signal X',
                             y_label: str = 'Signal Y',
                             save_path: 'str | None' = None,
                             show_plot: bool = True) -> plt.Figure:
    """
    タスク間 (task01/02/03) の位相遅れを棒グラフで比較する。

    Parameters
    ----------
    lag_results : dict  {task_name: find_phase_lag() の戻り値}
    x_label, y_label : 信号名
    save_path, show_plot : 保存・表示制御
    """
    tasks = list(lag_results.keys())
    lags_pct = [lag_results[t]['lag_pct'] for t in tasks]
    lags_ms  = [lag_results[t]['lag_ms']  for t in tasks]
    corrs    = [lag_results[t]['peak_corr'] for t in tasks]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = ['#1f77b4', '#d62728', '#2ca02c']
    bar_colors = colors[:len(tasks)]

    ax0.bar(tasks, lags_ms, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.6)
    ax0.axhline(0, color='black', linewidth=0.8)
    ax0.set_ylabel('Phase lag (ms)')
    ax0.set_title(f'Phase lag: {y_label} relative to {x_label}', fontsize=10)
    ax0.grid(True, axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(lags_ms):
        ax0.text(i, v, f'{v:.0f}', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=9)

    ax1.bar(tasks, corrs, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.6)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Peak correlation coefficient')
    ax1.set_title('Coupling strength', fontsize=10)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(corrs):
        ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Task Comparison — Phase Lag Analysis', fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    return fig