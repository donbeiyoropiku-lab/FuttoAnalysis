"""
visualizer/plot_emg.py
=======================
EMG生体ネットワーク層の可視化。

出力:
  1. EMG相関ヒートマップ（16×16、タスク比較）
  2. 筋協調サーキュラーネットワーク図（円周グラフ）
  3. 筋肉ごとの Degree/Strength 棒グラフ
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from emg_network.correlation    import EMGCorrelationResult
from emg_network.network_metrics import EMGNetworkMetrics

plt.rcParams.update({
    'font.family'    : 'sans-serif',
    'font.size'      : 10,
    'axes.titlesize' : 12,
    'figure.dpi'     : 150,
})

TASK_COLORS = {
    'task01': '#1f77b4',
    'task02': '#ff7f0e',
    'task03': '#2ca02c',
}

# 左右の筋色
COLOR_LEFT  = '#e74c3c'   # 赤系
COLOR_RIGHT = '#3498db'   # 青系


# =============================================================================
# 1. EMG 相関ヒートマップ
# =============================================================================

def plot_emg_heatmap(
    cr_dict  : dict[str, EMGCorrelationResult],
    save_dir : Path,
    phase    : int,
    speed    : str,
) -> None:
    """
    複数タスクの EMG 相関行列をヒートマップで並べて表示する。

    Parameters
    ----------
    cr_dict : {task_key: EMGCorrelationResult}
    """
    tasks    = list(cr_dict.keys())
    n_tasks  = len(tasks)
    muscles  = CFG.MUSCLE_NAMES   # 16筋

    fig, axes = plt.subplots(1, n_tasks, figsize=(7 * n_tasks, 6.5))
    if n_tasks == 1:
        axes = [axes]

    vmin, vmax = -1.0, 1.0

    for ax, tk in zip(axes, tasks):
        corr = cr_dict[tk].corr_matrix
        im   = ax.imshow(corr, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         aspect='equal', interpolation='nearest')

        ax.set_xticks(range(16))
        ax.set_yticks(range(16))
        ax.set_xticklabels(muscles, rotation=90, fontsize=7)
        ax.set_yticklabels(muscles, fontsize=7)

        # 左右ブロック境界線
        ax.axhline(7.5, color='k', linewidth=1.5)
        ax.axvline(7.5, color='k', linewidth=1.5)

        # ブロックラベル
        ax.text(3.5, -1.8, 'Right', ha='center', fontsize=9,
                color=COLOR_RIGHT, fontweight='bold')
        ax.text(11.5, -1.8, 'Left', ha='center', fontsize=9,
                color=COLOR_LEFT, fontweight='bold')

        ax.set_title(
            f"{CFG.TASK_TITLES.get(tk, tk)}\n"
            f"mean|r|={cr_dict[tk].mean_abs_corr:.3f}  "
            f"mod={cr_dict[tk].modularity_hint:.3f}",
            fontsize=10,
        )

    # 共通カラーバー
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Pearson r')

    fig.suptitle(f'EMG Correlation Matrix  —  Phase{phase} {speed}m/s',
                 fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fname = save_dir / f"emg_heatmap_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Heatmap] 保存 → {fname.name}")


# =============================================================================
# 2. 筋協調サーキュラーネットワーク図
# =============================================================================

def plot_emg_circular_network(
    cr       : EMGCorrelationResult,
    em       : EMGNetworkMetrics,
    save_dir : Path,
    task_key : str,
    phase    : int,
    speed    : str,
    threshold: float = 0.4,
) -> None:
    """
    16筋を円周上に配置し、相関が強い筋同士をエッジで結んだ
    サーキュラーネットワーク図を描画する。

    左脚（赤）・右脚（青）を上下半円に配置。
    エッジ色: 赤=同側相関、紫=左右クロス相関（対側性効果）
    エッジ幅: 相関強度に比例
    ノードサイズ: Weighted Degree に比例
    """
    muscles = CFG.MUSCLE_NAMES    # ['R_GM', ..., 'R_TA', 'L_GM', ..., 'L_TA']
    N       = len(muscles)
    corr    = cr.corr_matrix
    wd      = em.weighted_degree

    # ── ノード座標（円周配置）──────────────────────────────────
    # 右脚(0〜7): 上半円 (π→0)
    # 左脚(8〜15): 下半円 (0→-π)
    angles = np.zeros(N)
    for i in range(8):    # Right: 上半円
        angles[i] = np.pi - i * np.pi / 7
    for i in range(8):    # Left: 下半円
        angles[8 + i] = -i * np.pi / 7

    R_circle = 1.0
    xs = R_circle * np.cos(angles)
    ys = R_circle * np.sin(angles)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect('equal')
    ax.axis('off')

    # ── エッジ描画 ──────────────────────────────────────────────
    for i in range(N):
        for j in range(i + 1, N):
            r = corr[i, j]
            if abs(r) < threshold:
                continue

            # 同側 or 対側
            i_side = 'R' if i < 8 else 'L'
            j_side = 'R' if j < 8 else 'L'
            cross  = (i_side != j_side)

            lw    = abs(r) * 4.0
            alpha = 0.3 + abs(r) * 0.5
            color = '#9b59b6' if cross else (COLOR_RIGHT if i_side == 'R' else COLOR_LEFT)

            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color=color, linewidth=lw, alpha=alpha, zorder=1)

    # ── ノード描画 ──────────────────────────────────────────────
    wd_max = wd.max() if wd.max() > 0 else 1.0
    for i in range(N):
        side  = 'R' if i < 8 else 'L'
        color = COLOR_RIGHT if side == 'R' else COLOR_LEFT
        size  = 80 + (wd[i] / wd_max) * 400

        ax.scatter(xs[i], ys[i], s=size, c=color, zorder=5,
                   edgecolors='white', linewidths=1.5)

        # ラベル
        label_r = 1.15
        ha = 'left' if np.cos(angles[i]) > 0 else 'right'
        ax.text(label_r * xs[i], label_r * ys[i],
                muscles[i].replace('_', '\n'),
                ha=ha, va='center', fontsize=8, fontweight='bold',
                color=color)

    # ── 凡例 ───────────────────────────────────────────────────
    leg_handles = [
        mpatches.Patch(color=COLOR_RIGHT,  label='Right leg (ipsilateral)'),
        mpatches.Patch(color=COLOR_LEFT,   label='Left leg (ipsilateral)'),
        mpatches.Patch(color='#9b59b6',    label='Contralateral coupling'),
        Line2D([0], [0], color='gray', linewidth=1.5, label=f'Threshold |r|≥{threshold}'),
    ]
    ax.legend(handles=leg_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)

    # 中央テキスト
    cross_r = cr.corr_matrix[8:16, 0:8]
    mean_cr = float(np.abs(cross_r).mean())
    ax.text(0, 0,
            f"cross/ipsi\n= {mean_cr:.3f}",
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#2c3e50')

    ax.set_title(
        f'Muscle Coordination Network\n'
        f'{CFG.TASK_TITLES.get(task_key, task_key)} — Phase{phase} {speed}m/s',
        fontsize=13, pad=20,
    )

    fname = save_dir / f"emg_circular_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Circular] 保存 → {fname.name}")


# =============================================================================
# 3. Degree/Strength 棒グラフ（タスク比較）
# =============================================================================

def plot_emg_degree_comparison(
    em_dict  : dict[str, EMGNetworkMetrics],
    save_dir : Path,
    phase    : int,
    speed    : str,
) -> None:
    """
    複数タスクの Weighted Degree を筋肉ごとに棒グラフで比較。
    """
    tasks   = list(em_dict.keys())
    muscles = CFG.MUSCLE_NAMES
    N       = len(muscles)
    x       = np.arange(N)
    width   = 0.8 / len(tasks)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # ── 上段: Weighted Degree ──────────────────────────────────
    ax = axes[0]
    for ti, tk in enumerate(tasks):
        offset = (ti - len(tasks) / 2 + 0.5) * width
        wd = em_dict[tk].weighted_degree
        bars = ax.bar(x + offset, wd, width * 0.9,
                      label=CFG.TASK_TITLES.get(tk, tk),
                      color=TASK_COLORS.get(tk, f'C{ti}'), alpha=0.8)

    ax.axvline(7.5, color='k', linestyle='--', linewidth=1)
    ax.text(3.5, ax.get_ylim()[1] * 0.95, 'Right', ha='center',
            fontsize=10, color=COLOR_RIGHT)
    ax.text(11.5, ax.get_ylim()[1] * 0.95, 'Left', ha='center',
            fontsize=10, color=COLOR_LEFT)
    ax.set_ylabel('Weighted Degree')
    ax.set_title(f'Muscle Weighted Degree — Phase{phase} {speed}m/s')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # ── 下段: Betweenness Centrality ─────────────────────────
    ax2 = axes[1]
    for ti, tk in enumerate(tasks):
        offset = (ti - len(tasks) / 2 + 0.5) * width
        bc = em_dict[tk].betweenness
        ax2.bar(x + offset, bc, width * 0.9,
                label=CFG.TASK_TITLES.get(tk, tk),
                color=TASK_COLORS.get(tk, f'C{ti}'), alpha=0.8)

    ax2.axvline(7.5, color='k', linestyle='--', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(muscles, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Betweenness Centrality')
    ax2.set_title('Muscle Betweenness Centrality')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"emg_degree_comparison_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Degree] 保存 → {fname.name}")


# helper for Line2D import
from matplotlib.lines import Line2D
