"""
visualizer/plot_physical.py
============================
物理層（Futto力学ネットワーク）の可視化。

出力:
  1. Futto 3Dネットワークスナップショット（IC・立脚中期・TO など）
  2. Strength Centrality 時系列折れ線グラフ（ハブ移動）
  3. ネットワーク効率・最大固有値の時系列グラフ
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from futto_network.build_graph import FuttoGraph
from futto_network.centrality  import CentralityResult
from futto_network.efficiency  import EfficiencyResult, spectral_gap

# ── 描画スタイル共通設定 ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 150,
})

TASK_COLORS = {
    'task01': '#1f77b4',
    'task02': '#ff7f0e',
    'task03': '#2ca02c',
}

# 歩行周期の代表イベント（%）
GAIT_EVENTS = {
    'IC'     : 0,    # Initial Contact
    'LR'     : 12,   # Loading Response
    'MSt'    : 31,   # Mid Stance
    'TSt'    : 50,   # Terminal Stance / 対側 IC
    'PSw'    : 62,   # Pre Swing
    'ISw'    : 75,   # Initial Swing
    'MSw'    : 87,   # Mid Swing
    'TSw'    : 100,  # Terminal Swing
}


# =============================================================================
# 1. Futto 3D ネットワーク スナップショット
# =============================================================================

def plot_futto_3d_snapshots(
    graph        : FuttoGraph,
    cr           : CentralityResult,
    marker_csv   : Optional[str | Path],
    save_dir     : Path,
    task_key     : str,
    phase        : int,
    speed        : str,
    events       : Optional[dict[str, int]] = None,
) -> None:
    """
    歩行周期の代表イベント時刻における Futto の 3D ネットワーク図を保存する。

    マーカー座標 CSV が存在する場合はそれを使用。
    存在しない場合はダミー配置でネットワーク構造のみ描画する。

    Parameters
    ----------
    marker_csv : マーカー座標CSV（gait_cycle_%, id, x, y, z 形式）
    events     : {'ラベル': %} の辞書。None なら GAIT_EVENTS を使用
    """
    if events is None:
        events = GAIT_EVENTS

    # マーカー座標の読み込み
    positions = _load_marker_positions(marker_csv)

    lines_def = CFG.TASK_CONFIGS.get(task_key, {}).get('LINES_TO_DRAW', {})

    # イベントごとに1枚ずつ保存
    for ev_name, ev_pct in events.items():
        t_idx = int(ev_pct)
        fig   = plt.figure(figsize=(9, 7))
        ax    = fig.add_subplot(111, projection='3d')

        strength_t = cr.strength[t_idx]          # shape (N,)
        s_max      = strength_t.max() if strength_t.max() > 0 else 1.0

        # ── エッジ描画 ──────────────────────────────────────────
        for seg_name, (id1, id2) in lines_def.items():
            tension_arr = graph.segment_tension(seg_name)
            if tension_arr is None:
                continue
            tension_val = tension_arr[t_idx]

            p1 = _get_pos(positions, id1, graph, t_idx)
            p2 = _get_pos(positions, id2, graph, t_idx)
            if p1 is None or p2 is None:
                continue

            # 張力 → 線幅・色
            norm_t  = tension_val / (graph.W.max() + 1e-9)
            lw      = 1.0 + norm_t * 6.0
            color   = cm.plasma(norm_t)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linewidth=lw, alpha=0.85)

        # ── ノード描画 ──────────────────────────────────────────
        for local_i, nid in enumerate(graph.node_ids):
            p = _get_pos(positions, nid, graph, t_idx)
            if p is None:
                continue
            s_norm = strength_t[local_i] / s_max
            size   = 30 + s_norm * 200
            color  = cm.RdYlGn(1.0 - s_norm)
            ax.scatter(*p, s=size, c=[color], zorder=5, edgecolors='k', linewidths=0.5)

            # ハブノードにはラベル
            if local_i == cr.hub_idx[t_idx]:
                ax.text(p[0], p[1], p[2] + 5, str(nid),
                        fontsize=7, color='red', fontweight='bold')

        ax.set_title(f"{task_key} Phase{phase} {speed}m/s  —  {ev_name} ({ev_pct}%)",
                     fontsize=11)
        ax.set_xlabel('X [mm]', fontsize=9)
        ax.set_ylabel('Y [mm]', fontsize=9)
        ax.set_zlabel('Z [mm]', fontsize=9)

        # カラーバー（張力スケール）
        sm = cm.ScalarMappable(cmap='plasma',
                               norm=plt.Normalize(0, graph.W.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Tension [N]', shrink=0.6, pad=0.1)

        fname = save_dir / f"futto_3d_{task_key}_ph{phase}_{speed}_{ev_name}.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

    print(f"  [3D] {len(events)} 枚保存 → {save_dir}")


def _load_marker_positions(
    marker_csv: Optional[str | Path],
) -> Optional[dict[int, np.ndarray]]:
    """マーカーCSVを読んで {id: shape(T,3)} を返す。失敗時は None。"""
    if marker_csv is None:
        return None
    path = Path(marker_csv)
    if not path.exists():
        return None
    try:
        import pandas as pd
        from scipy.interpolate import interp1d
        df  = pd.read_csv(path)
        pos = {}
        for mid, grp in df.groupby('id'):
            grp  = grp.sort_values('gait_cycle_%')
            gc   = grp['gait_cycle_%'].values
            xyz  = grp[['x', 'y', 'z']].values.astype(float)
            if len(gc) != 101:
                x_new = np.arange(0, 101, dtype=float)
                arr   = np.zeros((101, 3))
                for d in range(3):
                    arr[:, d] = interp1d(gc, xyz[:, d], kind='linear',
                                         fill_value=(xyz[0, d], xyz[-1, d]),
                                         bounds_error=False)(x_new)
                xyz = arr
            pos[int(mid)] = xyz
        return pos
    except Exception as e:
        print(f"  [警告] マーカーCSV読込失敗: {e}")
        return None


def _get_pos(
    positions : Optional[dict],
    marker_id : int,
    graph     : FuttoGraph,
    t_idx     : int,
) -> Optional[np.ndarray]:
    """マーカー座標を返す。positionsがNullなら仮想2D配置を使う。"""
    if positions is not None and marker_id in positions:
        return positions[marker_id][t_idx]

    # ── フォールバック：ノードインデックスから仮想配置 ──────────
    if marker_id not in graph.id_to_idx:
        return None
    i  = graph.id_to_idx[marker_id]
    N  = graph.N
    # 円周上に配置
    angle = 2 * np.pi * i / max(N, 1)
    return np.array([np.cos(angle) * 100, np.sin(angle) * 100, i * 20.0])


# =============================================================================
# 2. Strength Centrality 時系列（ハブ移動）
# =============================================================================

def plot_strength_timeseries(
    graph    : FuttoGraph,
    cr       : CentralityResult,
    save_dir : Path,
    task_key : str,
    phase    : int,
    speed    : str,
    top_n    : int = 6,
) -> None:
    """
    上位 top_n ノードの Strength Centrality 時系列を重ねてプロット。
    横軸: 歩行周期 0〜100%
    縦軸: Strength [N]
    """
    gait  = np.arange(0, 101)
    lines = CFG.TASK_CONFIGS.get(task_key, {}).get('LINES_TO_DRAW', {})

    # 平均強度でトップN選択
    mean_s = cr.mean_strength               # shape (N,)
    top_idx = np.argsort(mean_s)[::-1][:top_n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ── 上段: 各ノードの Strength ────────────────────────────
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    for ci, li in enumerate(top_idx):
        nid   = graph.node_ids[li]
        label = f"Node {nid}"
        ax.plot(gait, cr.strength[:, li], color=colors[ci],
                linewidth=1.8, label=label)

    # 歩行イベント縦線
    for ev, pct in GAIT_EVENTS.items():
        ax.axvline(pct, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.text(pct + 0.5, ax.get_ylim()[1] * 0.95, ev,
                fontsize=7, color='gray', va='top')

    ax.set_ylabel('Strength [N]')
    ax.set_title(f'Strength Centrality — {task_key} Ph{phase} {speed}m/s')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 下段: ハブノードの移動（時系列） ──────────────────────
    ax2 = axes[1]
    ax2.plot(gait, cr.hub_strength, color='crimson', linewidth=2, label='Hub strength')
    ax2.fill_between(gait, cr.hub_strength, alpha=0.2, color='crimson')

    # ハブが切り替わったタイミングをマーク
    prev_hub = cr.hub_idx[0]
    for t in range(1, 101):
        if cr.hub_idx[t] != prev_hub:
            ax2.axvline(t, color='navy', linestyle='--', linewidth=0.8, alpha=0.6)
            ax2.text(t, cr.hub_strength.max() * 0.9,
                     f"→{graph.node_ids[cr.hub_idx[t]]}",
                     fontsize=6, color='navy', rotation=90, va='top')
            prev_hub = cr.hub_idx[t]

    ax2.set_xlabel('Gait Cycle [%]')
    ax2.set_ylabel('Hub Strength [N]')
    ax2.set_title('Hub Node Trajectory')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"strength_timeseries_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Strength] 保存 → {fname.name}")


# =============================================================================
# 3. ネットワーク効率・最大固有値 時系列
# =============================================================================

def plot_efficiency_lambda(
    er_dict  : dict[str, EfficiencyResult],
    save_dir : Path,
    phase    : int,
    speed    : str,
) -> None:
    """
    複数タスクのネットワーク効率と最大固有値を1枚に重ねてプロット。

    Parameters
    ----------
    er_dict : {task_key: EfficiencyResult}
    """
    gait = np.arange(0, 101)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── ① 正規化ネットワーク効率 E_norm ─────────────────────
    ax = axes[0]
    for tk, er in er_dict.items():
        ax.plot(gait, er.efficiency_norm,
                color=TASK_COLORS.get(tk, 'black'),
                linewidth=2, label=CFG.TASK_TITLES.get(tk, tk))
    _add_gait_events(ax)
    ax.set_ylabel('E_norm')
    ax.set_title(f'Network Efficiency (normalized)  —  Phase{phase} {speed}m/s')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── ② 最大固有値 λ_max_norm ──────────────────────────────
    ax2 = axes[1]
    for tk, er in er_dict.items():
        ax2.plot(gait, er.lambda_max_norm,
                 color=TASK_COLORS.get(tk, 'black'),
                 linewidth=2, label=CFG.TASK_TITLES.get(tk, tk))
    _add_gait_events(ax2)
    ax2.set_ylabel('λ_max (normalized)')
    ax2.set_title('Maximum Eigenvalue (system stiffness)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── ③ スペクトルギャップ ────────────────────────────────
    ax3 = axes[2]
    for tk, er in er_dict.items():
        gap = spectral_gap(er)
        ax3.plot(gait, gap,
                 color=TASK_COLORS.get(tk, 'black'),
                 linewidth=2, label=CFG.TASK_TITLES.get(tk, tk))
    _add_gait_events(ax3)
    ax3.set_xlabel('Gait Cycle [%]')
    ax3.set_ylabel('Spectral Gap')
    ax3.set_title('Spectral Gap (λ₁ - λ₂)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 立脚期・遊脚期の背景色
    for ax in axes:
        ax.axvspan(0, 60, alpha=0.05, color='blue', label='Stance')
        ax.axvspan(60, 100, alpha=0.05, color='orange', label='Swing')

    plt.tight_layout()
    fname = save_dir / f"efficiency_lambda_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Efficiency] 保存 → {fname.name}")


def _add_gait_events(ax):
    for ev, pct in GAIT_EVENTS.items():
        ax.axvline(pct, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)


# =============================================================================
# 4. セグメントグループ別張力グラフ
# =============================================================================

def plot_segment_group_tensions(
    graph    : FuttoGraph,
    save_dir : Path,
    task_key : str,
    phase    : int,
    speed    : str,
) -> None:
    """
    SEGMENT_GROUPS (FK / FH_BK_BS / BH_BT_FA) ごとに
    個別セグメントの張力時系列を描画する。
    """
    from futto_network.centrality import segment_group_strength

    gait       = np.arange(0, 101)
    seg_groups = CFG.SEGMENT_GROUPS
    n_grp      = len(seg_groups)

    fig, axes = plt.subplots(n_grp, 1, figsize=(12, 4 * n_grp), sharex=True)
    if n_grp == 1:
        axes = [axes]

    for ax, (grp_name, seg_list) in zip(axes, seg_groups.items()):
        colors = plt.cm.Set2(np.linspace(0, 1, len(seg_list)))
        for ci, seg_name in enumerate(seg_list):
            arr = graph.segment_tension(seg_name)
            if arr is not None:
                ax.plot(gait, arr, color=colors[ci],
                        linewidth=1.8, label=seg_name)

        # グループ合計
        total = segment_group_strength(graph, grp_name)
        ax.plot(gait, total, color='black', linewidth=2.5,
                linestyle='--', label=f'{grp_name} Total', zorder=10)

        _add_gait_events(ax)
        ax.axvspan(0, 60, alpha=0.04, color='blue')
        ax.axvspan(60, 100, alpha=0.04, color='orange')
        ax.set_ylabel('Tension [N]')
        ax.set_title(f'Segment Group: {grp_name}')
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Gait Cycle [%]')
    fig.suptitle(f'{task_key} Phase{phase} {speed}m/s — Segment Group Tensions',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fname = save_dir / f"segment_tensions_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Segment] 保存 → {fname.name}")
