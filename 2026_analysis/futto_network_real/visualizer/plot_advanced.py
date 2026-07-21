"""
visualizer/plot_advanced.py
============================
タスクC/D/E の高度解析結果を可視化する。

出力:
  1. Flow 経路図（C-2対応）: 分配比率を矢印の太さで可視化
  2. NMF シナジーヒートマップ（D-1対応）: W行列 + H時系列 + タスク比較
  3. 効率トレードオフ推移グラフ（E-1対応）: 二軸折れ線
  4. 媒介中心性（ボトルネック）時系列グラフ（C-1対応）
  5. コミュニティ遷移図（C-3対応）
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "futto_common"))
import CONFIG as CFG
from futto_network.build_graph   import FuttoGraph
from futto_network.flow_analysis import BetweennessResult, FlowResult, CommunityResult
from emg_network.synergy         import SynergyResult
from multilayer_network.tradeoff import TradeoffResult

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

GAIT_EVENTS = {'IC': 0, 'LR': 12, 'MSt': 31, 'TSt': 50, 'PSw': 62, 'ISw': 75, 'TSw': 100}


# =============================================================================
# 1. Flow 経路図（C-2）
# =============================================================================

def plot_flow_diagram(
    graph      : FuttoGraph,
    fl_result  : FlowResult,
    bt_result  : BetweennessResult,
    save_dir   : Path,
    task_key   : str,
    phase      : int,
    speed      : str,
    event_pct  : int = 31,   # 代表時刻（デフォルト: MSt=31%）
) -> None:
    """
    指定歩行位相における Futto ネットワーク上の力の流れを可視化する。

    エッジ: 矢印の太さ＝張力、色＝分配比率
    ノード: 大きさ＝媒介中心性、赤枠＝最大ボトルネック
    """
    t = event_pct
    lines_def = CFG.TASK_CONFIGS.get(task_key, {}).get('LINES_TO_DRAW', {})

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect('equal')
    ax.axis('off')

    # ── ノード配置（円周 + 解剖学的レイヤー） ──────────────────
    node_positions = _anatomical_layout(graph.node_ids, task_key)

    # ── エッジ（矢印）描画 ─────────────────────────────────────
    w_max = graph.W.max() + 1e-9
    for seg_name, (id1, id2) in lines_def.items():
        t_arr = graph.segment_tension(seg_name)
        if t_arr is None:
            continue
        tension = float(t_arr[t])
        if tension <= 0:
            continue

        p1 = node_positions.get(id1)
        p2 = node_positions.get(id2)
        if p1 is None or p2 is None:
            continue

        # 分配比率（分岐ノードから出るエッジの比率）
        ratio = 0.5   # デフォルト
        for nid, ratios in fl_result.flow_ratio.items():
            if seg_name in ratios:
                ratio = float(ratios[seg_name][t])
                break

        lw    = 1.0 + (tension / w_max) * 8.0
        color = cm.YlOrRd(ratio)   # 比率が高いほど赤

        ax.annotate(
            '', xy=p2, xytext=p1,
            arrowprops=dict(
                arrowstyle=f'-|>, head_width={0.03 + ratio*0.07}',
                color=color,
                lw=lw,
                alpha=0.85,
            )
        )

        # 比率ラベル（分岐点のみ）
        if ratio > 0.1:
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.text(mid_x, mid_y, f'{ratio:.0%}',
                    fontsize=7, ha='center', va='center',
                    color='navy', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6))

    # ── ノード描画 ─────────────────────────────────────────────
    bc_t   = bt_result.betweenness_t[t]    # shape (N,)
    bc_max = bc_t.max() if bc_t.max() > 0 else 1.0

    for i, nid in enumerate(graph.node_ids):
        p = node_positions.get(nid)
        if p is None:
            continue

        bc_norm   = bc_t[i] / bc_max
        size      = 200 + bc_norm * 800
        color     = cm.Reds(0.3 + bc_norm * 0.7)
        edgecolor = 'darkred' if nid == bt_result.bottleneck_node_id[t] else 'gray'
        edgelw    = 3.0      if nid == bt_result.bottleneck_node_id[t] else 0.8

        ax.scatter(*p, s=size, c=[color], zorder=5,
                   edgecolors=edgecolor, linewidths=edgelw)
        ax.text(p[0], p[1] + 0.06, str(nid),
                ha='center', fontsize=7, color='black', fontweight='bold')

    # ── 凡例 ──────────────────────────────────────────────────
    sm_ratio = cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1))
    sm_ratio.set_array([])
    plt.colorbar(sm_ratio, ax=ax, label='Flow ratio (分配比率)',
                 shrink=0.5, pad=0.02, location='right')

    ax.set_title(
        f'Flow Diagram — {CFG.TASK_TITLES.get(task_key, task_key)}\n'
        f'Phase{phase} {speed}m/s  @  {event_pct}% gait cycle',
        fontsize=12,
    )

    fname = save_dir / f"flow_diagram_{task_key}_ph{phase}_{speed}_t{event_pct}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Flow] 保存 → {fname.name}")


def _anatomical_layout(node_ids: list[int], task_key: str) -> dict[int, tuple[float, float]]:
    """
    SEGMENTS 定義に基づいてノードを解剖学的レイヤー（上→下）に配置する。
    Hip=上、Foot=下 の縦配置。
    """
    seg_def   = CFG.TASK_CONFIGS.get(task_key, {}).get('SEGMENTS', {})
    layer_y   = {'Hip': 1.0, 'Thigh': 0.7, 'Knee': 0.4, 'Shank': 0.1, 'Foot': -0.2}
    positions: dict[int, tuple[float, float]] = {}

    layer_members: dict[str, list[int]] = {}
    for seg, members in seg_def.items():
        layer_members[seg] = members
        for m in members:
            if m not in positions:
                y = layer_y.get(seg, 0.5)
                # 同一レイヤー内で等間隔に配置
                existing = [p for nid, p in positions.items()
                            if abs(p[1] - y) < 0.15]
                x = len(existing) * 0.25 - 0.3
                positions[m] = (x, y)

    # 未配置ノードは円周に配置
    unplaced = [nid for nid in node_ids if nid not in positions]
    for i, nid in enumerate(unplaced):
        angle = 2 * np.pi * i / max(len(unplaced), 1)
        positions[nid] = (0.8 * np.cos(angle), 0.8 * np.sin(angle))

    return positions


# =============================================================================
# 2. NMF シナジーヒートマップ（D-1）
# =============================================================================

def plot_synergy_heatmap(
    syn_results : dict[str, SynergyResult],
    save_dir    : Path,
    phase       : int,
    speed       : str,
) -> None:
    """
    複数タスクのシナジー重み行列（W）と活性化時系列（H）を並べて可視化する。

    上段: W 行列ヒートマップ（筋 × シナジー）
    下段: H 時系列（歩行周期 × シナジー）
    """
    tasks  = list(syn_results.keys())
    n_task = len(tasks)

    # 最大シナジー数を統一（少ないタスクはゼロパディング）
    max_k = max(r.n_synergies for r in syn_results.values())

    fig_h = 4 + 3 * n_task
    fig, axes_all = plt.subplots(
        n_task * 2, 1,
        figsize=(max_k * 2.5 + 3, fig_h),
        gridspec_kw={'height_ratios': [3, 2] * n_task}
    )
    if n_task == 1:
        axes_all = list(axes_all)

    gait = np.arange(0, 101)

    for ti, tk in enumerate(tasks):
        r    = syn_results[tk]
        ax_w = axes_all[ti * 2]
        ax_h = axes_all[ti * 2 + 1]

        # ── W 行列（筋重みヒートマップ）─────────────────────────
        W_plot = np.zeros((len(r.muscle_names), max_k))
        W_plot[:, :r.n_synergies] = r.W

        im = ax_w.imshow(W_plot.T, cmap='YlOrRd', aspect='auto',
                         vmin=0, vmax=W_plot.max() + 1e-9,
                         interpolation='nearest')

        ax_w.set_yticks(range(max_k))
        ax_w.set_yticklabels([f'Syn {k+1}' for k in range(max_k)], fontsize=9)
        ax_w.set_xticks(range(len(r.muscle_names)))
        ax_w.set_xticklabels(r.muscle_names, rotation=45, ha='right', fontsize=7)
        ax_w.set_title(
            f'{CFG.TASK_TITLES.get(tk, tk)}  N={r.n_synergies}  VAF={r.vaf_final:.3f}',
            fontsize=11, color=TASK_COLORS.get(tk, 'black'),
        )

        # 各シナジーの主動筋をアノテーション
        for k in range(r.n_synergies):
            dom = r.dominant_muscles[k][:2] if r.dominant_muscles else []
            ax_w.text(len(r.muscle_names) - 0.5, k,
                      '\n'.join(dom),
                      fontsize=6, va='center', ha='left',
                      color='navy')

        plt.colorbar(im, ax=ax_w, label='Weight', shrink=0.8)

        # ── H 時系列 ─────────────────────────────────────────────
        colors_h = plt.cm.Set1(np.linspace(0, 1, max_k))
        for k in range(r.n_synergies):
            ax_h.plot(gait, r.H[k], color=colors_h[k],
                      linewidth=1.8, label=f'Syn {k+1}')

        for ev, pct in GAIT_EVENTS.items():
            ax_h.axvline(pct, color='gray', linestyle=':', linewidth=0.7)
        ax_h.axvspan(0, 60, alpha=0.04, color='blue')
        ax_h.axvspan(60, 100, alpha=0.04, color='orange')
        ax_h.set_ylabel('Activation')
        ax_h.set_xlim(0, 100)
        ax_h.legend(loc='upper right', fontsize=7, ncol=min(r.n_synergies, 4))
        ax_h.grid(True, alpha=0.3)

    axes_all[-1].set_xlabel('Gait Cycle [%]')
    fig.suptitle(f'Muscle Synergy (NMF)  —  Phase{phase} {speed}m/s', fontsize=13)
    plt.tight_layout()

    fname = save_dir / f"synergy_heatmap_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Synergy] 保存 → {fname.name}")


def plot_synergy_comparison_bar(
    syn_results : dict[str, SynergyResult],
    save_dir    : Path,
    phase       : int,
    speed       : str,
) -> None:
    """
    タスク間のシナジー数・VAF 比較棒グラフ。
    「Futto装着でシナジー数が減少する（神経制御の単純化）」を視覚化。
    """
    tasks   = list(syn_results.keys())
    n_syns  = [syn_results[t].n_synergies for t in tasks]
    vafs    = [syn_results[t].vaf_final   for t in tasks]
    labels  = [CFG.TASK_TITLES.get(t, t)  for t in tasks]
    colors  = [TASK_COLORS.get(t, 'gray') for t in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # シナジー数
    ax = axes[0]
    bars = ax.bar(labels, n_syns, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, n_syns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(v), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Synergies')
    ax.set_title('Synergy Number Comparison\n(↓ = Neural Simplification)')
    ax.set_ylim(0, max(n_syns) + 1.5)
    ax.grid(axis='y', alpha=0.3)

    # VAF
    ax2 = axes[1]
    bars2 = ax2.bar(labels, vafs, color=colors, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars2, vafs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{v:.3f}', ha='center', fontsize=10)
    ax2.axhline(0.90, color='red', linestyle='--', linewidth=1.2, label='VAF=0.90 threshold')
    ax2.set_ylabel('VAF (Variance Accounted For)')
    ax2.set_title('Reconstruction Quality (VAF)')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Muscle Synergy Comparison  —  Phase{phase} {speed}m/s', fontsize=13)
    plt.tight_layout()

    fname = save_dir / f"synergy_comparison_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Synergy比較] 保存 → {fname.name}")


# =============================================================================
# 3. 効率トレードオフ推移グラフ（E-1）
# =============================================================================

def plot_tradeoff_timeseries(
    to_results : dict[str, TradeoffResult],
    save_dir   : Path,
    phase      : int,
    speed      : str,
) -> None:
    """
    E_Futto(t) と E_EMG(t) を二軸折れ線グラフで表示する。

    左軸: E_Futto（物理層が仕事している量）
    右軸: E_EMG（神経筋系の負担）
    """
    tasks = list(to_results.keys())
    gait  = np.arange(0, 101)
    n_t   = len(tasks)

    fig, axes = plt.subplots(n_t, 1, figsize=(13, 4.5 * n_t), sharex=True)
    if n_t == 1:
        axes = [axes]

    for ax, tk in zip(axes, tasks):
        r     = to_results[tk]
        color = TASK_COLORS.get(tk, 'black')
        ax2   = ax.twinx()

        # E_Futto: 左軸（実線）
        l1, = ax.plot(gait, r.E_futto, color=color,
                      linewidth=2.5, label='$E_{Futto}$ (Physical)',
                      linestyle='-')
        ax.fill_between(gait, r.E_futto, alpha=0.12, color=color)

        # E_EMG: 右軸（破線）
        l2, = ax2.plot(gait, r.E_emg, color=color,
                       linewidth=2.0, label='$E_{EMG}$ (Biological)',
                       linestyle='--', alpha=0.8)
        ax2.fill_between(gait, r.E_emg, alpha=0.06, color=color)

        # 立脚期・遊脚期の背景
        ax.axvspan(0, 60, alpha=0.05, color='steelblue')
        ax.axvspan(60, 100, alpha=0.05, color='darkorange')
        ax.text(30, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
                'Stance', ha='center', fontsize=9, color='steelblue', alpha=0.8)
        ax.text(80, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.95,
                'Swing', ha='center', fontsize=9, color='darkorange', alpha=0.8)

        for ev, pct in GAIT_EVENTS.items():
            ax.axvline(pct, color='gray', linestyle=':', linewidth=0.7)

        # 統計情報テキスト
        verdict_str = '✓ Trade-off 成立' if r.is_tradeoff else '△ 不明'
        info_str = (f"r={r.pearson_r:.3f} (p={r.pearson_p:.3f})  "
                    f"Ratio(stance)={r.ratio_stance:.2f}  {verdict_str}")
        ax.set_title(
            f'{CFG.TASK_TITLES.get(tk, tk)} — Phase{phase} {speed}m/s\n{info_str}',
            fontsize=10, color=color,
        )
        ax.set_ylabel('$E_{Futto}$ (normalized)', color=color)
        ax2.set_ylabel('$E_{EMG}$ (normalized)', color=color)
        ax.grid(True, alpha=0.25)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Gait Cycle [%]')
    fig.suptitle('Efficiency Trade-off: Futto vs Neuromuscular System',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    fname = save_dir / f"tradeoff_timeseries_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Tradeoff] 保存 → {fname.name}")


# =============================================================================
# 4. 媒介中心性（ボトルネック）時系列グラフ（C-1）
# =============================================================================

def plot_betweenness_timeseries(
    bt_result : BetweennessResult,
    save_dir  : Path,
    task_key  : str,
    phase     : int,
    speed     : str,
    top_n     : int = 5,
) -> None:
    """
    上位 top_n ノードの Betweenness Centrality 時系列を描画する。
    ボトルネック（力の中継局）がどこにあるかを時系列で追跡。
    """
    gait    = np.arange(0, 101)
    bc_mean = bt_result.betweenness_mean
    top_idx = np.argsort(bc_mean)[::-1][:top_n]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ── 上段: 各ノードの Betweenness 時系列 ──────────────────
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    for ci, li in enumerate(top_idx):
        nid   = bt_result.node_ids[li]
        ax.plot(gait, bt_result.betweenness_t[:, li],
                color=colors[ci], linewidth=1.8, label=f'Node {nid}')

    for ev, pct in GAIT_EVENTS.items():
        ax.axvline(pct, color='gray', linestyle=':', linewidth=0.7)
    ax.axvspan(0, 60, alpha=0.05, color='blue')
    ax.axvspan(60, 100, alpha=0.05, color='orange')
    ax.set_ylabel('Betweenness Centrality')
    ax.set_title(f'Betweenness (Bottleneck) Centrality — {task_key} Ph{phase} {speed}m/s')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 下段: ボトルネックノードの移動 ───────────────────────
    ax2 = axes[1]
    bc_max_t = bt_result.betweenness_t.max(axis=1)   # 各時刻の最大BC
    ax2.plot(gait, bc_max_t, color='darkred', linewidth=2)
    ax2.fill_between(gait, bc_max_t, alpha=0.2, color='darkred')

    prev = bt_result.bottleneck_idx[0]
    for t in range(1, 101):
        cur = bt_result.bottleneck_idx[t]
        if cur != prev:
            ax2.axvline(t, color='navy', linestyle='--', linewidth=0.7)
            ax2.text(t, bc_max_t.max() * 0.9,
                     f'→{bt_result.node_ids[cur]}',
                     fontsize=6, color='navy', rotation=90, va='top')
            prev = cur

    ax2.set_xlabel('Gait Cycle [%]')
    ax2.set_ylabel('Max Betweenness')
    ax2.set_title('Bottleneck Node Trajectory')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"betweenness_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Betweenness] 保存 → {fname.name}")


# =============================================================================
# 5. コミュニティ遷移図（C-3）
# =============================================================================

def plot_community_transition(
    cm_result : CommunityResult,
    save_dir  : Path,
    task_key  : str,
    phase     : int,
    speed     : str,
) -> None:
    """
    歩行周期中のノードのコミュニティ帰属変化を可視化する。
    横軸: 歩行周期, 縦軸: ノード, 色: コミュニティラベル
    """
    if len(cm_result.node_ids) == 0:
        return

    N       = len(cm_result.node_ids)
    gait    = np.arange(0, 101)
    n_comm  = int(cm_result.labels_t.max()) + 1
    cmap    = plt.cm.get_cmap('Set2', n_comm)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── 上段: コミュニティ帰属ヒートマップ ───────────────────
    ax = axes[0]
    ax.imshow(cm_result.labels_t.T, aspect='auto', cmap=cmap,
              vmin=-0.5, vmax=n_comm - 0.5,
              extent=[0, 100, -0.5, N - 0.5])
    ax.set_yticks(range(N))
    ax.set_yticklabels([str(n) for n in cm_result.node_ids], fontsize=7)
    for ev, pct in GAIT_EVENTS.items():
        ax.axvline(pct, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(pct, N - 0.3, ev, ha='center', fontsize=7, color='white')
    ax.set_xlabel('Gait Cycle [%]')
    ax.set_ylabel('Node ID')
    ax.set_title(f'Community Assignment — {task_key} Ph{phase} {speed}m/s')

    # コミュニティ名の凡例
    patches = [mpatches.Patch(color=cmap(i),
                               label=f'Comm {i}: {cm_result.community_names.get(i, "?")}')
               for i in range(n_comm)]
    ax.legend(handles=patches, loc='lower right', fontsize=8, ncol=2)

    # ── 下段: モジュラリティ Q 時系列 ────────────────────────
    ax2 = axes[1]
    ax2.plot(gait, cm_result.modularity_t, color='purple', linewidth=2)
    ax2.fill_between(gait, cm_result.modularity_t, alpha=0.2, color='purple')
    for ev, pct in GAIT_EVENTS.items():
        ax2.axvline(pct, color='gray', linestyle=':', linewidth=0.7)
    ax2.axvspan(0, 60, alpha=0.05, color='blue')
    ax2.axvspan(60, 100, alpha=0.05, color='orange')
    ax2.set_xlabel('Gait Cycle [%]')
    ax2.set_ylabel('Modularity Q')
    ax2.set_title(f'Modularity Q  (mean={cm_result.modularity_t.mean():.3f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = save_dir / f"community_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Community] 保存 → {fname.name}")
