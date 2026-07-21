"""
visualizer/plot_multilayer.py
==============================
多層ネットワーク統合解析・タスク比較の可視化。

出力:
  1. 多層ネットワーク構造図（3層を縦に並べた模式図）
  2. 速度×タスク比較レーダーチャート
  3. 速度×タスク比較折れ線グラフ（箱ひげ図）
  4. 対側性効果サマリー棒グラフ
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG
from multilayer_network.fusion import MultilayerResult

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
TASK_MARKERS = {'task01': 'o', 'task02': 's', 'task03': '^'}
SPEEDS = ['0.7', '0.9', '1.1', '1.3', '1.5']
SPEED_VALS = [float(s) for s in SPEEDS]


# =============================================================================
# 1. 多層ネットワーク構造図
# =============================================================================

def plot_multilayer_structure(
    ml       : MultilayerResult,
    save_dir : Path,
    task_key : str,
    phase    : int,
    speed    : str,
) -> None:
    """
    上層=EMG、中層=関節、下層=Futto の3層構造を模式的に可視化する。
    ノード間のインタ層エッジ（PJ / JE）も描画。
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.3, 3.3)
    ax.axis('off')

    # ── 層の Y 座標 ────────────────────────────────────────────
    Y = {'physical': 0.0, 'joint': 1.5, 'emg': 3.0}
    layer_colors = {'physical': '#3498db', 'joint': '#2ecc71', 'emg': '#e74c3c'}
    layer_labels = {'physical': 'Physical Layer (Futto)',
                    'joint'   : 'Joint Layer',
                    'emg'     : 'EMG Layer'}

    # ── 各層の背景帯 ────────────────────────────────────────────
    for layer, y in Y.items():
        ax.axhspan(y - 0.35, y + 0.35, alpha=0.08,
                   color=layer_colors[layer])
        ax.text(-0.45, y, layer_labels[layer],
                va='center', fontsize=11, fontweight='bold',
                color=layer_colors[layer])

    # ── Physical ノード（Futto マーカー） ──────────────────────
    supra = ml.supra_adjacency
    pr    = ml.multilayer_pagerank
    N_P   = ml.N_physical
    N_J   = ml.N_joint
    N_E   = ml.N_emg

    pr_max = pr.max() if pr.max() > 0 else 1.0

    def _place_nodes(n, y_base, offset_x=0.0, spread=1.0):
        """n個のノードをy_baseに均等配置し、x座標リストを返す"""
        if n == 0:
            return []
        xs = np.linspace(offset_x, offset_x + spread, n)
        return list(xs)

    xs_P = _place_nodes(N_P, Y['physical'])
    xs_J = _place_nodes(N_J, Y['joint'],   offset_x=0.25, spread=0.5)
    xs_E = _place_nodes(N_E, Y['emg'])

    # Physical ノード
    for i, x in enumerate(xs_P):
        size   = 80 + pr[i] / pr_max * 300
        col    = plt.cm.Blues(0.4 + pr[i] / pr_max * 0.6)
        ax.scatter(x, Y['physical'], s=size, c=[col], zorder=5,
                   edgecolors='navy', linewidths=0.8)

    # Joint ノード
    j_names = ['Hip', 'Knee', 'Ankle']
    for i, x in enumerate(xs_J):
        idx  = N_P + i
        size = 120 + pr[idx] / pr_max * 400
        col  = plt.cm.Greens(0.5 + pr[idx] / pr_max * 0.5)
        ax.scatter(x, Y['joint'], s=size, c=[col], zorder=5,
                   edgecolors='darkgreen', linewidths=1.0)
        ax.text(x, Y['joint'] + 0.25, j_names[i],
                ha='center', fontsize=9, color='darkgreen')

    # EMG ノード
    muscles = CFG.MUSCLE_NAMES
    for i, x in enumerate(xs_E):
        idx  = N_P + N_J + i
        size = 60 + pr[idx] / pr_max * 250
        side = 'R' if i < 8 else 'L'
        col  = '#3498db' if side == 'R' else '#e74c3c'
        ax.scatter(x, Y['emg'], s=size, c=col, zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.8)

    # ── インタ層エッジ（PJ / JE 強いもの上位のみ描画） ──────────
    # PJ ブロック: supra[0:N_P, N_P:N_P+N_J]
    pj_block = supra[:N_P, N_P:N_P+N_J]
    pj_max   = pj_block.max() if pj_block.max() > 0 else 1.0
    for pi in range(N_P):
        for ji in range(N_J):
            w = pj_block[pi, ji]
            if w < pj_max * 0.3:
                continue
            if pi >= len(xs_P) or ji >= len(xs_J):
                continue
            alpha = 0.2 + w / pj_max * 0.6
            lw    = 0.5 + w / pj_max * 2.0
            ax.plot([xs_P[pi], xs_J[ji]],
                    [Y['physical'], Y['joint']],
                    color='#3498db', linewidth=lw, alpha=alpha, zorder=2)

    # JE ブロック
    je_block = supra[N_P:N_P+N_J, N_P+N_J:]
    je_max   = je_block.max() if je_block.max() > 0 else 1.0
    for ji in range(N_J):
        for ei in range(N_E):
            w = je_block[ji, ei]
            if w < je_max * 0.3:
                continue
            if ji >= len(xs_J) or ei >= len(xs_E):
                continue
            alpha = 0.2 + w / je_max * 0.5
            lw    = 0.5 + w / je_max * 1.5
            ax.plot([xs_J[ji], xs_E[ei]],
                    [Y['joint'], Y['emg']],
                    color='#2ecc71', linewidth=lw, alpha=alpha, zorder=2)

    # ── 指標テキスト ────────────────────────────────────────────
    info = (
        f"GES = {ml.gait_efficiency_score:.4f}\n"
        f"PJ coupling = {ml.coupling_PJ:.3f}\n"
        f"JE coupling = {ml.coupling_JE:.3f}\n"
        f"PE coupling = {ml.coupling_PE:.3f}"
    )
    ax.text(1.45, 1.5, info, va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_title(
        f'Multilayer Network Structure\n'
        f'{CFG.TASK_TITLES.get(task_key, task_key)} — Phase{phase} {speed}m/s',
        fontsize=13,
    )

    fname = save_dir / f"multilayer_structure_{task_key}_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [MultiLayer] 保存 → {fname.name}")


# =============================================================================
# 2. レーダーチャート（タスク比較）
# =============================================================================

def plot_radar_comparison(
    summary_by_task : dict[str, dict],
    save_dir        : Path,
    phase           : int,
    speed           : str,
) -> None:
    """
    各タスクのネットワーク指標をレーダーチャートで比較。

    Parameters
    ----------
    summary_by_task : {task_key: summary_dict}  run_single の返り値
    """
    metrics = {
        'E_norm'       : 'Physical\nEfficiency',
        'λ_norm'       : 'System\nStiffness',
        'EMG_E_global' : 'EMG\nEfficiency',
        'IL_coupling'  : 'Interlimb\nCoupling',
        'cross_ipsi'   : 'Contralateral\nEffect',
        'GES'          : 'Gait\nEfficiency',
    }
    labels = list(metrics.values())
    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    for tk, s in summary_by_task.items():
        vals = [
            min(s.get('E_norm_mean', 0) * 5, 1),           # 0〜1 に正規化
            min(s.get('lambda_max_norm_mean', 0) * 3, 1),
            min(s.get('emg_global_efficiency', 0), 1),
            min(s.get('joint_il_proxy', 0), 1),
            min((s.get('emg_cross_ipsi_ratio', 1) - 0.8) * 5, 1),
            min(s.get('gait_efficiency_score', 0) * 2, 1),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2,
                color=TASK_COLORS.get(tk, 'gray'),
                label=CFG.TASK_TITLES.get(tk, tk))
        ax.fill(angles, vals, alpha=0.15,
                color=TASK_COLORS.get(tk, 'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.set_title(f'Task Comparison Radar\nPhase{phase} {speed}m/s',
                 fontsize=13, pad=20)

    fname = save_dir / f"radar_comparison_ph{phase}_{speed}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Radar] 保存 → {fname.name}")


# =============================================================================
# 3. 速度 × タスク 比較折れ線グラフ
# =============================================================================

def plot_speed_task_comparison(
    all_summaries : list[dict],
    save_dir      : Path,
) -> None:
    """
    全タスク・全速度のサマリーから、速度変化に対する各指標の推移を描画。

    横軸: 歩行速度 (0.7〜1.5 m/s)
    縦軸: 各ネットワーク指標
    """
    # 指標定義 (key, 表示名)
    metrics = [
        ('E_norm_mean',           'Physical Network Efficiency (E_norm)'),
        ('lambda_max_norm_mean',  'System Stiffness (λ_max norm)'),
        ('emg_global_efficiency', 'EMG Global Efficiency'),
        ('emg_cross_ipsi_ratio',  'Contralateral Effect (cross/ipsi)'),
        ('joint_il_proxy',        'Interlimb Joint Coupling'),
        ('gait_efficiency_score', 'Gait Efficiency Score (GES)'),
    ]

    tasks = sorted({s['task_key'] for s in all_summaries})
    n_met = len(metrics)
    ncols = 2
    nrows = (n_met + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    for ax_i, (key, title) in enumerate(metrics):
        ax = axes[ax_i]
        for tk in tasks:
            # フェーズ内平均（全フェーズ集約）
            by_speed: dict[str, list[float]] = {}
            for s in all_summaries:
                if s['task_key'] != tk:
                    continue
                spd = s.get('speed', '')
                val = s.get(key)
                if val is not None:
                    by_speed.setdefault(spd, []).append(float(val))

            spd_vals, means, sems = [], [], []
            for spd in SPEEDS:
                if spd in by_speed and by_speed[spd]:
                    arr = by_speed[spd]
                    spd_vals.append(float(spd))
                    means.append(np.mean(arr))
                    sems.append(np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0)

            if not spd_vals:
                continue
            ax.errorbar(spd_vals, means, yerr=sems,
                        color=TASK_COLORS.get(tk, 'gray'),
                        marker=TASK_MARKERS.get(tk, 'o'),
                        linewidth=2, markersize=7, capsize=4,
                        label=CFG.TASK_TITLES.get(tk, tk))

        ax.set_xlabel('Walking Speed [m/s]', fontsize=10)
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(SPEED_VALS)

    # 余りの軸を非表示
    for ax_i in range(n_met, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Speed × Task Comparison — All Phases',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fname = save_dir / "speed_task_comparison.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Speed×Task] 保存 → {fname.name}")


# =============================================================================
# 4. 対側性効果サマリー棒グラフ（task01 vs task03）
# =============================================================================

def plot_contralateral_effect(
    all_summaries : list[dict],
    save_dir      : Path,
) -> None:
    """
    task01（Futto着用）と task03（非着用）の差分を
    速度ごとに棒グラフで表示する。
    """
    metrics = [
        ('gait_efficiency_score', 'ΔGES'),
        ('emg_global_efficiency', 'ΔEMG Global Eff'),
        ('emg_cross_ipsi_ratio',  'ΔCross/Ipsi ratio'),
        ('joint_il_proxy',        'ΔInterlimb Coupling'),
    ]

    speeds = SPEEDS
    t01 = {s['speed']: s for s in all_summaries if s['task_key'] == 'task01'}
    t03 = {s['speed']: s for s in all_summaries if s['task_key'] == 'task03'}

    common_speeds = [sp for sp in speeds if sp in t01 and sp in t03]
    if not common_speeds:
        print("  [警告] task01 と task03 の共通速度データがありません")
        return

    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 5))
    if n_met == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        deltas = []
        for sp in common_speeds:
            v1 = t01[sp].get(key, 0)
            v3 = t03[sp].get(key, 0)
            deltas.append(float(v1) - float(v3) if v1 is not None and v3 is not None else 0)

        colors = ['#e74c3c' if d > 0 else '#95a5a6' for d in deltas]
        bars = ax.bar(common_speeds, deltas, color=colors, edgecolor='white',
                      linewidth=1.2, width=0.6)

        # ゼロライン
        ax.axhline(0, color='black', linewidth=1.0)

        # 値のラベル
        for bar, d in zip(bars, deltas):
            va  = 'bottom' if d >= 0 else 'top'
            y   = bar.get_height() if d >= 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (0.002 if d >= 0 else -0.002),
                    f'{d:+.3f}', ha='center', va=va, fontsize=8)

        ax.set_xlabel('Walking Speed [m/s]')
        ax.set_ylabel(label)
        ax.set_title(f'{label}\n(Task01 − Task03)')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Contralateral Effect of Futto\n(task01 vs task03)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fname = save_dir / "contralateral_effect_summary.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Contralateral] 保存 → {fname.name}")


# =============================================================================
# 5. 統計比較テーブル（t検定）
# =============================================================================

def compute_stats_table(
    all_summaries : list[dict],
    save_dir      : Path,
) -> None:
    """
    タスク間の各指標について対応なし t 検定と Cohen's d を計算し CSV に保存する。
    """
    import pandas as pd

    metrics = [
        'E_norm_mean', 'lambda_max_norm_mean',
        'emg_global_efficiency', 'emg_cross_ipsi_ratio',
        'joint_il_proxy', 'gait_efficiency_score',
    ]
    task_pairs = [('task01', 'task03'), ('task02', 'task03'), ('task01', 'task02')]

    rows = []
    for (ta, tb) in task_pairs:
        vals_a = {m: [s[m] for s in all_summaries
                      if s['task_key'] == ta and m in s and s[m] is not None]
                  for m in metrics}
        vals_b = {m: [s[m] for s in all_summaries
                      if s['task_key'] == tb and m in s and s[m] is not None]
                  for m in metrics}

        for m in metrics:
            a = np.array(vals_a[m], dtype=float)
            b = np.array(vals_b[m], dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue

            t_stat, p_val = stats.ttest_ind(a, b)
            # Cohen's d
            pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
            cohens_d   = (a.mean() - b.mean()) / (pooled_std + 1e-12)

            rows.append({
                'comparison'  : f'{ta} vs {tb}',
                'metric'      : m,
                'mean_A'      : round(a.mean(), 4),
                'mean_B'      : round(b.mean(), 4),
                'delta'       : round(a.mean() - b.mean(), 4),
                't_stat'      : round(t_stat, 3),
                'p_value'     : round(p_val, 4),
                'sig'         : '***' if p_val < 0.001 else ('**' if p_val < 0.01
                                 else ('*' if p_val < 0.05 else 'n.s.')),
                'cohens_d'    : round(cohens_d, 3),
            })

    if not rows:
        print("  [統計] データ不足のため統計テーブルをスキップしました")
        return

    df = pd.DataFrame(rows)
    fname = save_dir / "stats_comparison_table.csv"
    df.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f"  [Stats] 保存 → {fname.name}")

    # 表示用サマリー
    print("\n  ── 統計比較サマリー ──")
    print(df[['comparison', 'metric', 'delta', 'p_value', 'sig', 'cohens_d']].to_string(index=False))
