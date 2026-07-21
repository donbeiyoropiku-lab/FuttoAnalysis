"""
emg_network_advanced/graphical_lasso.py
=========================================
① Graphical Lasso による偏相関ネットワーク

【なぜ単純な相関では不十分か】
  例: ILIO → RF → TA という連鎖で活動した場合、
  Pearson 相関では ILIO と TA の間にも強い相関が出る（疑似相関）。
  Graphical Lasso は「他の全筋の影響を引いた上での直接的な関係」
  = 偏相関 を推定し、疑似相関エッジを数理的にゼロにする。

  結果の解釈:
    edge(i,j) > 0  → 2筋が直接的な協調関係を持つ
    edge(i,j) = 0  → 他の筋を介した間接的な関係のみ

【アルゴリズム】
  精度行列（逆共分散行列）Θ を L1 正則化付きで推定する。
    最大化: log det Θ - tr(S Θ) - α ||Θ||₁
  α（alpha）が大きいほどスパースなネットワークになる。

【実装】
  sklearn.covariance.GraphicalLassoCV で alpha を自動選択、
  または GraphicalLasso で alpha を手動指定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class GraphicalLassoResult:
    task_key      : str
    phase         : int
    speed         : str
    channel_names : list[str]
    alpha         : float

    # 偏相関行列  shape (N, N)  —  直接的な筋間結合の強さ
    partial_corr  : np.ndarray

    # スパース精度行列  shape (N, N)
    precision     : np.ndarray

    # 隣接行列（閾値処理済み）  shape (N, N)
    adjacency     : np.ndarray

    # ネットワーク指標  shape (N,)
    degree        : np.ndarray   # 重み付き次数
    hub_score     : np.ndarray   # 次数を正規化したハブスコア

    # タスク間比較用スカラー
    n_edges       : int          # ゼロでないエッジ数
    mean_abs_pcor : float        # 平均偏相関絶対値


# =============================================================================
# Graphical Lasso の実行
# =============================================================================

def compute_graphical_lasso(
    emg          : np.ndarray,          # shape (N_ch, T_samples)
    channel_names: list[str],
    task_key     : str  = "",
    phase        : int  = 0,
    speed        : str  = "",
    alpha        : Optional[float] = None,   # None = CV で自動決定
    alpha_range  : tuple = (0.01, 0.5),
    max_iter     : int  = 1000,
) -> GraphicalLassoResult:
    """
    Graphical Lasso で偏相関ネットワークを推定する。

    Parameters
    ----------
    emg          : shape (N_ch, T)  正規化済み EMG（%Peak, 0〜1）
    channel_names: 筋肉名リスト
    alpha        : L1 正則化強度。None なら CV で自動選択（推奨）
    alpha_range  : CV 探索範囲

    Returns
    -------
    GraphicalLassoResult
    """
    from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

    N, T = emg.shape
    X    = emg.T   # (T, N)  sklearn は (samples, features) を期待

    print(f"[GraphicalLasso] {task_key} Ph{phase}  N={N}, T={T}")

    # alpha の自動決定 or 手動指定
    if alpha is None:
        print("  alpha を交差検証で自動決定中...")
        model = GraphicalLassoCV(
            alphas    = 10,
            n_refinements = 4,
            cv        = 5,
            max_iter  = max_iter,
            tol       = 1e-4,
            n_jobs    = -1,
        )
        model.fit(X)
        alpha = float(model.alpha_)
        print(f"  最適 alpha = {alpha:.4f}")
    else:
        model = GraphicalLasso(
            alpha    = alpha,
            max_iter = max_iter,
            tol      = 1e-4,
        )
        model.fit(X)

    # 精度行列 → 偏相関行列に変換
    precision    = model.precision_                  # (N, N)
    partial_corr = _precision_to_partial_corr(precision)

    # 隣接行列（対角をゼロ化）
    adjacency = partial_corr.copy()
    np.fill_diagonal(adjacency, 0)

    # ネットワーク指標
    degree    = np.abs(adjacency).sum(axis=1)
    hub_score = degree / (degree.max() + 1e-12)

    off_diag      = adjacency[~np.eye(N, dtype=bool)]
    n_edges       = int((np.abs(off_diag) > 1e-9).sum()) // 2
    mean_abs_pcor = float(np.abs(off_diag).mean())

    print(f"  edges={n_edges}  mean|pcor|={mean_abs_pcor:.4f}")

    return GraphicalLassoResult(
        task_key      = task_key,
        phase         = phase,
        speed         = speed,
        channel_names = channel_names,
        alpha         = alpha,
        partial_corr  = partial_corr,
        precision     = precision,
        adjacency     = adjacency,
        degree        = degree,
        hub_score     = hub_score,
        n_edges       = n_edges,
        mean_abs_pcor = mean_abs_pcor,
    )


def _precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """
    精度行列 Θ から偏相関行列 P に変換する。

    P_ij = -Θ_ij / sqrt(Θ_ii * Θ_jj)

    P_ij ∈ [-1, 1]
      正値: 他筋の影響を除いた上での正の直接協調
      負値: 他筋の影響を除いた上での拮抗関係
    """
    N    = precision.shape[0]
    diag = np.sqrt(np.diag(precision))
    P    = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                P[i, j] = -precision[i, j] / (diag[i] * diag[j] + 1e-12)
    np.fill_diagonal(P, 1.0)
    return np.clip(P, -1.0, 1.0)


# =============================================================================
# タスク間比較
# =============================================================================

def compare_tasks(
    results: dict[str, GraphicalLassoResult],
) -> pd.DataFrame:
    """
    複数タスクの Graphical Lasso 結果を比較する DataFrame を返す。

    出力列: task | n_edges | mean_abs_pcor | alpha | top_hub
    """
    rows = []
    for tk, r in results.items():
        top_hub_idx = int(np.argmax(r.hub_score))
        rows.append({
            'task'         : tk,
            'n_edges'      : r.n_edges,
            'mean_abs_pcor': round(r.mean_abs_pcor, 4),
            'alpha'        : round(r.alpha, 4),
            'top_hub'      : r.channel_names[top_hub_idx],
        })
    return pd.DataFrame(rows)


def edge_difference(
    r1: GraphicalLassoResult,
    r2: GraphicalLassoResult,
) -> np.ndarray:
    """
    2タスク間の偏相関行列の差分を返す。

    result = r1.partial_corr - r2.partial_corr
    正値: r1 で強いエッジ、負値: r2 で強いエッジ
    """
    return r1.partial_corr - r2.partial_corr


# =============================================================================
# 保存
# =============================================================================

def save_results(
    result  : GraphicalLassoResult,
    out_dir : str | Path,
) -> None:
    """偏相関行列と次数を CSV に保存する。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch = result.channel_names

    pd.DataFrame(result.partial_corr, index=ch, columns=ch).to_csv(
        out_dir / f"glasso_partial_corr_{result.task_key}_Ph{result.phase}_{result.speed}.csv",
        float_format='%.6f'
    )
    pd.DataFrame({
        'muscle' : ch,
        'degree' : result.degree,
        'hub_score': result.hub_score,
    }).to_csv(
        out_dir / f"glasso_degree_{result.task_key}_Ph{result.phase}_{result.speed}.csv",
        index=False, float_format='%.4f'
    )
    print(f"  [GLasso] Saved -> {out_dir}")


# =============================================================================
# 可視化
# =============================================================================

def plot_partial_corr_heatmap(
    results   : dict[str, GraphicalLassoResult],
    save_path : Optional[str | Path] = None,
) -> None:
    """
    複数タスクの偏相関行列をヒートマップで横並び比較する。

    task01 vs task02 vs task03 の直接的な筋間結合パターンの違いを視覚化。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tasks   = list(results.keys())
    n       = len(tasks)
    ch      = results[tasks[0]].channel_names
    N       = len(ch)

    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, tk in zip(axes, tasks):
        r  = results[tk]
        im = ax.imshow(r.partial_corr, cmap='RdBu_r',
                       vmin=-0.8, vmax=0.8, aspect='equal')
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(ch, rotation=90, fontsize=7)
        ax.set_yticklabels(ch, fontsize=7)
        ax.axhline(7.5, color='k', linewidth=1.5)
        ax.axvline(7.5, color='k', linewidth=1.5)
        ax.set_title(
            f"{tk}\nalpha={r.alpha:.3f}  edges={r.n_edges}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, label='Partial correlation', shrink=0.8)

    fig.suptitle(
        f'Graphical Lasso: Partial Correlation Network\n'
        f'Phase{results[tasks[0]].phase}  {results[tasks[0]].speed}m/s',
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_network_graph(
    result    : GraphicalLassoResult,
    threshold : float = 0.1,
    save_path : Optional[str | Path] = None,
) -> None:
    """
    偏相関ネットワークをサーキュラーグラフで描画する。

    右脚（青・上半円）/ 左脚（赤・下半円）
    エッジ色: 正の偏相関=青、負の偏相関（拮抗）=赤
    エッジ幅: |偏相関| に比例
    ノードサイズ: Degree に比例
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ch  = result.channel_names
    N   = len(ch)
    adj = result.partial_corr.copy()
    np.fill_diagonal(adj, 0)

    # 円周配置（右=上半円、左=下半円）
    angles = np.zeros(N)
    right_idx = [i for i, c in enumerate(ch) if c.startswith('R_')]
    left_idx  = [i for i, c in enumerate(ch) if c.startswith('L_')]
    for rank, idx in enumerate(right_idx):
        angles[idx] = np.pi - rank * np.pi / max(len(right_idx) - 1, 1)
    for rank, idx in enumerate(left_idx):
        angles[idx] = -rank * np.pi / max(len(left_idx) - 1, 1)

    xs = np.cos(angles)
    ys = np.sin(angles)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # エッジ
    for i in range(N):
        for j in range(i + 1, N):
            pcor = adj[i, j]
            if abs(pcor) < threshold:
                continue
            lw    = abs(pcor) * 5.0
            color = '#2980b9' if pcor > 0 else '#e74c3c'
            alpha = 0.2 + abs(pcor) * 0.6
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color=color, linewidth=lw, alpha=alpha, zorder=1)

    # ノード
    deg_max = result.degree.max() + 1e-9
    for i, c in enumerate(ch):
        color  = '#3498db' if c.startswith('R_') else '#e74c3c'
        size   = 80 + (result.degree[i] / deg_max) * 350
        ax.scatter(xs[i], ys[i], s=size, c=color, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ha = 'left' if xs[i] >= 0 else 'right'
        ax.text(1.16 * xs[i], 1.16 * ys[i],
                c.replace('_', '\n'),
                ha=ha, va='center', fontsize=8,
                fontweight='bold', color=color)

    # 中央テキスト
    ax.text(0, 0,
            f"edges={result.n_edges}\n"
            f"alpha={result.alpha:.3f}\n"
            f"|pcor|={result.mean_abs_pcor:.3f}",
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    # 凡例
    handles = [
        mpatches.Patch(color='#3498db', label='Right leg'),
        mpatches.Patch(color='#e74c3c', label='Left leg'),
        plt.Line2D([0], [0], color='#2980b9', lw=2, label='Positive pcor (synergy)'),
        plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Negative pcor (antagonism)'),
    ]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=9)
    ax.set_title(
        f'Graphical Lasso Network - {result.task_key}\n'
        f'Phase{result.phase}  {result.speed}m/s  '
        f'(threshold={threshold})',
        fontsize=12, pad=15,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)
