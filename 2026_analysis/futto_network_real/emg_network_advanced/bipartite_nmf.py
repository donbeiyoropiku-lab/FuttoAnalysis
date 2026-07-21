"""
emg_network_advanced/bipartite_nmf.py
=======================================
④ 2部グラフ（Bipartite Graph）による NMF の再解釈

【仕組み】
  NMF の結果 EMG ≈ W × H を、ネットワーク理論のフォーマットで表現する。

  ノード種別:
    左側ノード: シナジー（Synergy 1, 2, ...）
    右側ノード: 16個の筋肉（R_GM, ..., L_TA）

  エッジ:
    シナジー k → 筋肉 i
    重み = W[i, k]（筋肉 i が シナジー k にどれほど寄与するか）

【なぜ有用か】
  従来の棒グラフでは「1シナジーずつ別々の図」になっていたが、
  2部グラフにすると「どの筋が複数のシナジーを共有しているか」
  = モジュール構造が一目で分かる。

  例: L_SOL が Synergy 1 と Synergy 3 に高い重みを持つ
      → ヒラメ筋は複数の推進・安定化シナジーを担うハブ筋

【入力】
  emg_network/synergy.py の SynergyResult.W  shape (N_muscles, N_synergies)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "futto_common"))

try:
    from emg_network.synergy import SynergyResult, compute_synergy
except ModuleNotFoundError:
    # synergy.py がない環境向けのスタブ
    from dataclasses import dataclass, field as dc_field
    @dataclass
    class SynergyResult:
        task_key: str; phase: int; speed: str
        muscle_names: list; n_synergies: int; vaf_curve: list
        W: 'np.ndarray'; H: 'np.ndarray'; vaf_final: float
        dominant_muscles: list = dc_field(default_factory=list)
    def compute_synergy(*a, **k):
        raise ImportError("emg_network.synergy が見つかりません")

from emg_network_advanced.data_loader import build_normalized_csv_path


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class BipartiteResult:
    task_key      : str
    phase         : int
    speed         : str
    channel_names : list[str]   # 16筋
    n_synergies   : int

    W             : np.ndarray   # shape (N_muscles, N_syn)  筋重み行列
    H             : np.ndarray   # shape (N_syn, T)          活性化時系列
    vaf           : float

    # 2部グラフ隣接情報（可視化用）
    # synergy_labels: ['Syn1', 'Syn2', ...]
    synergy_labels: list[str]

    # 各筋が最も大きく寄与するシナジーのインデックス
    dominant_syn  : np.ndarray   # shape (N_muscles,)

    # 各シナジーの主要筋 Top-3
    top_muscles   : list[list[str]]


# =============================================================================
# NMF → 2部グラフへの変換
# =============================================================================

def build_bipartite(
    syn_result    : SynergyResult,
) -> BipartiteResult:
    """
    SynergyResult から 2部グラフ表現を構築する。

    Parameters
    ----------
    syn_result : emg_network.synergy.compute_synergy() の出力

    Returns
    -------
    BipartiteResult
    """
    W  = syn_result.W           # (N_muscles, N_syn)
    H  = syn_result.H           # (N_syn, T)
    ch = syn_result.muscle_names
    K  = syn_result.n_synergies

    syn_labels    = [f'Syn{k+1}' for k in range(K)]
    dominant_syn  = np.argmax(W, axis=1)   # 各筋の主要シナジー
    top_muscles   = [
        [ch[i] for i in np.argsort(W[:, k])[::-1][:3]]
        for k in range(K)
    ]

    return BipartiteResult(
        task_key      = syn_result.task_key,
        phase         = syn_result.phase,
        speed         = syn_result.speed,
        channel_names = ch,
        n_synergies   = K,
        W             = W,
        H             = H,
        vaf           = syn_result.vaf_final,
        synergy_labels= syn_labels,
        dominant_syn  = dominant_syn,
        top_muscles   = top_muscles,
    )


# =============================================================================
# 保存
# =============================================================================

def save_results(result: BipartiteResult, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result.task_key}_Ph{result.phase}_{result.speed}"

    # W 行列
    pd.DataFrame(
        result.W,
        index   = result.channel_names,
        columns = result.synergy_labels,
    ).to_csv(out_dir / f"bipartite_W_{tag}.csv", float_format='%.4f')

    # 各筋の主要シナジー
    pd.DataFrame({
        'muscle'       : result.channel_names,
        'dominant_syn' : [result.synergy_labels[i] for i in result.dominant_syn],
        'weight'       : [result.W[i, result.dominant_syn[i]]
                          for i in range(len(result.channel_names))],
    }).to_csv(out_dir / f"bipartite_dominant_{tag}.csv", index=False, float_format='%.4f')

    print(f"  [Bipartite] Saved -> {out_dir}")


# =============================================================================
# 可視化
# =============================================================================

def plot_bipartite_graph(
    result    : BipartiteResult,
    threshold : float = 0.15,
    save_path : Optional[str | Path] = None,
) -> None:
    """
    2部グラフを描画する。

    左列  : シナジーノード（Syn1, Syn2, ...）— 黄色
    右列  : 筋肉ノード（R_GM, ..., L_TA）— 青/赤
    エッジ: 重み W[i,k] ≥ threshold のみ描画
    エッジ幅: W[i,k] に比例
    ノードサイズ: シナジー=一定 / 筋肉=その筋の全シナジー重み合計に比例
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    W   = result.W
    K   = result.n_synergies
    N   = len(result.channel_names)
    ch  = result.channel_names

    # シナジーの色（タブカラー）
    syn_colors = plt.cm.Set2(np.linspace(0, 1, K))

    fig, ax = plt.subplots(figsize=(13, max(9, N * 0.55)))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1, max(N, K) + 1)
    ax.axis('off')

    # シナジーノードの Y 座標（左列, x=0.5）
    syn_ys = np.linspace(max(N, K) - 1, 0, K)
    # 筋肉ノードの Y 座標（右列, x=2.5）
    mus_ys = np.linspace(max(N, K) - 1, 0, N)

    W_max = W.max() + 1e-9

    # エッジ
    for k in range(K):
        for i in range(N):
            w = W[i, k]
            if w < threshold:
                continue
            lw    = 0.5 + (w / W_max) * 6.0
            alpha = 0.2 + (w / W_max) * 0.65
            ax.plot([0.5, 2.5], [syn_ys[k], mus_ys[i]],
                    color=syn_colors[k], linewidth=lw, alpha=alpha, zorder=1)

    # シナジーノード（左）
    for k in range(K):
        ax.scatter(0.5, syn_ys[k], s=350, c=[syn_colors[k]],
                   zorder=5, edgecolors='white', linewidths=2.0)
        ax.text(0.5, syn_ys[k], result.synergy_labels[k],
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        # Top3 筋をシナジーラベルの左に表示
        top3_str = ', '.join(result.top_muscles[k][:2])
        ax.text(0.15, syn_ys[k], top3_str,
                ha='right', va='center', fontsize=7, color='gray')

    # 筋肉ノード（右）
    muscle_sum = W.sum(axis=1)   # 各筋の全シナジー合計重み
    ms_max     = muscle_sum.max() + 1e-9
    for i, c in enumerate(ch):
        color = '#3498db' if c.startswith('R_') else '#e74c3c'
        size  = 100 + (muscle_sum[i] / ms_max) * 250
        ax.scatter(2.5, mus_ys[i], s=size, c=color,
                   zorder=5, edgecolors='white', linewidths=1.5)
        ax.text(2.65, mus_ys[i], c, ha='left', va='center',
                fontsize=8, fontweight='bold', color=color)

        # その筋の最大寄与シナジーを小さく表示
        dom_k  = int(result.dominant_syn[i])
        dom_w  = W[i, dom_k]
        ax.text(2.35, mus_ys[i],
                f"{result.synergy_labels[dom_k]}({dom_w:.2f})",
                ha='right', va='center', fontsize=7, color='gray')

    # 列ラベル
    ax.text(0.5, max(N, K) + 0.3, 'Synergies',
            ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    ax.text(2.5, max(N, K) + 0.3, 'Muscles',
            ha='center', fontsize=12, fontweight='bold', color='#2c3e50')

    # 凡例
    handles = (
        [mpatches.Patch(color=syn_colors[k], label=result.synergy_labels[k])
         for k in range(K)]
        + [
            mpatches.Patch(color='#3498db', label='Right leg muscles'),
            mpatches.Patch(color='#e74c3c', label='Left leg muscles'),
        ]
    )
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(1.2, 0.0), fontsize=9, ncol=1)

    ax.set_title(
        f'NMF Bipartite Graph - {result.task_key}\n'
        f'Phase{result.phase}  {result.speed}m/s  '
        f'N_syn={result.n_synergies}  VAF={result.vaf:.3f}  '
        f'(threshold={threshold})',
        fontsize=12, pad=10,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_w_heatmap_and_h_timeseries(
    results   : dict[str, BipartiteResult],
    save_path : Optional[str | Path] = None,
) -> None:
    """
    複数タスクの W 行列（筋重み）と H 時系列（活性化パターン）を
    縦に並べて比較する。

    タスク間でシナジーの構成筋と活性化タイミングがどう変化するかを視覚化。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tasks = list(results.keys())
    n     = len(tasks)
    gait  = np.arange(0, 101)

    # 最大シナジー数
    max_k = max(r.n_synergies for r in results.values())
    ch    = results[tasks[0]].channel_names

    fig, axes = plt.subplots(n * 2, 1, figsize=(max_k * 2 + 4, 4.5 * n),
                              gridspec_kw={'height_ratios': [3, 2] * n})

    for ti, tk in enumerate(tasks):
        r    = results[tk]
        ax_w = axes[ti * 2]
        ax_h = axes[ti * 2 + 1]

        # W ヒートマップ
        W_plot = np.zeros((len(ch), max_k))
        W_plot[:, :r.n_synergies] = r.W
        im = ax_w.imshow(W_plot.T, cmap='YlOrRd', aspect='auto',
                         vmin=0, vmax=W_plot.max() + 1e-9)
        ax_w.set_yticks(range(max_k))
        ax_w.set_yticklabels([f'Syn{k+1}' for k in range(max_k)], fontsize=9)
        ax_w.set_xticks(range(len(ch)))
        ax_w.set_xticklabels(ch, rotation=45, ha='right', fontsize=7)
        ax_w.axvline(7.5, color='k', linewidth=1)   # 左右境界
        ax_w.set_title(
            f'{tk}  N_syn={r.n_synergies}  VAF={r.vaf:.3f}',
            fontsize=11,
        )
        plt.colorbar(im, ax=ax_w, label='Weight', shrink=0.8)

        # H 時系列
        colors_h = plt.cm.Set1(np.linspace(0, 1, max_k))
        for k in range(r.n_synergies):
            ax_h.plot(gait, r.H[k], color=colors_h[k],
                      linewidth=2.0, label=f'Syn{k+1}')
        ax_h.axvspan(0, 60, alpha=0.05, color='blue')
        ax_h.axvspan(60, 100, alpha=0.05, color='orange')
        ax_h.set_xlim(0, 100)
        ax_h.set_ylabel('Activation')
        ax_h.legend(loc='upper right', fontsize=8, ncol=min(r.n_synergies, 4))
        ax_h.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Gait Cycle [%]')
    fig.suptitle(
        f'NMF Synergy: W matrix & H timeseries\n'
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
