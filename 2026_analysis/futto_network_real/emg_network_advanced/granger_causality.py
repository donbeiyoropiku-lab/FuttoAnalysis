"""
emg_network_advanced/granger_causality.py
==========================================
② グランジャー因果性（Granger Causality）による有向筋ネットワーク

【仕組み】
  「筋 A の過去の波形を使うと、筋 B の現在の活動を
   B 自身の過去データだけで予測するよりも正確に予測できるか？」
  を統計的に検定する（F 検定）。

  A → B が有意: 筋 A が筋 B の活動を「引き起こしている」
                = 運動制御において A が主導権を持つ

  出力されるエッジ:
    A → B （有向グラフ）
    例: 「立脚初期に GM → SOL」= 大殿筋の活動が下腿ヒラメ筋を制御

【パラメータ】
  maxlag  : 検定に用いる最大遅延 [サンプル数]
            EMG 2000Hz で lag=20 → 10ms 以内の因果を検出
  p_thresh: 有意水準（デフォルト 0.05）

【計算量】
  16筋 × 15筋の相手 = 240 ペア
  各ペアで VAR モデルを lag ×2 回フィット → CPU で数十秒
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class GrangerResult:
    task_key      : str
    phase         : int
    speed         : str
    channel_names : list[str]
    maxlag        : int
    p_thresh      : float

    # p 値行列  shape (N, N)
    # p_matrix[i, j] = 「筋 i が筋 j に Granger 因果を持つ」の p 値
    p_matrix      : np.ndarray

    # F 統計量行列  shape (N, N)
    f_matrix      : np.ndarray

    # 有向隣接行列  shape (N, N)  1 = 有意な因果あり
    adjacency     : np.ndarray

    # 有向次数（out-degree: 他の筋を制御している度合い）
    out_degree    : np.ndarray   # shape (N,)

    # 有向次数（in-degree: 他の筋から制御されている度合い）
    in_degree     : np.ndarray   # shape (N,)

    # ハブ: out_degree が最も高い筋（「主導役」）
    @property
    def hub_muscle(self) -> str:
        return self.channel_names[int(np.argmax(self.out_degree))]

    @property
    def n_significant_edges(self) -> int:
        return int(self.adjacency.sum())


# =============================================================================
# グランジャー因果性の計算
# =============================================================================

def compute_granger_causality(
    emg          : np.ndarray,
    channel_names: list[str],
    task_key     : str  = "",
    phase        : int  = 0,
    speed        : str  = "",
    fs_original  : int  = 2000,
    fs_target    : int  = 200,
    maxlag_ms      : float = 100.0,
    use_gait_cycle : bool  = True,   # ★ True推奨: 歩行周期101点で非定常問題を回避
    p_thresh     : float = 0.05,
    verbose      : bool  = True,
) -> GrangerResult:
    """
    全筋ペアのグランジャー因果性を計算する。

    【ダウンサンプリングによる高速化】
    前処理済み EMG は 10Hz ローパスフィルタ後の包絡線である。
    ナイキスト定理より fs ≥ 20Hz で情報は完全保持されるため、
    2000Hz → 200Hz にダウンサンプリングしてもデータの損失はない。
    サンプル数が 1/10 になることで計算量は約 1/100 に削減される。

    Parameters
    ----------
    emg          : shape (N_ch, T)  正規化済み EMG（%Peak, 0〜1）
    channel_names: 筋肉名リスト
    fs_original  : 入力データのサンプリング周波数 [Hz]（デフォルト 2000）
    fs_target    : ダウンサンプリング後の周波数 [Hz]（デフォルト 200）
                   10Hz 包絡線に対してナイキスト周波数の 10 倍なので十分
    maxlag_ms    : Granger 検定に用いる最大遅延 [ms]（デフォルト 100ms）
                   fs_target でのラグ数に自動変換される
    p_thresh     : 有意水準
    verbose      : 進捗を表示するか

    Returns
    -------
    GrangerResult
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    from joblib import Parallel, delayed

    # ── データ準備（2モード） ─────────────────────────────────────
    if use_gait_cycle:
        # 【推奨】歩行周期平均波形モード (101点)
        # 連続60秒データは歩行リズムの繰り返しで「非定常」であるため
        # 全ペアが p<0.05 になる過検出が発生する。
        # 101点の代表波形は定常性を仮定でき、筋間の時間的先行関係を正確に捉える。
        T_total = emg.shape[1]
        indices = np.linspace(0, T_total - 1, 101, dtype=int)
        emg_ds  = emg[:, indices]      # shape (N_ch, 101)
        maxlag  = 5                     # 歩行周期の5% ≈ 25ms
        mode_str = "gait-cycle 101pts  maxlag=5(5%cycle)"
    else:
        # 連続データ + ダウンサンプリングモード
        ratio = max(1, int(fs_original / fs_target))
        if ratio > 1:
            from scipy.signal import decimate
            emg_ds = np.zeros((emg.shape[0], (emg.shape[1] - 1) // ratio + 1))
            for ch_i in range(emg.shape[0]):
                emg_ds[ch_i] = decimate(emg[ch_i], ratio, ftype='fir', zero_phase=True)
            fs_used = fs_original / ratio
        else:
            emg_ds  = emg
            fs_used = fs_original
        maxlag   = max(5, min(20, int(maxlag_ms / 1000.0 * fs_used)))
        mode_str = f"continuous {fs_used:.0f}Hz  maxlag={maxlag}"

    N, T   = emg_ds.shape
    n_pairs = N * (N - 1)
    print(f"[Granger] {task_key} Ph{phase}  {mode_str}  T={T}  {n_pairs} pairs")

    # ── 1ペア計算の内部関数（joblib で並列化） ──────────────────
    def _compute_pair(i: int, j: int):
        data = np.column_stack([emg_ds[j], emg_ds[i]])
        try:
            import warnings
            with np.errstate(invalid='ignore', divide='ignore'),                  warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_vals = [gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
            f_vals = [gc_res[lag][0]['ssr_ftest'][0] for lag in range(1, maxlag + 1)]
            best = int(np.argmin(p_vals))
            return i, j, float(p_vals[best]), float(f_vals[best])
        except Exception:
            return i, j, 1.0, 0.0

    pairs   = [(i, j) for i in range(N) for j in range(N) if i != j]
    results_list = Parallel(n_jobs=-1, verbose=5 if verbose else 0)(
        delayed(_compute_pair)(i, j) for i, j in pairs
    )

    p_mat = np.ones((N, N))
    f_mat = np.zeros((N, N))
    for i, j, p_val, f_val in results_list:
        p_mat[i, j] = p_val
        f_mat[i, j] = f_val

    # 有向隣接行列（有意なペアのみ 1）
    adjacency = (p_mat < p_thresh).astype(float)
    np.fill_diagonal(adjacency, 0)

    out_deg = adjacency.sum(axis=1)   # i から出るエッジ数
    in_deg  = adjacency.sum(axis=0)   # j に入るエッジ数

    n_sig = int(adjacency.sum())
    print(f"  Significant edges: {n_sig} / {n_pairs}")
    print(f"  Hub (max out-degree): {channel_names[int(np.argmax(out_deg))]}"
          f"  (out={int(out_deg.max())})")

    return GrangerResult(
        task_key      = task_key,
        phase         = phase,
        speed         = speed,
        channel_names = channel_names,
        maxlag        = maxlag,
        p_thresh      = p_thresh,
        p_matrix      = p_mat,
        f_matrix      = f_mat,
        adjacency     = adjacency,
        out_degree    = out_deg,
        in_degree     = in_deg,
    )


# =============================================================================
# タスク間比較・保存・可視化
# =============================================================================

def compare_tasks(results: dict[str, GrangerResult]) -> pd.DataFrame:
    """複数タスクの Granger 結果を比較する DataFrame を返す。"""
    rows = []
    for tk, r in results.items():
        rows.append({
            'task'          : tk,
            'n_edges'       : r.n_significant_edges,
            'hub_muscle'    : r.hub_muscle,
            'max_out_degree': int(r.out_degree.max()),
            'max_in_degree' : int(r.in_degree.max()),
        })
    return pd.DataFrame(rows)


def save_results(result: GrangerResult, out_dir: str | Path) -> None:
    """p 値・F 値・隣接行列を CSV に保存する。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch  = result.channel_names
    tag = f"{result.task_key}_Ph{result.phase}_{result.speed}"

    pd.DataFrame(result.p_matrix,   index=ch, columns=ch).to_csv(
        out_dir / f"granger_pval_{tag}.csv", float_format='%.4f')
    pd.DataFrame(result.adjacency,  index=ch, columns=ch).to_csv(
        out_dir / f"granger_adjacency_{tag}.csv")
    pd.DataFrame({
        'muscle'    : ch,
        'out_degree': result.out_degree,
        'in_degree' : result.in_degree,
    }).to_csv(out_dir / f"granger_degree_{tag}.csv", index=False)
    print(f"  [Granger] Saved -> {out_dir}")


def plot_directed_network(
    result    : GrangerResult,
    save_path : Optional[str | Path] = None,
    layout    : str = 'circular',
) -> None:
    """
    グランジャー因果性の有向ネットワークを描画する。

    エッジ: 矢印（A → B = A が B を Granger 因果）
    ノード色: 右脚=青 / 左脚=赤
    ノードサイズ: out_degree（主導役ほど大きい）に比例
    矢印の太さ: F 統計量に比例（因果の強さ）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ch  = result.channel_names
    N   = len(ch)
    adj = result.adjacency
    f   = result.f_matrix

    # 円周配置
    angles    = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # 右脚を上半円、左脚を下半円に配置
    right_idx = [i for i, c in enumerate(ch) if c.startswith('R_')]
    left_idx  = [i for i, c in enumerate(ch) if c.startswith('L_')]
    angles_arr = np.zeros(N)
    for rank, idx in enumerate(right_idx):
        angles_arr[idx] = np.pi - rank * np.pi / max(len(right_idx) - 1, 1)
    for rank, idx in enumerate(left_idx):
        angles_arr[idx] = -rank * np.pi / max(len(left_idx) - 1, 1)

    xs = np.cos(angles_arr)
    ys = np.sin(angles_arr)

    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    f_max = f.max() + 1e-9

    # 有向エッジ（矢印）
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 0 or i == j:
                continue
            lw    = 0.5 + (f[i, j] / f_max) * 3.0
            # 矢印の向き: i → j
            dx = xs[j] - xs[i]
            dy = ys[j] - ys[i]
            # 矢印をノードの少し手前で終わる
            shrink = 0.12
            ax.annotate(
                '', xy=(xs[j] - dx * shrink, ys[j] - dy * shrink),
                xytext=(xs[i] + dx * shrink, ys[i] + dy * shrink),
                arrowprops=dict(
                    arrowstyle='->', color='#7f8c8d',
                    lw=lw, alpha=0.6,
                    mutation_scale=12,
                )
            )

    # ノード
    out_max = result.out_degree.max() + 1e-9
    for i, c in enumerate(ch):
        color = '#3498db' if c.startswith('R_') else '#e74c3c'
        size  = 100 + (result.out_degree[i] / out_max) * 400
        ax.scatter(xs[i], ys[i], s=size, c=color, zorder=5,
                   edgecolors='white', linewidths=1.5)
        ha = 'left' if xs[i] >= 0 else 'right'
        ax.text(1.18 * xs[i], 1.18 * ys[i],
                c.replace('_', '\n'),
                ha=ha, va='center', fontsize=8,
                fontweight='bold', color=color)

    # 中央テキスト
    ax.text(0, 0,
            f"edges={result.n_significant_edges}\n"
            f"hub={result.hub_muscle}\n"
            f"p<{result.p_thresh}",
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray', alpha=0.9))

    handles = [
        mpatches.Patch(color='#3498db', label='Right leg'),
        mpatches.Patch(color='#e74c3c', label='Left leg'),
        plt.Line2D([0], [0], color='#7f8c8d', lw=2,
                   marker='>', markersize=8, label='Granger causality A->B'),
    ]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9)
    ax.set_title(
        f'Granger Causality Network - {result.task_key}\n'
        f'Phase{result.phase}  {result.speed}m/s  '
        f'(maxlag={result.maxlag}, p<{result.p_thresh})',
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_causality_heatmap(
    results   : dict[str, GrangerResult],
    save_path : Optional[str | Path] = None,
) -> None:
    """
    複数タスクの Granger 因果性隣接行列をヒートマップで比較する。
    行 = 原因側の筋、列 = 結果側の筋。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tasks = list(results.keys())
    n     = len(tasks)
    ch    = results[tasks[0]].channel_names
    N     = len(ch)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, tk in zip(axes, tasks):
        r  = results[tk]
        # -log10(p) をヒートマップで表示（値が大きいほど有意）
        with np.errstate(divide='ignore'):
            neg_log_p = -np.log10(np.clip(r.p_matrix, 1e-10, 1.0))
        np.fill_diagonal(neg_log_p, 0)

        im = ax.imshow(neg_log_p, cmap='YlOrRd', aspect='equal',
                       vmin=0, vmax=5)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(ch, rotation=90, fontsize=7)
        ax.set_yticklabels(ch, fontsize=7)
        ax.axhline(7.5, color='k', linewidth=1)
        ax.axvline(7.5, color='k', linewidth=1)
        ax.set_xlabel('Target (caused)', fontsize=9)
        ax.set_ylabel('Source (causing)', fontsize=9)
        ax.set_title(
            f"{tk}\nedges={r.n_significant_edges}  "
            f"hub={r.hub_muscle}",
            fontsize=10,
        )
        plt.colorbar(im, ax=ax, label='-log10(p)', shrink=0.8)

    fig.suptitle(
        'Granger Causality (-log10 p-value)\n'
        f'Phase{results[tasks[0]].phase}  '
        f'{results[tasks[0]].speed}m/s',
        fontsize=13,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)