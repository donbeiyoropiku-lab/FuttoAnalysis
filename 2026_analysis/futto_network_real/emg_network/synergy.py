"""
emg_network/synergy.py
=======================
タスクD: 筋シナジー（Muscle Synergy）の抽出（NMF解析）

EMG ≈ W × H の分解:
  W : 筋重み行列  shape (n_muscles=16, n_synergies)  — どの筋がどのシナジーに寄与するか
  H : 活性化時系列 shape (n_synergies, T=101)         — 各シナジーの歩行周期中の活性化パターン

評価:
  - シナジー数の決定: VAF (Variance Accounted For) 曲線から決定
  - タスク間比較: シナジー数・構成筋の変化で神経系の単純化を評価
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "futto_common"))
import CONFIG as CFG
from emg_network.correlation import EMGCorrelationResult, load_emg_average_csv, _simulate_emg


# =============================================================================
# NMF コア実装（sklearn 不使用の純 NumPy 版）
# =============================================================================

def _nmf_multiplicative(
    V        : np.ndarray,
    n_comp   : int,
    max_iter : int = 500,
    tol      : float = 1e-6,
    seed     : int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    乗算更新則（Lee & Seung 2001）による NMF。

    V ≈ W @ H   (V, W, H ≥ 0)

    Parameters
    ----------
    V      : shape (n_features, n_samples)  — (16筋, 101時点)
    n_comp : シナジー数

    Returns
    -------
    W : shape (n_features, n_comp)
    H : shape (n_comp, n_samples)
    """
    rng = np.random.default_rng(seed)
    n_feat, n_samp = V.shape

    # 非負ランダム初期化
    W = rng.uniform(0, 1, (n_feat, n_comp))
    H = rng.uniform(0, 1, (n_comp, n_samp))

    eps = 1e-9   # ゼロ除算防止

    prev_err = np.inf
    for _ in range(max_iter):
        # H の更新
        WtV  = W.T @ V                             # (K, S)
        WtWH = W.T @ W @ H + eps                   # (K, S)
        H   *= WtV / WtWH
        H    = np.maximum(H, eps)

        # W の更新
        VHt  = V @ H.T                             # (F, K)
        WHHt = W @ H @ H.T + eps                   # (F, K)
        W   *= VHt / WHHt
        W    = np.maximum(W, eps)

        # 収束判定
        err = float(np.linalg.norm(V - W @ H, 'fro'))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    # W を列ごとに正規化（各シナジーのノルムを1に揃える）
    norms = np.linalg.norm(W, axis=0, keepdims=True) + eps
    W    /= norms
    H    *= norms.T

    return W, H


def _vaf(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """
    VAF (Variance Accounted For) = 1 - ||V - WH||^2 / ||V||^2
    1.0 が完全な再構成。
    """
    residual = V - W @ H
    vaf = 1.0 - (np.linalg.norm(residual, 'fro') ** 2
                 / (np.linalg.norm(V, 'fro') ** 2 + 1e-12))
    return float(np.clip(vaf, 0, 1))


def _determine_n_synergies(
    V          : np.ndarray,
    max_k      : int = 8,
    vaf_thresh : float = 0.90,
) -> tuple[int, list[float]]:
    """
    VAF ≥ vaf_thresh を初めて満たす最小シナジー数を返す。

    Returns
    -------
    n_opt  : 最適シナジー数
    vafs   : k=1..max_k の VAF 値リスト
    """
    vafs = []
    for k in range(1, max_k + 1):
        W_k, H_k = _nmf_multiplicative(V, k)
        vafs.append(_vaf(V, W_k, H_k))
        if vafs[-1] >= vaf_thresh:
            return k, vafs
    return max_k, vafs


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class SynergyResult:
    task_key    : str
    phase       : int
    speed       : str
    muscle_names: list[str]

    n_synergies : int                 # 決定されたシナジー数
    vaf_curve   : list[float]         # k=1..max_k の VAF 値

    # W: 筋重み行列  shape (16, n_synergies)
    W           : np.ndarray
    # H: 活性化時系列 shape (n_synergies, 101)
    H           : np.ndarray

    vaf_final   : float               # 最終 VAF

    # 各シナジーの主動筋（重み上位3筋）
    dominant_muscles : list[list[str]] = field(default_factory=list)


# =============================================================================
# 公開 API
# =============================================================================

def compute_synergy(
    csv_path   : str | Path,
    task_key   : str,
    phase      : int,
    speed      : str,
    max_k      : int   = 8,
    vaf_thresh : float = 0.90,
    top_muscles: int   = 3,
) -> SynergyResult:
    """
    EMG データからシナジーを抽出する。

    Parameters
    ----------
    csv_path   : Ide_{task}_Phase{phase}_average.csv のパス
    task_key   : 'task01', 'task02', 'task03'
    phase      : 1〜5
    speed      : '0.7'〜'1.5'
    max_k      : 探索するシナジー数の上限
    vaf_thresh : この VAF 以上で最小シナジー数を採用
    top_muscles: 各シナジーの「主動筋」として表示する上位筋数
    """
    # データ読み込み
    emg = load_emg_average_csv(csv_path, phase)
    if emg is None:
        print(f"[情報] EMG CSV なし → シミュレーションデータで NMF: {csv_path}")
        emg = _simulate_emg(task_key, phase)

    muscles = list(CFG.MUSCLE_NAMES)

    # NMF は (n_features, n_samples) = (16, 101)
    V = emg.T   # shape (16, 101)  非負保証

    # シナジー数決定
    n_opt, vafs = _determine_n_synergies(V, max_k, vaf_thresh)

    # 最終 NMF（複数回実行して最良解を選択）
    best_vaf = -np.inf
    best_W   = None
    best_H   = None
    for seed in range(5):
        Wk, Hk = _nmf_multiplicative(V, n_opt, seed=seed)
        v = _vaf(V, Wk, Hk)
        if v > best_vaf:
            best_vaf = v
            best_W   = Wk
            best_H   = Hk

    # 各シナジーの主動筋
    dom_muscles = []
    for k in range(n_opt):
        weights    = best_W[:, k]
        top_idx    = np.argsort(weights)[::-1][:top_muscles]
        dom_muscles.append([muscles[i] for i in top_idx])

    return SynergyResult(
        task_key         = task_key,
        phase            = phase,
        speed            = speed,
        muscle_names     = muscles,
        n_synergies      = n_opt,
        vaf_curve        = vafs,
        W                = best_W,
        H                = best_H,
        vaf_final        = best_vaf,
        dominant_muscles = dom_muscles,
    )


def compare_synergy_tasks(
    results : dict[str, SynergyResult]
) -> dict:
    """
    複数タスクのシナジー解析結果を比較する。

    主な着目点:
      - task03（非装着）よりも task01/02（装着）でシナジー数が少ない
        → 神経系の単純化（自由度の低減）
      - 各シナジーの VAF ≥ 90% を達成する必要シナジー数
    """
    task_keys = sorted(results.keys())
    return {
        'task_keys'         : task_keys,
        'n_synergies'       : [results[t].n_synergies for t in task_keys],
        'vaf_final'         : [round(results[t].vaf_final, 4) for t in task_keys],
        'dominant_muscles'  : {t: results[t].dominant_muscles for t in task_keys},
    }


def synergy_cosine_similarity(r1: SynergyResult, r2: SynergyResult) -> np.ndarray:
    """
    2タスク間のシナジーベクトル（W列）の最大コサイン類似度行列を返す。
    シナジーの構成が似ているほど高値（1.0 = 完全一致）。

    Returns
    -------
    sim : np.ndarray shape (r1.n_synergies, r2.n_synergies)
    """
    W1 = r1.W   # (16, K1)
    W2 = r2.W   # (16, K2)

    # 正規化
    n1 = np.linalg.norm(W1, axis=0, keepdims=True) + 1e-12
    n2 = np.linalg.norm(W2, axis=0, keepdims=True) + 1e-12
    W1_norm = W1 / n1
    W2_norm = W2 / n2

    return W1_norm.T @ W2_norm   # (K1, K2)


# =============================================================================
# CSV 保存ヘルパー
# =============================================================================

def save_synergy_results(result: SynergyResult, out_dir: Path) -> None:
    """シナジー解析結果を CSV に保存する。"""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    # W 行列（筋重み）
    W_df = pd.DataFrame(
        result.W,
        index   = result.muscle_names,
        columns = [f"Synergy_{k+1}" for k in range(result.n_synergies)],
    )
    W_df.index.name = 'muscle'
    W_df.to_csv(out_dir / "synergy_W_weights.csv", float_format='%.4f')

    # H 行列（活性化時系列）
    H_df = pd.DataFrame(
        result.H,
        index   = [f"Synergy_{k+1}" for k in range(result.n_synergies)],
        columns = [f"gc_{t}" for t in range(101)],
    )
    H_df.index.name = 'synergy'
    H_df.to_csv(out_dir / "synergy_H_activation.csv", float_format='%.4f')

    # VAF カーブ
    pd.DataFrame({
        'n_synergies' : range(1, len(result.vaf_curve) + 1),
        'VAF'         : result.vaf_curve,
    }).to_csv(out_dir / "synergy_vaf_curve.csv", index=False, float_format='%.4f')

    print(f"  [Synergy] N={result.n_synergies}  VAF={result.vaf_final:.4f}  "
          f"保存 → {out_dir}")
