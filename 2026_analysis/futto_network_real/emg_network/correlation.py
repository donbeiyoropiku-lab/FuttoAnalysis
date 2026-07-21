"""
emg_network/correlation.py
===========================
16筋EMGデータの相互相関行列を計算する。

実データパス:
  C:\\Users\\ihika\\2026_experiment\\Ide\\analysis_results\\
    Ide_{task}_Phase{phase}_average.csv

列名規則（actual CSV）:
  L_GM_mean, L_ILIO_mean, ..., R_TA_mean
  → CONFIG.MUSCLE_NAMES = ['R_GM', 'R_ILIO', ..., 'L_TA']

対側性効果（Contralateral Effect）の定量:
  Futto装着条件（task01/02）vs 非装着（task03）で
  左右筋間の相互相関が増加するかを測定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG


# =============================================================================
# 実験プロトコル定数
# =============================================================================

MUSCLE_NAMES       = CFG.MUSCLE_NAMES           # 16筋のチャンネル名
MUSCLE_NAMES_L     = CFG.MUSCLE_NAMES_L         # 左側8筋
MUSCLE_NAMES_R     = CFG.MUSCLE_NAMES_R         # 右側8筋
N_MUSCLES          = len(MUSCLE_NAMES)          # = 16

# CSV の _mean 列名 (Ide_task01_Phase3_average.csv の列)
EMG_COL_SUFFIX = '_mean'
EMG_COLS_EXPECTED = [m + EMG_COL_SUFFIX for m in MUSCLE_NAMES]


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class EMGCorrelationResult:
    task_key      : str
    phase         : int
    speed         : str
    muscle_names  : list[str]            # 16筋名 (index に対応)

    corr_matrix   : np.ndarray           # shape (16, 16)  全周期平均相関
    corr_sliding  : np.ndarray           # shape (T, 16, 16) 時変相関
    adj_matrix    : np.ndarray           # 閾値処理済み (16, 16)
    symmetry_index: np.ndarray           # shape (8,)  左右対称性

    mean_abs_corr : float
    modularity_hint: float               # within - between ブロック相関差


# =============================================================================
# データ読み込み
# =============================================================================

def load_emg_average_csv(
    csv_path: str | Path,
    phase: int,
    phase_start_pct: float = 0.0,
    phase_end_pct: float   = 100.0,
) -> Optional[np.ndarray]:
    """
    Ide_{task}_Phase{phase}_average.csv を読み込み、
    歩行周期に正規化された EMG 行列 shape (T, 16) を返す。

    列順は CONFIG.MUSCLE_NAMES に統一する。
    ファイルが存在しない場合は None を返す。
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # gait_cycle_% 列の確認
    gc_col = None
    for c in ['gait_cycle_%', 'gait_%', 'cycle_%', 'cycle']:
        if c in df.columns:
            gc_col = c
            break

    emg_matrix = np.zeros((101, N_MUSCLES))

    for m_idx, m_name in enumerate(MUSCLE_NAMES):
        col = m_name + EMG_COL_SUFFIX
        if col in df.columns:
            vals = df[col].values.astype(float)
            # 0〜100 の 101 点に補間
            if len(vals) != 101:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 100, len(vals))
                x_new = np.arange(0, 101, dtype=float)
                vals  = interp1d(x_old, vals, kind='linear',
                                 fill_value=(vals[0], vals[-1]),
                                 bounds_error=False)(x_new)
            
            # 部位ごとの最大値で正規化（%Peak: 0.0〜1.0）
            max_val = vals.max()
            if max_val > 1e-9:
                vals = vals / max_val
            else:
                vals = np.zeros_like(vals)
                
            emg_matrix[:, m_idx] = vals
        # else: 列なし → 0 のまま

    return emg_matrix


def _simulate_emg(
    task_key: str,
    phase: int,
    T: int = 101,
) -> np.ndarray:
    """
    実データ未取得時の擬似 EMG（開発用）。
    task01 では左右のクロス相関がやや高めになるよう設計。
    """
    rng = np.random.default_rng(hash(task_key + str(phase)) % 2**31)
    t   = np.linspace(0, 2 * np.pi, T)
    emg = np.zeros((T, N_MUSCLES))

    # 左脚8筋 (index 8〜15 = MUSCLE_NAMES_L)
    phases_L = rng.uniform(0, np.pi, 8)
    amps_L   = rng.uniform(0.3, 0.9, 8)

    for k in range(8):
        sig = amps_L[k] * (np.sin(t + phases_L[k]) + 1) / 2
        emg[:, 8 + k] = np.clip(sig + rng.normal(0, 0.05, T), 0, 1)

    # 右脚 (index 0〜7 = MUSCLE_NAMES_R) = 左脚 50% シフト + ノイズ
    # task01 ではシフト後の相関が高い（Futto 効果のモデル）
    cross_noise = 0.05 if task_key == 'task01' else 0.15
    for k in range(8):
        shifted = np.roll(emg[:, 8 + k], T // 2)
        emg[:, k] = np.clip(shifted + rng.normal(0, cross_noise, T), 0, 1)

    return emg


# =============================================================================
# 相関行列計算
# =============================================================================

def _sliding_corr(emg: np.ndarray, window: int = 20) -> np.ndarray:
    """スライディングウィンドウで時変相関行列を計算する。"""
    T, C  = emg.shape
    corr_t = np.zeros((T, C, C))

    for t in range(T):
        ts = max(0, t - window // 2)
        te = min(T, t + window // 2 + 1)
        seg = emg[ts:te, :]
        if seg.shape[0] < 3 or np.any(seg.std(axis=0) < 1e-10):
            corr_t[t] = np.eye(C)
        else:
            c = np.corrcoef(seg.T)
            np.fill_diagonal(c, 0)
            corr_t[t] = c

    return corr_t


def _symmetry_index(emg: np.ndarray) -> np.ndarray:
    """
    左右対称筋ペアの対称性指標 SI = 1 - |r(L, R)|。
    SI=0 が完全対称、SI=1 が非対称。
    """
    si = np.zeros(8)
    for k in range(8):
        L = emg[:, 8 + k]   # 左側
        R = emg[:, k]        # 右側
        r = float(np.corrcoef(L, R)[0, 1])
        si[k] = 1.0 - abs(r)
    return si


def _modularity_hint(corr: np.ndarray) -> float:
    """
    左（8〜15）ブロック内相関 vs 左右間相関の差。
    高値 = 左右が独立したモジュール（Futto 非装着に典型）
    低値 = 左右が強く協調（Futto 装着時の対側効果）
    """
    left  = list(range(8, 16))
    right = list(range(0, 8))

    def mean_abs(rows, cols):
        vals = [abs(corr[r, c]) for r in rows for c in cols if r != c]
        return float(np.mean(vals)) if vals else 0.0

    within  = (mean_abs(left, left) + mean_abs(right, right)) / 2
    between = mean_abs(left, right)
    return within - between


# =============================================================================
# 公開 API
# =============================================================================

def compute_emg_correlation(
    csv_path: str | Path,
    task_key: str,
    phase: int,
    speed: str,
    window: int = 20,
    threshold: float = 0.3,
) -> EMGCorrelationResult:
    """
    EMG相関行列を計算する。

    Parameters
    ----------
    csv_path  : Ide_{task}_Phase{phase}_average.csv のパス
    task_key  : 'task01', 'task02', 'task03'
    phase     : 1〜5
    speed     : '0.7', ..., '1.5'
    window    : スライディングウィンドウ幅
    threshold : 相関閾値（弱い相関をゼロにする）
    """
    emg = load_emg_average_csv(csv_path, phase)
    if emg is None:
        print(f"[情報] EMG CSV なし → シミュレーションデータで代替: {csv_path}")
        emg = _simulate_emg(task_key, phase)

    # 全周期相関
    corr = np.corrcoef(emg.T)    # (16, 16)
    np.fill_diagonal(corr, 0)

    # 閾値処理
    adj = np.where(np.abs(corr) >= threshold, corr, 0.0)
    np.fill_diagonal(adj, 0)

    # 時変相関
    corr_slid = _sliding_corr(emg, window)

    # 対称性
    si = _symmetry_index(emg)

    off_diag  = corr[~np.eye(16, dtype=bool)]
    mean_abs  = float(np.abs(off_diag).mean())
    mod_hint  = _modularity_hint(corr)

    return EMGCorrelationResult(
        task_key       = task_key,
        phase          = phase,
        speed          = speed,
        muscle_names   = list(MUSCLE_NAMES),
        corr_matrix    = corr,
        corr_sliding   = corr_slid,
        adj_matrix     = adj,
        symmetry_index = si,
        mean_abs_corr  = mean_abs,
        modularity_hint= mod_hint,
    )


def build_emg_csv_path(
    task_key: str,
    phase: int,
    subject: str = "Ide",
    base_dir: str = r"C:\Users\ihika\2026_experiment",
) -> Path:
    """
    実験プロトコルに従った EMG CSV パスを返す。

    例: C:\\Users\\ihika\\2026_experiment\\Ide\\analysis_results\\
          Ide_task01_Phase3_average.csv
    """
    return (
        Path(base_dir) / subject / "analysis_results"
        / f"{subject}_{task_key}_Phase{phase}_average.csv"
    )


def contralateral_coupling(result: EMGCorrelationResult) -> dict[str, float]:
    """
    左右クロス相関と同側相関の比率（対側性効果の指標）。

    Returns
    -------
    dict:
      'cross_mean'       : 左右クロス相関の絶対値平均
      'ipsi_mean'        : 同側相関の絶対値平均
      'cross_ipsi_ratio' : cross / ipsi  (>1 なら対側効果が強い)
    """
    C    = result.corr_matrix
    left = list(range(8, 16))
    right= list(range(0, 8))

    cross = [abs(C[l, r]) for l in left for r in right]
    ipsi_L= [abs(C[l1, l2]) for l1 in left  for l2 in left  if l1 != l2]
    ipsi_R= [abs(C[r1, r2]) for r1 in right for r2 in right if r1 != r2]

    cm = float(np.mean(cross))  if cross  else 0.0
    im = float(np.mean(ipsi_L + ipsi_R)) if (ipsi_L + ipsi_R) else 0.0
    return {
        'cross_mean'       : cm,
        'ipsi_mean'        : im,
        'cross_ipsi_ratio' : cm / im if im > 0 else float('inf'),
    }


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    print("=== EMG Correlation テスト ===\n")
    for tk in ['task01', 'task02', 'task03']:
        path = build_emg_csv_path(tk, phase=3)
        r    = compute_emg_correlation(path, tk, phase=3, speed='1.1')
        cc   = contralateral_coupling(r)
        print(f"{tk}:")
        print(f"  mean |corr|      : {r.mean_abs_corr:.4f}")
        print(f"  Modularity hint  : {r.modularity_hint:.4f}")
        print(f"  Symmetry Index   : {r.symmetry_index.round(3)}")
        print(f"  cross/ipsi ratio : {cc['cross_ipsi_ratio']:.4f}")
        print()
