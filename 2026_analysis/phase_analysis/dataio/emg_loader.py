# =============================================================================
# phase_analysis/dataio/emg_loader.py
#
# 役割:
#   既存 EMG 前処理 (emg_synergy と同じ average CSV) を読み込み、
#   位相解析用に単一筋の EMG 時系列を取得する。
#
# 列名規則 (emg_synergy/dataio/emg_loader.py と同じ):
#   {L|R}_{筋名}_mean   (例: L_GM_mean, R_ILIO_mean)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.phase_config import get_emg_average_csv_path

# 主要筋の略称 (emg_synergy と共通)
MUSCLE_NAMES = ['GM', 'ILIO', 'ST', 'RF', 'VL', 'BF', 'SOL', 'TA']

# 位相解析で「股関節の力学」に最も直接関わる筋 (大殿筋 = 股関節伸展)
DEFAULT_HIP_MUSCLE = 'GM'


def load_emg_average_csv(subject: str, task: str, phase_num: int
                         ) -> 'pd.DataFrame | None':
    """
    既存 EMG 前処理が出力した average CSV を読み込む。

    列: GaitCycle_%(index), L_GM_mean, R_GM_mean, ...
    """
    csv_path = get_emg_average_csv_path(subject, task, phase_num)

    if not csv_path.exists():
        print(f"  -> EMG average CSV が見つかりません: {csv_path}")
        return None

    df = pd.read_csv(csv_path, index_col=0)
    print(f"  -> EMG average CSV 読み込み完了: {csv_path.name}  ({len(df)} フレーム)")
    return df


def get_emg_series(subject: str, task: str, phase_num: int,
                   muscle: str = DEFAULT_HIP_MUSCLE,
                   leg: str = 'L',
                   target_cycles: 'np.ndarray | None' = None
                   ) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    指定した筋の EMG 時系列 (正規化前の平均値) を取得する。

    Parameters
    ----------
    subject, task, phase_num : 被験者・タスク・フェーズ
    muscle        : str  筋略称 (例: 'GM')
    leg           : str  'L' または 'R'
    target_cycles : np.ndarray または None  補間先の gait_cycle_%

    Returns
    -------
    (cycles, emg) のタプル、または None
    """
    df = load_emg_average_csv(subject, task, phase_num)
    if df is None:
        return None

    col = f'{leg}_{muscle}_mean'
    if col not in df.columns:
        print(f"  -> 警告: 列 '{col}' が存在しません。"
              f" 利用可能: {[c for c in df.columns if c.endswith('_mean')]}")
        return None

    try:
        cycles = df.index.astype(float).values
    except (ValueError, TypeError):
        cycles = np.linspace(0, 100, len(df))

    emg = df[col].values.astype(float)
    emg = np.nan_to_num(emg, nan=0.0)

    if target_cycles is not None:
        emg = np.interp(target_cycles, cycles, emg)
        cycles = target_cycles

    return cycles, emg