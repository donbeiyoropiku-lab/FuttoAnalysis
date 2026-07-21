# =============================================================================
# phase_analysis/dataio/angle_loader.py
#
# 役割:
#   mechanics_analysis が出力した関節角度 CSV を読み込む。
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.phase_config import get_joint_angles_csv_path


def load_joint_angles_csv(task: str, phase_num: int, speed: str
                          ) -> 'pd.DataFrame | None':
    """
    mechanics_analysis が出力した関節角度 CSV を読み込む。

    列: gait_cycle_%, hip_angle_deg, knee_angle_deg, ankle_angle_deg
    """
    csv_path = get_joint_angles_csv_path(task, phase_num, speed)

    if not csv_path.exists():
        print(f"  -> 関節角度 CSV が見つかりません: {csv_path}")
        print("     mechanics_analysis でメニュー5 (関節角度) を実行してください。")
        return None

    df = pd.read_csv(csv_path)
    print(f"  -> 関節角度 CSV 読み込み完了: {csv_path.name}  ({len(df)} フレーム)")
    return df


def get_angle_series(task: str, phase_num: int, speed: str,
                     joint: str = 'hip',
                     target_cycles: 'np.ndarray | None' = None
                     ) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    指定した関節の角度時系列を取得する。

    Parameters
    ----------
    task, phase_num, speed : タスク・フェーズ・速度
    joint         : str  'hip', 'knee', 'ankle' のいずれか
    target_cycles : np.ndarray または None  補間先の gait_cycle_%

    Returns
    -------
    (cycles, angle_deg) のタプル、または None
    """
    df = load_joint_angles_csv(task, phase_num, speed)
    if df is None:
        return None

    col = f'{joint}_angle_deg'
    if col not in df.columns:
        print(f"  -> 警告: 列 '{col}' が存在しません。"
              f" 利用可能: {list(df.columns)}")
        return None

    cycles = df['gait_cycle_%'].values
    angle  = df[col].values

    if target_cycles is not None:
        angle = np.interp(target_cycles, cycles, angle)
        cycles = target_cycles

    return cycles, angle