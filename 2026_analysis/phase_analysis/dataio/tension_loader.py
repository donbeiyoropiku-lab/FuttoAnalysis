# =============================================================================
# phase_analysis/dataio/tension_loader.py
#
# 役割:
#   strength_visualize / mechanics_analysis が出力した張力 CSV を読み込み、
#   位相解析用の時系列 (gait_cycle_%, tension) に変換する。
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.phase_config import get_tension_csv_path, HIP_SEGMENTS


def load_tension_csv(task: str, phase_num: int, speed: str) -> 'pd.DataFrame | None':
    """
    張力 CSV を読み込む (ワイド形式・ロング形式の両方に対応)。

    ワイド形式: gait_cycle_%, Front_Upper_In, Back_Upper_In, ...
    ロング形式: gait_cycle_%, segment, tension_N

    Returns
    -------
    pd.DataFrame (ワイド形式に統一して返す) または None
    """
    csv_path = get_tension_csv_path(task, phase_num, speed)

    if not csv_path.exists():
        print(f"  -> 張力 CSV が見つかりません: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"  -> 張力 CSV 読み込み完了: {csv_path.name}  ({len(df)} フレーム)")

    # ロング形式ならワイド形式に変換して統一する
    if 'segment' in df.columns and 'tension_N' in df.columns:
        df = df.pivot(index='gait_cycle_%', columns='segment',
                       values='tension_N').reset_index()
        df.columns.name = None

    return df


def get_tension_series(task: str, phase_num: int, speed: str,
                       segment: str,
                       target_cycles: 'np.ndarray | None' = None
                       ) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    指定した1本のゴムセグメントの張力時系列を取得する。

    Parameters
    ----------
    task, phase_num, speed : タスク・フェーズ・速度
    segment       : str  セグメント名 (例: 'Front_Upper_In')
    target_cycles : np.ndarray または None
        指定時はこの gait_cycle_% 点に線形補間して返す。

    Returns
    -------
    (cycles, tension) のタプル、または None (データがない場合)
    """
    df = load_tension_csv(task, phase_num, speed)
    if df is None:
        return None

    if segment not in df.columns:
        print(f"  -> 警告: セグメント '{segment}' が張力CSVに存在しません。")
        print(f"     利用可能なセグメント: {[c for c in df.columns if c != 'gait_cycle_%']}")
        return None

    cycles  = df['gait_cycle_%'].values
    tension = df[segment].values

    if target_cycles is not None:
        tension = np.interp(target_cycles, cycles, tension)
        cycles = target_cycles

    return cycles, tension


def get_hip_tension_sum(task: str, phase_num: int, speed: str,
                        target_cycles: 'np.ndarray | None' = None
                        ) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    股関節周りのゴム (HIP_SEGMENTS) の張力合計を取得する。

    股関節角度との位相解析では、個別セグメントよりも股関節を跨ぐ
    ゴム全体の合計張力の方が力学的意味が明確な場合がある。

    Returns
    -------
    (cycles, tension_sum) のタプル、または None
    """
    df = load_tension_csv(task, phase_num, speed)
    if df is None:
        return None

    available = [s for s in HIP_SEGMENTS if s in df.columns]
    if not available:
        print(f"  -> 警告: 股関節セグメントが1つも見つかりません。"
              f" 期待: {HIP_SEGMENTS}")
        return None
    missing = [s for s in HIP_SEGMENTS if s not in df.columns]
    if missing:
        print(f"  -> 注意: 一部の股関節セグメントが見つかりません: {missing}")

    cycles = df['gait_cycle_%'].values
    tension_sum = df[available].sum(axis=1).values

    if target_cycles is not None:
        tension_sum = np.interp(target_cycles, cycles, tension_sum)
        cycles = target_cycles

    return cycles, tension_sum


def list_available_segments(task: str, phase_num: int, speed: str) -> list:
    """張力 CSV に含まれるセグメント名一覧を返す。"""
    df = load_tension_csv(task, phase_num, speed)
    if df is None:
        return []
    return [c for c in df.columns if c != 'gait_cycle_%']