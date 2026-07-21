"""
emg_network_advanced/data_loader.py
=====================================
前処理済み EMG CSV を読み込み、各解析モジュールへ渡す共通ローダー。

入力ファイル:
  C:\\FuttoAnalysis\\result\\2026\\{subject}\\{task}\\{speed}\\
    {task}_Phase{N}_{speed}ms_emg_normalized.csv

列フォーマット:
  Time_s | R_GM | R_ILIO | R_Ham | R_RF | R_VL | R_BF | R_SOL | R_TA |
          | L_GM | L_ILIO | L_Ham | L_RF | L_VL | L_BF | L_SOL | L_TA
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# チャンネル名（前処理後の統一名）
CHANNEL_NAMES = [
    'R_GM', 'R_ILIO', 'R_Ham', 'R_RF', 'R_VL', 'R_BF', 'R_SOL', 'R_TA',
    'L_GM', 'L_ILIO', 'L_Ham', 'L_RF', 'L_VL', 'L_BF', 'L_SOL', 'L_TA',
]
N_CH = len(CHANNEL_NAMES)


def build_normalized_csv_path(
    subject    : str,
    task_key   : str,
    phase      : int,
    speed      : str,
    result_dir : str = r"C:\FuttoAnalysis\result",
    year       : str = "2026",
) -> Path:
    """
    前処理済み正規化 CSV のパスを返す。

    例: C:\\FuttoAnalysis\\result\\2026\\Ide\\task01\\1.1\\
          task01_Phase3_1.1ms_emg_normalized.csv
    """
    return (
        Path(result_dir) / year / subject / task_key / speed
        / f"{task_key}_Phase{phase}_{speed}ms_emg_normalized.csv"
    )


def load_normalized_emg(
    csv_path      : str | Path,
    channel_names : Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    正規化済み EMG CSV を読み込む。

    Returns
    -------
    emg    : np.ndarray shape (N_ch, T_samples)  値域 [0, 1]
    time_s : np.ndarray shape (T_samples,)
    names  : list[str]  チャンネル名
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV が見つかりません: {path}")

    df = pd.read_csv(path)
    time_s = df['Time_s'].values.astype(float)

    if channel_names is None:
        channel_names = [c for c in df.columns if c != 'Time_s']

    emg = df[channel_names].values.T.astype(float)   # (N_ch, T)
    return emg, time_s, channel_names


def load_all_tasks(
    subject    : str,
    phase      : int,
    speed      : str,
    task_keys  : list[str] = None,
    result_dir : str = r"C:\FuttoAnalysis\result",
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """
    同一被験者・同一フェーズの全タスクを一括読み込みする。

    Returns
    -------
    dict: {task_key: (emg, time_s, channel_names)}
    """
    if task_keys is None:
        task_keys = ['task01', 'task02', 'task03']

    results = {}
    for tk in task_keys:
        path = build_normalized_csv_path(subject, tk, phase, speed, result_dir)
        try:
            emg, time_s, names = load_normalized_emg(path)
            results[tk] = (emg, time_s, names)
            print(f"  [Loaded] {tk}: {emg.shape}")
        except FileNotFoundError:
            print(f"  [Skip] {tk}: ファイルなし ({path.name})")
    return results
