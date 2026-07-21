# =============================================================================
# mechanics_analysis/io_loader.py
#
# 役割:
#   mechanics_analysis 全体で使うデータ読み込み・CSV保存を集約する。
#   昨年度プログラムの load_data() を今年度 CONFIG 構造に対応させたもの。
#
# 今年度の変更点:
#   - パスキーが MEAN_CYCLE_BASE_PATH + Phase/speed の動的生成に変わったため、
#     呼び出し元から解決済みパスを受け取る形に統一。
#   - 旧: cfg['MEAN_CYCLE_OUTPUT_PATH'] / cfg['TENSION_DATA_OUTPUT_PATH']
#   - 新: build_analysis_paths() でパスを生成して渡す。
# =============================================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path


def build_analysis_paths(cfg: dict, task_key: str, phase: int, speed: str,
                         result_dir: str) -> dict:
    """
    タスク・フェーズ・速度から解析用の全パスを一括生成する。

    Parameters
    ----------
    cfg        : TASK_CONFIGS[task_key]
    task_key   : 'task01', 'task02', 'task03'
    phase      : 1-5
    speed      : '0.7', '0.9', ...  (m/s の数値部分)
    result_dir : CONFIG.RESULT_DIR

    Returns
    -------
    dict with keys:
        opti_csv      : 歩行周期平均 OptiTrack CSV
        tension_csv   : 張力データ CSV
        torque_out    : トルク出力 CSV
        work_out      : 仕事量出力 CSV
        graph_dir     : グラフ保存ディレクトリ
    """
    base_mean = cfg.get('MEAN_CYCLE_BASE_PATH',
                        rf"C:\FuttoAnalysis\opti\20260217\{task_key}_mean_cycle")

    opti_csv    = f"{base_mean}_Phase{phase}_{speed}ms.csv"

    # 張力データは strength_visualize が出力した result_dir 配下のファイルを参照する
    out_base    = Path(result_dir) / "2026" / task_key / speed
    tension_csv = str(out_base / f"{task_key}_Phase{phase}_{speed}ms_tension.csv")

    return {
        'opti_csv':    opti_csv,
        'tension_csv': tension_csv,
        'torque_out':  str(out_base / f"{task_key}_Phase{phase}_{speed}ms_torque.csv"),
        'work_out':    str(out_base / f"{task_key}_Phase{phase}_{speed}ms_work.csv"),
        'graph_dir':   str(out_base / "mechanics"),
    }


def load_opti_and_tension(opti_csv_path: str,
                          tension_csv_path: str) -> tuple:
    """
    OptiTrack 平均化 CSV と張力 CSV を読み込む。

    Returns
    -------
    (df_mean_cycle, df_tension) または (None, None)
    """
    df_mean = _safe_read_csv(opti_csv_path,    "OptiTrack平均化データ")
    df_ten  = _safe_read_csv(tension_csv_path, "張力データ")
    return df_mean, df_ten


def save_csv(df: pd.DataFrame, output_path: str, label: str = "データ") -> None:
    """DataFrame を CSV に保存する。"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"  -> {label}を保存しました: {output_path}")
    except Exception as e:
        print(f"  -> {label}の保存エラー: {e}")


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _safe_read_csv(path: str, label: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        print(f"エラー: {label}ファイルが見つかりません: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"  -> {label}を読み込みました: {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"エラー: {label}の読み込みに失敗しました: {e}")
        return None