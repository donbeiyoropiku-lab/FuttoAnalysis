# =============================================================================
# strength_visualize/io_utils.py
#
# 役割:
#   すべてのファイル読み書き処理を集約する。
#   - OptiTrack 平均化CSVの読み込み
#   - EMG CSVの読み込みと最大値算出
#   - ゴム物性Excelの読み込みと補間関数の生成
#   - 張力データのCSV保存
#
# ★ 他のプログラムから CONFIG.py を使っている場合でも、
#    このモジュールだけ変更すればデータ形式の変更に対応できる。
# =============================================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


def load_opti_csv(opti_csv_path):
    """
    OptiTrack平均化CSVを読み込む。

    Parameters
    ----------
    opti_csv_path : str or Path

    Returns
    -------
    pd.DataFrame
        読み込み成功時はDataFrame。失敗時は None を返す。
    """
    print(f"\nOptiTrackデータを読み込みます: {opti_csv_path}")
    try:
        df = pd.read_csv(opti_csv_path)
        return df
    except Exception as e:
        print(f"ファイル読み込みエラー: {e}")
        print("指定されたパスに平均化されたCSVが存在するか確認してください。")
        return None


def load_rubber_properties(excel_path, sheet_name):
    """
    ゴム物性Excelを読み込み、ひずみ→力の補間関数を返す。

    Parameters
    ----------
    excel_path : str or Path
    sheet_name : str

    Returns
    -------
    scipy.interpolate.interp1d or None
        読み込み失敗時は None を返す。
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=3)
        strain_col = next((c for c in df.columns if '伸び' in str(c)), None)
        force_col  = next((c for c in df.columns if '荷重' in str(c)), None)

        if strain_col and force_col:
            strain_series = df[strain_col]
            force_series  = df[force_col]
        else:
            strain_series = df.iloc[:, 1]
            force_series  = df.iloc[:, 3]

        valid_idx = (
            pd.to_numeric(strain_series, errors='coerce').notna() &
            pd.to_numeric(force_series,  errors='coerce').notna()
        )
        strain = pd.to_numeric(strain_series[valid_idx]).values
        force  = pd.to_numeric(force_series[valid_idx]).values

        max_force = force[-1] if len(force) > 0 else 0
        return interp1d(strain, force, kind='linear',
                        fill_value=(0, max_force), bounds_error=False)
    except Exception as e:
        print(f"エラー: ゴム物性データの読み込みに失敗しました: {e}")
        return None


def load_emg_csv(emg_csv_path, muscle_indicators_def):
    """
    EMG CSVを読み込み、筋肉ごとの最大値辞書を返す。

    Parameters
    ----------
    emg_csv_path : str or Path
    muscle_indicators_def : dict
        CONFIG.py の MUSCLE_INDICATORS に相当する辞書。

    Returns
    -------
    tuple[pd.DataFrame or None, dict]
        (emg_data, max_emg_vals)
        読み込み失敗時は (None, {}) を返す。
    """
    if not emg_csv_path or not os.path.exists(emg_csv_path):
        print(f"警告: 筋電(EMG)データが見つかりません。パス: {emg_csv_path}")
        return None, {}

    print(f"筋電データの読み込みを開始します: {emg_csv_path}")
    try:
        emg_data = pd.read_csv(emg_csv_path)
        max_emg_vals = {}

        if not muscle_indicators_def:
            print("  -> [警告] 筋肉マーカーの定義が空です。")
        else:
            for m_name, m_info in muscle_indicators_def.items():
                col_name = m_info.get('emg_col') or m_info.get('emg')
                if col_name:
                    if col_name in emg_data.columns:
                        max_emg_vals[m_name] = emg_data[col_name].max()
                    else:
                        print(f"  -> [警告] 筋肉 '{m_name}' の指定列 '{col_name}' がCSV内に存在しません。")
                else:
                    print(f"  -> [警告] '{m_name}' の設定に 'emg_col' が指定されていません。")

        print(f"  -> 筋電データを正常に読み込み、{len(max_emg_vals)} 個の筋肉の列データを認識しました。")
        return emg_data, max_emg_vals
    except Exception as e:
        print(f"エラー: EMGデータの読み込みに失敗しました: {e}")
        return None, {}


def save_tension_csv(tension_df_for_csv, output_path):
    """
    張力データをCSVに保存する。

    Parameters
    ----------
    tension_df_for_csv : pd.DataFrame
    output_path : str or Path
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tension_df_for_csv.to_csv(output_path, index=False, float_format='%.4f')
        print(f"張力データをCSVに保存しました: {output_path}")
    except Exception as e:
        print(f"張力CSVの保存エラー: {e}")


def build_paths(cfg, task_key, phase, speed, subject, base_dir, result_dir):
    """
    タスク・フェーズ・被験者から各種ファイルパスを一括生成する。

    Parameters
    ----------
    cfg : dict
        CONFIG.TASK_CONFIGS[task_key]
    task_key : str
    phase : int
    speed : str  例: '0.7'
    subject : str
    base_dir : str or Path
    result_dir : str or Path

    Returns
    -------
    dict
        'opti_csv', 'emg_csv', 'tension_out' の3キーを持つ辞書。
    """
    base_path = cfg.get(
        'MEAN_CYCLE_BASE_PATH',
        rf"C:\FuttoAnalysis\opti\20260217\{task_key}_mean_cycle"
    )
    opti_csv = f"{base_path}_Phase{phase}_{speed}ms.csv"

    emg_csv = (
        Path(base_dir) / subject / "analysis_results"
        / f"{subject}_{task_key}_Phase{phase}_average.csv"
    )

    tension_out = (
        Path(result_dir) / "2026" / task_key / speed
        / f"{task_key}_Phase{phase}_{speed}ms_tension.csv"
    )

    return {
        'opti_csv':    opti_csv,
        'emg_csv':     emg_csv,
        'tension_out': tension_out,
    }