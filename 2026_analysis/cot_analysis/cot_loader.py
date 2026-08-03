# =============================================================================
# cot_analysis/cot_loader.py
#
# 役割:
#   CPET(呼気代謝測定器)の生データ(.txt)を読み込むローダー。
#
# ファイル仕様 (COT.ipynb 参考プログラムに基づく):
#   1行目      : タブ区切りのメタデータ行。"Weight (Kg):" の次の列に体重が入っている。
#   "t"で始まる行: 列ヘッダー行 (これ以降がデータ本体)。
#   データ列   : "t" (経過時間, H:MM:SS形式), "VO2" (酸素摂取量, ml/min) 等。
#   エンコーディング: Shift-JIS。
# =============================================================================

from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_cpet_txt(path) -> tuple[pd.DataFrame, float | None]:
    """
    CPET生データ(.txt)を読み込み、経過時間(秒)付きのDataFrameと
    ファイルヘッダーに記録された体重(kg)を返す。

    Parameters
    ----------
    path : str または Path

    Returns
    -------
    (df, recorded_weight_kg)
        df : 'Time_sec' (計測開始からの経過秒), 'VO2' (ml/min) 等の列を持つ DataFrame
        recorded_weight_kg : ファイルヘッダーに記録された体重。見つからない場合 None。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CPETファイルが見つかりません: {path}")

    with open(path, 'r', encoding='shift-jis', errors='replace') as f:
        lines = f.readlines()

    # --- 1行目から体重を抽出 ---
    recorded_weight_kg = None
    header_cols = lines[0].split('\t')
    for i, col in enumerate(header_cols):
        if "Weight (Kg):" in col and i + 1 < len(header_cols):
            try:
                recorded_weight_kg = float(header_cols[i + 1])
            except ValueError:
                pass
            break

    # --- データ本体の開始行 ("t" で始まる列ヘッダー行) を探す ---
    data_start_index = None
    for i, line in enumerate(lines):
        if line.startswith("t\t"):
            data_start_index = i
            break
    if data_start_index is None:
        raise ValueError(f"データヘッダー行 ('t' で始まる行) が見つかりません: {path}")

    df = pd.read_csv(path, sep='\t', skiprows=data_start_index, encoding='shift-jis')

    time_col = 't'
    t = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce')
    df['Time_sec'] = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(float)

    df['VO2'] = pd.to_numeric(df['VO2'], errors='coerce')
    df = df.dropna(subset=['Time_sec', 'VO2']).reset_index(drop=True)

    return df, recorded_weight_kg
