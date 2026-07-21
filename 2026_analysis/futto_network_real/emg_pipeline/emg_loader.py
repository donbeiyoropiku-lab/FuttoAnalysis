"""
emg_pipeline/emg_loader.py
===========================
Cometa Pico が出力する .txt ファイルを読み込み、
pandas DataFrame として返すローダー。

ファイル仕様（task01.txt より確認済み）:
  1行目 : ファイル名（task01.c3d）
  2〜4行: 空行
  5行目 : 列ヘッダー（タブ区切り）
           Time(s) | rtGM(uV) | rtIL(uV) | rtHam(uV) | rtRF(uV) |
           rtVL(uV) | rtBF(uV) | rtSOL(uV) | rtTA(uV) |
           ltGM(uV) | ltIL(uV) | ltHam(uV) | ltRF(uV) |
           ltVL(uV) | ltBF(uV) | ltSOL(uV) | ltTA(uV) |
           Acc_1:X(g) 〜 Acc_16:Z(g)（加速度・今回は使用しない）
  6行目〜: データ（タブ区切り、最終行は空行）

列名のマッピング（Cometa → CONFIG.MUSCLE_NAMES）:
  rt* → R_*  （右脚）
  lt* → L_*  （左脚）
  Ham → Ham  （ハムストリングス: ST・BFとは別チャンネルとして保持）

サンプリング周波数: 2000 Hz（ステップ 0.0005 s）
総データ長: 約 360 秒（静止40s + 歩行60s×5 + 静止20s）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# =============================================================================
# Cometa 列名 → 統一チャンネル名 マッピング
# =============================================================================

# Cometa 生データの EMG 列名（順番通り）
COMETA_EMG_COLS = [
    'rtGM',  'rtIL',  'rtHam', 'rtRF',  'rtVL',  'rtBF',  'rtSOL', 'rtTA',
    'ltGM',  'ltIL',  'ltHam', 'ltRF',  'ltVL',  'ltBF',  'ltSOL', 'ltTA',
]

# 統一チャンネル名（CONFIG.MUSCLE_NAMES と対応）
# Ham はハムストリングス全体チャンネルとして保持
UNIFIED_CHANNEL_NAMES = [
    'R_GM',  'R_ILIO', 'R_Ham', 'R_RF',  'R_VL',  'R_BF',  'R_SOL', 'R_TA',
    'L_GM',  'L_ILIO', 'L_Ham', 'L_RF',  'L_VL',  'L_BF',  'L_SOL', 'L_TA',
]

# Cometa列名 → 統一名 辞書
COMETA_TO_UNIFIED: dict[str, str] = {
    c: u for c, u in zip(COMETA_EMG_COLS, UNIFIED_CHANNEL_NAMES)
}


# =============================================================================
# ローダー
# =============================================================================

def load_cometa_txt(
    txt_path   : str | Path,
    rename_cols: bool = True,
    drop_acc   : bool = True,
) -> pd.DataFrame:
    """
    Cometa Pico の .txt ファイルを読み込んで DataFrame を返す。

    Parameters
    ----------
    txt_path   : .txt ファイルのパス
                 例: C:\\Users\\ihika\\2026_experiment\\Ide\\EMG\\task01\\task01.txt
    rename_cols: True の場合、列名を UNIFIED_CHANNEL_NAMES に変換する
    drop_acc   : True の場合、加速度列（Acc_*）を削除する

    Returns
    -------
    pd.DataFrame
      列: Time_s | R_GM | R_ILIO | R_Ham | R_RF | R_VL | R_BF | R_SOL | R_TA |
                  L_GM | L_ILIO | L_Ham | L_RF | L_VL | L_BF | L_SOL | L_TA
      単位: μV（生信号・未フィルタリング）
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"EMGファイルが見つかりません: {path}")

    print(f"[EMG Loader] 読み込み中: {path.name}")

    # ヘッダーは5行目（0-indexed: 4行目）
    df = pd.read_csv(
        path,
        sep       = '\t',
        skiprows  = 4,          # 1〜4行目をスキップ（ファイル名・空行）
        header    = 0,          # 5行目をヘッダーとして使用
        index_col = False,
        na_values = [''],
        low_memory= False,
    )

    # 末尾の空行を除去
    df = df.dropna(how='all').reset_index(drop=True)

    # 列名の整理
    # ヘッダー行の列名には単位表記（uV）や空白が含まれるため正規化する
    df.columns = [_normalize_col_name(c) for c in df.columns]

    # Time 列を確認・リネーム
    time_col = _find_time_col(df.columns)
    if time_col is None:
        raise ValueError("Time 列が見つかりません。ファイル形式を確認してください。")
    df = df.rename(columns={time_col: 'Time_s'})

    # 加速度列を除去
    if drop_acc:
        acc_cols = [c for c in df.columns if c.startswith('Acc')]
        df = df.drop(columns=acc_cols)

    # 列名を統一名に変換
    if rename_cols:
        rename_map = {}
        for col in df.columns:
            # Cometa 列名 → 統一名
            for cometa_name, unified_name in COMETA_TO_UNIFIED.items():
                if cometa_name.lower() in col.lower():
                    rename_map[col] = unified_name
                    break
        df = df.rename(columns=rename_map)

    # 数値型に変換（読み込み時に str になる場合があるため）
    for col in df.columns:
        if col != 'Time_s':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Time_s'] = pd.to_numeric(df['Time_s'], errors='coerce')

    # NaN 行を除去
    df = df.dropna(subset=['Time_s']).reset_index(drop=True)

    # サンプリング周波数の推定
    if len(df) > 1:
        dt = float(df['Time_s'].iloc[1] - df['Time_s'].iloc[0])
        fs_estimated = round(1.0 / dt)
        print(f"  → {len(df)} サンプル  推定 fs={fs_estimated} Hz  "
              f"総時間={df['Time_s'].iloc[-1]:.1f} s")
    else:
        print(f"  → {len(df)} サンプル")

    return df


def _normalize_col_name(name: str) -> str:
    """
    列名から単位表記・空白・特殊文字を除去して正規化する。
    例: 'rtGM(uV):' → 'rtGM'
        'Time(s):'  → 'Time'
        'Acc_1 :X(g):' → 'Acc_1_X'
    """
    import re
    name = str(name).strip()
    # 単位を除去: (uV), (g), (s) など
    name = re.sub(r'\([^)]*\)', '', name)
    # コロンを除去
    name = name.replace(':', '')
    # 空白を _ に変換
    name = re.sub(r'\s+', '_', name.strip())
    # 末尾の _ を除去
    name = name.rstrip('_')
    return name


def _find_time_col(columns) -> Optional[str]:
    """Time 列を探して列名を返す。"""
    for col in columns:
        if col.lower().startswith('time') or col.lower() == 'time_s':
            return col
    return None


# =============================================================================
# フェーズ区間の切り出し
# =============================================================================

# CONFIG.PHASES に対応（静止→歩行×5の時間定義）
PHASE_INTERVALS = {
    0: {'name': 'static_pre',  'start': 0.0,   'end': 40.0},   # 静止前（ベースライン）
    1: {'name': '0.7m/s',      'start': 40.0,  'end': 100.0},
    2: {'name': '0.9m/s',      'start': 100.0, 'end': 160.0},
    3: {'name': '1.1m/s',      'start': 160.0, 'end': 220.0},
    4: {'name': '1.3m/s',      'start': 220.0, 'end': 280.0},
    5: {'name': '1.5m/s',      'start': 280.0, 'end': 340.0},
    6: {'name': 'static_post', 'start': 340.0, 'end': 360.0},  # 静止後
}


def extract_phase(
    df    : pd.DataFrame,
    phase : int,
    margin_s: float = 0.0,
) -> pd.DataFrame:
    """
    DataFrame から指定フェーズの区間を切り出す。

    Parameters
    ----------
    df      : load_cometa_txt() の出力
    phase   : 0=静止前 / 1〜5=歩行フェーズ / 6=静止後
    margin_s: 区間の前後に加えるマージン [s]（デフォルト 0）

    Returns
    -------
    pd.DataFrame  指定区間のデータ
    """
    if phase not in PHASE_INTERVALS:
        raise ValueError(f"phase は 0〜6 で指定してください。（指定値: {phase}）")

    info  = PHASE_INTERVALS[phase]
    t_s   = info['start'] - margin_s
    t_e   = info['end']   + margin_s

    mask = (df['Time_s'] >= t_s) & (df['Time_s'] < t_e)
    out  = df[mask].copy().reset_index(drop=True)

    print(f"  Phase {phase} ({info['name']}): "
          f"{info['start']}〜{info['end']} s  →  {len(out)} サンプル")
    return out


def get_emg_array(
    df           : pd.DataFrame,
    channel_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    DataFrame から EMG チャンネルの numpy 配列を取得する。

    Parameters
    ----------
    df            : load_cometa_txt() または extract_phase() の出力
    channel_names : 取得する列名のリスト。None なら全EMG列を使用

    Returns
    -------
    emg   : np.ndarray shape (N_channels, T_samples)  単位: μV
    names : list[str]  チャンネル名リスト
    """
    if channel_names is None:
        # EMG 列のみ抽出（Time_s・Acc_* 以外）
        channel_names = [c for c in df.columns
                         if c != 'Time_s' and not c.startswith('Acc')]

    emg = df[channel_names].values.T.astype(float)   # (N_ch, T)
    return emg, channel_names


# =============================================================================
# パス生成ヘルパー
# =============================================================================

def build_emg_raw_path(
    task_key  : str,
    subject   : str = "Ide",
    base_dir  : str = r"C:\Users\ihika\2026_experiment",
) -> Path:
    """
    Cometa .txt ファイルのパスを生成する。

    例:
      C:\\Users\\ihika\\2026_experiment\\Ide\\EMG\\task01\\task01.txt

    Parameters
    ----------
    task_key : 'task01', 'task02', 'task03'
    subject  : 被験者名
    base_dir : 実験データのルートディレクトリ
    """
    return (
        Path(base_dir) / subject / "EMG" / task_key / f"{task_key}.txt"
    )


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    import sys

    # テスト用: アップロードされたファイルのパスを直接指定
    test_path = sys.argv[1] if len(sys.argv) > 1 else r"task01.txt"

    print("=== EMG Loader テスト ===\n")
    df = load_cometa_txt(test_path)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n先頭5行:\n{df.head()}")
    print(f"\n基本統計:\n{df.describe().round(3)}")

    # フェーズ切り出しテスト
    print("\n--- フェーズ切り出し ---")
    for ph in [0, 1, 3, 5]:
        sub = extract_phase(df, ph)
        emg, names = get_emg_array(sub)
        print(f"  Phase {ph}: emg shape={emg.shape}")
