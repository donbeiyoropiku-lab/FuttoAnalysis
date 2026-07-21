# =============================================================================
# phase_analysis/dataio/labchart_loader.py
#
# 役割:
#   LabChart で計測した床反力 (GRF) の生データ (.txt) と、
#   gaitcycle/data_processing.py が出力した歩行周期リスト (_gait_cycles.csv)
#   を読み込み、位相解析用の1周期分の正規化波形を取得する。
#
# 前処理は既存の gaitcycle/data_processing.py と同じロジックに揃えている:
#   ローパスフィルタ (13Hz, 1000Hz, 4次 Butterworth, filtfilt)
#   → ベースライン補正 (40-45秒区間の下位10%平均をゼロ点とする)
#
# 歩行周期の定義:
#   data_processing.py の calculate_gait_cycles() が返す
#   hs_frame (接地) 〜 next_hs_frame (次の接地) を「1歩行周期全体」として扱う。
#   これは mechanics_analysis / emg_synergy の gait_cycle_% (0-100%) の
#   定義 (踵接地 → 同側踵の再接地) と一致させるための選択。
#   (data_processing.py 側の calculate_gait_cycles 自体は立脚期 hs〜to の
#    検出に使われているが、1周期全体は hs〜next_hs で取得できる)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ..config.phase_config import (
    get_labchart_txt_path, get_gait_cycles_csv_path,
    GRF_CHANNEL_MAP, LABCHART_FS,
    GRF_FILTER_CUTOFF_HZ, GRF_FILTER_ORDER,
    GRF_BASELINE_SAMPLE_RANGE, GRF_BODYWEIGHT_SAMPLE_RANGE,
)


# ---------------------------------------------------------------------------
# 生データ読み込み・前処理
# ---------------------------------------------------------------------------

def _read_labchart_txt(filepath) -> 'pd.DataFrame | None':
    """
    LabChart の生データ TXT を読み込む。

    先頭6行 (Interval, ExcelDateTime, TimeFormat, DateFormat,
    ChannelTitle, Range) をスキップし、タブ区切りで読み込む。
    列0 = Time [s]、列1〜13 = チャンネル1〜13。
    """
    if not filepath.exists():
        print(f"  -> LabChart ファイルが見つかりません: {filepath}")
        return None

    encodings = ['utf-8', 'shift-jis', 'latin-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, skiprows=6, header=None,
                             sep='\t', encoding=encoding)
            print(f"  -> LabChart ファイル読み込み完了: {filepath.name}"
                  f"  ({len(df)} サンプル, {len(df)/LABCHART_FS:.1f} 秒)")
            return df
        except UnicodeDecodeError:
            continue
    print(f"  -> エラー: {filepath.name} をデコードできませんでした。")
    return None


def _lowpass_filter(data: np.ndarray,
                    cutoff: float = GRF_FILTER_CUTOFF_HZ,
                    fs: float = LABCHART_FS,
                    order: int = GRF_FILTER_ORDER) -> np.ndarray:
    """4次 Butterworth ローパスフィルタ (data_processing.py と同じ設定)。"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def _baseline_correct(filtered_data: np.ndarray,
                      sample_range: tuple = GRF_BASELINE_SAMPLE_RANGE
                      ) -> np.ndarray:
    """
    ベースライン補正 (data_processing.py の adjusted_data と同じロジック)。

    指定区間 (デフォルト 40-45秒) の下位10%の平均をゼロ点として引く。
    """
    start, end = sample_range
    if end > len(filtered_data):
        # データが短い場合は全体から推定 (フォールバック)
        print(f"  -> 警告: ベースライン区間 ({start}-{end}) がデータ長"
              f" ({len(filtered_data)}) を超えています。全区間で代用します。")
        sample_data = filtered_data
    else:
        sample_data = filtered_data[start:end]

    lower_10_percent = np.percentile(sample_data, 10)
    baseline_data = sample_data[sample_data <= lower_10_percent]
    baseline = np.mean(baseline_data)
    return filtered_data - baseline


def get_grf_raw(date: str, task: str, leg: str = 'L',
                component: str = 'Fz') -> 'np.ndarray | None':
    """
    フィルタ済み・ベースライン補正済みの床反力生データ (1000Hz) を取得する。

    Parameters
    ----------
    date      : str  計測日 (例: '20260217')
    task      : str  タスク名 (例: 'task01')
    leg       : str  'L' または 'R'
    component : str  'Fx', 'Fy', 'Fz' (または 'Mx','My','Mz')

    Returns
    -------
    np.ndarray (1000Hz の時系列) または None
    """
    filepath = get_labchart_txt_path(date, task)
    df = _read_labchart_txt(filepath)
    if df is None:
        return None

    if leg not in GRF_CHANNEL_MAP or component not in GRF_CHANNEL_MAP[leg]:
        print(f"  -> 警告: leg='{leg}', component='{component}' の組み合わせは"
              f" 未定義です。GRF_CHANNEL_MAP を確認してください。")
        return None

    col = GRF_CHANNEL_MAP[leg][component]
    if col >= df.shape[1]:
        print(f"  -> エラー: 列 {col} がデータ範囲外です"
              f" (データは {df.shape[1]} 列)。")
        return None

    raw = df.iloc[:, col].to_numpy()
    filtered = _lowpass_filter(raw)
    corrected = _baseline_correct(filtered)
    return corrected


def estimate_body_weight(date: str, task: str, leg: str = 'L') -> float:
    """
    立位区間の Fz から体重 (相対値, V単位) を推定する。

    gaitcycle_force_labchart.py と同じロジック:
        standing_force = mean(Fz[1000:10000])  (安定立位 1-10秒)
        body_weight = standing_force * 2  (片足荷重の2倍)
    """
    fz = get_grf_raw(date, task, leg=leg, component='Fz')
    if fz is None:
        return 1.0

    start, end = GRF_BODYWEIGHT_SAMPLE_RANGE
    standing_force = np.mean(fz[start:end])
    if standing_force <= 0:
        print("  -> 警告: 立位荷重が0以下です。体重推定をスキップします (1.0を使用)。")
        return 1.0
    return float(standing_force * 2)


# ---------------------------------------------------------------------------
# 歩行周期リストの読み込み
# ---------------------------------------------------------------------------

def load_gait_cycles_csv(date: str, task: str) -> 'pd.DataFrame | None':
    """
    data_processing.py が出力した歩行周期リスト CSV を読み込む。

    列: hs_time, to_time, next_hs_time, hs_frame, to_frame, next_hs_frame
    """
    csv_path = get_gait_cycles_csv_path(date, task)
    if not csv_path.exists():
        print(f"  -> 歩行周期リスト CSV が見つかりません: {csv_path}")
        print("     gaitcycle/gaitcycle_force_labchart.py を実行してください。")
        return None

    df = pd.read_csv(csv_path)
    print(f"  -> 歩行周期リスト読み込み完了: {csv_path.name}  ({len(df)} 周期)")
    return df


def list_available_cycles(date: str, task: str) -> 'pd.DataFrame | None':
    """検出された歩行周期の一覧 (時刻・継続時間) を返す。"""
    df = load_gait_cycles_csv(date, task)
    if df is None:
        return None
    df = df.copy()
    df['duration_s'] = df['next_hs_time'] - df['hs_time']
    return df


# ---------------------------------------------------------------------------
# 1周期分の正規化波形取得
# ---------------------------------------------------------------------------

def _normalize_cycle(data: np.ndarray, start: int, end: int,
                     n_points: int = 101) -> np.ndarray:
    """指定区間を n_points 点に線形補間して正規化する。"""
    cycle_data = data[start:end]
    if len(cycle_data) < 2:
        raise ValueError(f"周期区間が短すぎます (start={start}, end={end})")
    x_new = np.linspace(0, 100, n_points)
    x_old = np.linspace(0, 100, len(cycle_data))
    return np.interp(x_new, x_old, cycle_data)


def get_grf_cycle_series(date: str, task: str, cycle_idx: int,
                         leg: str = 'L', component: str = 'Fz',
                         normalize_bw: bool = True,
                         n_points: int = 101
                         ) -> 'tuple[np.ndarray, np.ndarray] | None':
    """
    指定した周期番号の床反力波形を、歩行周期 0-100% に正規化して取得する。

    1周期の定義: hs_frame (接地) 〜 next_hs_frame (次の接地)
    ( mechanics_analysis / emg_synergy の gait_cycle_% と同じ定義 )

    Parameters
    ----------
    date, task    : 計測日・タスク名
    cycle_idx     : int  gait_cycles.csv の行番号 (0始まり)
    leg           : str  'L' または 'R'
    component     : str  'Fx', 'Fy', 'Fz'
    normalize_bw  : bool  True の場合、体重で正規化した %BW 単位で返す
    n_points      : int  正規化後の点数 (デフォルト101 = 歩行周期%と同じ)

    Returns
    -------
    (cycles, force) のタプル ( cycles: 0-100 の gait_cycle_%, force: 波形 )
    または None
    """
    cycles_df = load_gait_cycles_csv(date, task)
    if cycles_df is None:
        return None

    if not (0 <= cycle_idx < len(cycles_df)):
        print(f"  -> 警告: cycle_idx={cycle_idx} は範囲外です"
              f" (検出周期数: {len(cycles_df)})")
        return None

    row = cycles_df.iloc[cycle_idx]
    start, end = int(row['hs_frame']), int(row['next_hs_frame'])

    force_raw = get_grf_raw(date, task, leg=leg, component=component)
    if force_raw is None:
        return None

    if end > len(force_raw):
        print(f"  -> 警告: 周期終了フレーム ({end}) がデータ長 "
              f"({len(force_raw)}) を超えています。")
        return None

    if normalize_bw:
        body_weight = estimate_body_weight(date, task, leg=leg)
        force_raw = (force_raw / body_weight) * 100.0

    force_cycle = _normalize_cycle(force_raw, start, end, n_points=n_points)
    gait_cycle_pct = np.linspace(0, 100, n_points)

    return gait_cycle_pct, force_cycle


def get_grf_phase_average_series(date: str, task: str,
                                 leg: str = 'L', component: str = 'Fz',
                                 normalize_bw: bool = True,
                                 n_points: int = 101,
                                 min_duration_s: float = 0.5,
                                 max_duration_s: float = 3.0
                                 ) -> 'dict | None':
    """
    タスク内で検出された全歩行周期を正規化・平均し、
    フェーズを代表する平均床反力波形を取得する。

    各周期を個別に 0-100% (n_points 点) へ正規化してから
    サンプル方向に平均する (gaitcycle_force_labchart.py の
    'Mean +/- Std. Dev.' 表示と同じ考え方)。

    異常に短い/長い周期 (検出ミスの可能性が高いもの) は
    min_duration_s 〜 max_duration_s の範囲外として除外する。

    Parameters
    ----------
    date, task     : 計測日・タスク名
    leg            : str  'L' または 'R'
    component      : str  'Fx', 'Fy', 'Fz'
    normalize_bw   : bool  True の場合、体重で正規化した %BW 単位で返す
    n_points       : int  正規化後の点数 (デフォルト101)
    min_duration_s : float  有効とみなす周期の最小継続時間 [秒]
    max_duration_s : float  有効とみなす周期の最大継続時間 [秒]

    Returns
    -------
    dict または None:
        'cycles'     : np.ndarray (n_points,)  gait_cycle_%
        'mean'       : np.ndarray (n_points,)  平均波形 (位相解析にはこれを使う)
        'std'        : np.ndarray (n_points,)  標準偏差
        'n_cycles'   : int   平均に使われた有効周期数
        'n_total'    : int   検出された全周期数
        'all_cycles' : np.ndarray (n_valid, n_points)  各周期の正規化波形
    """
    cycles_df = load_gait_cycles_csv(date, task)
    if cycles_df is None:
        return None

    force_raw = get_grf_raw(date, task, leg=leg, component=component)
    if force_raw is None:
        return None

    if normalize_bw:
        body_weight = estimate_body_weight(date, task, leg=leg)
        force_raw = (force_raw / body_weight) * 100.0

    all_normalized = []
    n_excluded_duration = 0
    n_excluded_range = 0

    for _, row in cycles_df.iterrows():
        start, end = int(row['hs_frame']), int(row['next_hs_frame'])
        duration_s = (end - start) / LABCHART_FS

        if not (min_duration_s <= duration_s <= max_duration_s):
            n_excluded_duration += 1
            continue
        if end > len(force_raw):
            n_excluded_range += 1
            continue

        try:
            normalized = _normalize_cycle(force_raw, start, end, n_points=n_points)
            all_normalized.append(normalized)
        except ValueError:
            n_excluded_range += 1
            continue

    if not all_normalized:
        print("  -> 有効な周期が1つも見つかりませんでした。")
        print(f"     (継続時間で除外: {n_excluded_duration}, 範囲外で除外: {n_excluded_range})")
        return None

    all_normalized = np.array(all_normalized)
    mean_curve = all_normalized.mean(axis=0)
    std_curve  = all_normalized.std(axis=0)
    gait_cycle_pct = np.linspace(0, 100, n_points)

    print(f"  -> フェーズ平均: {len(all_normalized)}/{len(cycles_df)} 周期を使用"
          f" (継続時間で除外: {n_excluded_duration}, 範囲外で除外: {n_excluded_range})")

    return {
        'cycles':     gait_cycle_pct,
        'mean':       mean_curve,
        'std':        std_curve,
        'n_cycles':   len(all_normalized),
        'n_total':    len(cycles_df),
        'all_cycles': all_normalized,
    }