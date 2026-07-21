#data_processing.py
# 歩行周期切り出し用のモジュール

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from scipy.signal import butter, filtfilt
import os

__all__ = ['process_all_files', 'calculate_gait_cycles']


#Rfz, Lfz, Btime, Kinema = 3, 10, 15, 16
# --- ▼▼▼ ここから追加 ▼▼▼ ---
# グローバル変数として列番号を定義
# ★列番号の定義をシンプルに（今回はLfzのみ使用）
#Lfx_col, Lfy_col, Lfz_col = 1, 2, 3
Lfx_col, Lfy_col, Lfz_col = 8, 9, 10
#20260217
# --- ▲▲▲ ここまで追加 ▲▲▲ ---

def get_txt_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def process_all_files(directory, based_or_not):
    files = get_txt_files(directory)
    results = {}
    for tn, file in enumerate(files):
        file_path = os.path.join(directory, file)
        print(f"処理中のファイル: {file_path}")
        data = process_data(file_path)
        results[tn] = {
            'Rfz': all_data(file_path, Rfz, based_or_not),
            'Lfz': all_data(file_path, Lfz, based_or_not),
            'Btime': all_data(file_path, Btime, based_or_not),
            'Kinema': all_data(file_path, Kinema, based_or_not)
        }
        print(f"ファイル {file} の処理結果:")
        for key, value in results[tn].items():
            print(f"{key}: {'データあり' if value is not None else 'データなし'}")
    return results


def process_data(file):
    encodings = ['utf-8', 'shift-jis', 'latin-1']
    for encoding in encodings:
        try:
            data = pd.read_csv(file, skiprows=6, header=None, sep='\t', encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the file with the specified encodings")



def interpolate_nan(col):
    mask = col.notna()
    x = np.flatnonzero(mask)
    y = col[mask]
    if len(x) < 4 or len(x) == len(col):  # NaNがない場合や補間に十分なデータがない場合
        return col
    interp = interpolate.interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return pd.Series(interp(np.arange(len(col))), index=col.index)

def lowpass_filter(data, cutoff, fs=1000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def treadmill_data(file, cn, cutoff_frequency=13):
    data = process_data(file)
    if cn >= data.shape[1]:
        raise ValueError(f"Column index {cn} is out of bounds. DataFrame has {data.shape[1]} columns.")
    data = data.iloc[:, cn]  # DataFrameの列を選択
    return lowpass_filter(data, cutoff_frequency)

def adjusted_data(file, cn):
    filtered_data = treadmill_data(file, cn)
    sample_data = filtered_data[40000:45000]
    #20-25秒
    #40-45に変更20260217
    # 下位10%のデータを抽出
    lower_10_percent = np.percentile(sample_data, 10)
    baseline_data = sample_data[sample_data <= lower_10_percent] 
    # 下位10%の平均を計算
    baseline = np.mean(baseline_data) 
    return filtered_data - baseline

'''
def all_data(file, cn, based_or_not):
    print(f"all_data関数: cn = {cn}, based_or_not = {based_or_not}")
    
    if cn in [Rfz, Lfz]:
        if based_or_not == 1:
            result = treadmill_data(file, cn)
        else:
            result = adjusted_data(file, cn)
    elif cn in [Btime, Kinema]:
        data = process_data(file)
        result = data.iloc[:, cn].to_numpy()  # Pandas SeriesをNumPy arrayに変換
    else:
        result = None
        print("無効なcn値")
    
    if result is not None:
        print(f"処理結果（最初の10要素）:")
        print(result[:10])
        print(f"all_data関数の戻り値の長さ: {len(result)}")
    else:
        print("all_data関数の戻り値: None")
    
    return result
'''

def all_data(file, cn, based_or_not):
    if based_or_not == 2: # adjusted_dataを呼び出す場合
        result = adjusted_data(file, cn)
    else: # 元データをフィルターだけかける場合
        result = treadmill_data(file, cn)
    return result

def calculate_gait_cycles(data):
    if len(data) == 0:
        print("警告: 空のデータセットです。")
        return []
    
    # 微分値の計算
    diff_data = np.diff(data)
    
    # 閾値の設定
    rise_threshold = 0.0001   # 立ち上がりの閾値
    fall_threshold = -0.0001  # 降下の閾値
    force_threshold = 0.2    # 力データの閾値
    window_size = 5         # 線形近似に使用するウィンドウサイズ
    max_cycle_duration = 2000  # 最大歩行周期時間（2秒 = 2000サンプル）
    
    stance_starts = []
    stance_ends = []
    
    # 立脚開始点の検出と補正
    for i in range(len(diff_data)-1):
        if (diff_data[i] > rise_threshold and 
            data[i] < force_threshold and 
            data[i+1] > force_threshold):
            # 検出点から前方のデータを使用して線形近似
            if i >= window_size:
                x = np.arange(i-window_size, i)
                y = data[i-window_size:i]
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # x切片を計算（実際の立脚開始点）
                actual_start = int(-intercept/slope)
                if 0 <= actual_start < len(data):  # インデックスの範囲チェック
                    stance_starts.append(actual_start)
    
    # 立脚終了点の検出と補正
    for i in range(len(diff_data)-1):
        if (diff_data[i] < fall_threshold and 
            data[i] > force_threshold and 
            data[i+1] < force_threshold):
            # 検出点から後方のデータを使用して線形近似
            if i + window_size < len(data):
                x = np.arange(i-window_size, i)
                y = data[i-window_size:i]
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # x切片を計算（実際の立脚終了点）
                actual_end = int(-intercept/slope)
                if 0 <= actual_end < len(data):  # インデックスの範囲チェック
                    stance_ends.append(actual_end)
    
   # 歩行周期のペアリング
    gait_cycles = []
    
    for start in stance_starts:
        # 現在の開始点以降で最も近い終了点を探す
        valid_ends = [end for end in stance_ends 
                     if end > start and 
                     end - start < max_cycle_duration]
        
        if valid_ends:
            end = min(valid_ends)  # 最も近い終了点を選択
            # ★★★ ここからが修正箇所 ★★★
            # 次の接地を探す
            next_starts = [s for s in stance_starts if s > end]
            if next_starts:
                next_start = min(next_starts)
                gait_cycles.append({
                    'hs_time': start / 1000.0,      # 接地時間 (秒)
                    'to_time': end / 1000.0,        # 離地時間 (秒)
                    'next_hs_time': next_start / 1000.0, # 次の接地時間 (秒)
                    'hs_frame': start,
                    'to_frame': end,
                    'next_hs_frame': next_start
                })
            # ★★★ ここまで修正 ★★★
        
    # 検出結果の出力
    print(f"\n検出された歩行周期の総数: {len(gait_cycles)}\n")
    
    return gait_cycles


def analyze_gait_phases(stance_starts, stance_ends, threshold=10):
    if not stance_starts or not stance_ends:
        print("警告: 歩行周期データが空です。")
        return []
    
    # サンプル数を秒に変換
    starts_sec = [s/1000 for s in stance_starts]
    ends_sec = [e/1000 for e in stance_ends]
    
    continuous_phases = []
    current_phase = [starts_sec[0]]
    current_ends = [ends_sec[0]]
    
    # 連続したフェーズの検出
    for i in range(1, len(starts_sec)):
        # 現在の開始点と直前の終了点との間隔を計算
        time_diff = starts_sec[i] - ends_sec[i-1]
        
        if time_diff <= threshold:
            # 連続とみなせる場合、現在のフェーズに追加
            current_phase.append(starts_sec[i])
            current_ends.append(ends_sec[i])
        else:
            # フェーズが途切れた場合の処理
            if len(current_phase) > 1:
                phase_start = min(current_phase)
                phase_end = max(current_ends)
                phase_duration = phase_end - phase_start
                
                # フェーズの妥当性チェック
                if (phase_duration >= threshold and  # フェーズ全体の長さが閾値以上
                    all(current_phase[j+1] - current_ends[j] <= threshold  # 全ての隣接する歩行周期間の間隔をチェック
                        for j in range(len(current_phase)-1))):
                    continuous_phases.append((phase_start, phase_end))
            
            # 新しいフェーズの開始
            current_phase = [starts_sec[i]]
            current_ends = [ends_sec[i]]
    
    # 最後のフェーズの処理
    if len(current_phase) > 1:
        phase_start = min(current_phase)
        phase_end = max(current_ends)
        phase_duration = phase_end - phase_start
        
        if phase_duration >= threshold:  # フェーズが10秒以上の場合のみ有効
            continuous_phases.append((phase_start, phase_end))
    
    # 検出結果の出力
    print("\n連続歩行フェーズ:")
    for i, (start, end) in enumerate(continuous_phases, 1):
        print(f"フェーズ {i}: {start:.2f}秒 - {end:.2f}秒 (継続時間: {end-start:.2f}秒)")
    
    return continuous_phases


