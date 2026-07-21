# =============================================================================
# analyze_raw_wavelet.py
#
# 目的:
#   平均化される前の「全時系列マーカーデータ」からゴムの張力を計算し、
#   連続ウェーブレット変換 (CWT) を行って、時間-周波数解析を行う。
#
# 処理フロー:
# 1. config.py からタスク設定を読み込む。
# 2. クリーンアップ済みマーカーデータ (例: task1_corrected_E.csv) を読み込む。
# 3. マーカーデータを "wide" 形式 (Time x ID_axis) に変換する。
# 4. 各ゴムセグメントについて、全期間の張力 (N) を計算する。
# 5. 張力データに対して CWT を実行し、スペクトログラムを描画する。
#    - 横軸: 時間 (秒)
#    - 縦軸: 周波数 (Hz)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config # 設定ファイルをインポート
from scipy.interpolate import interp1d
try:
    import pywt # 連続ウェーブレット変換ライブラリ
except ImportError:
    print("エラー: 'pywt' ライブラリが見つかりません。")
    print("ターミナルで `pip install pywavelets` を実行してください。")
    exit()

# --- ▼▼▼ 設定 ▼▼▼ ---
# 解析対象の周波数範囲 (Hz)
FREQ_RANGE = (0.5, 10.0) # 0.5Hz 〜 10Hz 程度を表示
# ウェーブレットの種類
WAVELET_NAME = 'morl'
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


def load_marker_data(cfg: dict) -> pd.DataFrame | None:
    """クリーンアップ済みマーカーデータを読み込み、Wide形式に変換する"""
    file_path = cfg.get('OUTPUT_CSV_PATH')
    if not file_path or not os.path.exists(file_path):
        print(f"エラー: マーカーデータファイルが見つかりません: {file_path}")
        return None

    print(f"マーカーデータを読み込み中: {file_path}")
    try:
        df_long = pd.read_csv(file_path)
        
        # Wide形式にピボット (Time をインデックス、id と 座標 を列に)
        # columns: (id, x), (id, y), (id, z) ...
        df_wide = df_long.pivot(index='Time', columns='id', values=['x', 'y', 'z'])
        
        # 列名を扱いやすく変更 (例: 15810_x)
        df_wide.columns = [f"{col[1]}_{col[0]}" for col in df_wide.columns]
        
        print(f"読み込み完了: {len(df_wide)} フレーム ({df_wide.index.min():.2f}s - {df_wide.index.max():.2f}s)")
        return df_wide
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def create_strain_force_interpolator(cfg: dict):
    """Excelからひずみ-荷重関係を読み込む (strength_visualize.py と同様)"""
    excel_path = config.RUBBER_PROPERTIES_EXCEL_PATH
    sheet_name = config.RUBBER_PROPERTIES_SHEET_NAME
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3)
        required_cols = ['ひずみ', '荷重(N)']
        if not all(col in df.columns for col in required_cols):
            print("Excelエラー: 必要な列が見つかりません。"); return None
        df = df[df['ひずみ'] >= 0].sort_values('ひずみ').dropna(subset=required_cols)
        if df.empty: return None
        
        return interp1d(df['ひずみ'], df['荷重(N)'], kind='linear', bounds_error=False,
                        fill_value=(df['荷重(N)'].iloc[0], df['荷重(N)'].iloc[-1]))
    except Exception as e:
        print(f"Excel読み込みエラー: {e}"); return None

def calculate_tension_series(df_wide: pd.DataFrame, p1: int, p2: int, natural_length: float, interp_func) -> pd.Series:
    """2点間の張力時系列を計算する"""
    try:
        # 座標データの抽出
        p1_cols = [f"{p1}_x", f"{p1}_y", f"{p1}_z"]
        p2_cols = [f"{p2}_x", f"{p2}_y", f"{p2}_z"]
        
        # データが存在するか確認
        if not all(c in df_wide.columns for c in p1_cols + p2_cols):
            return None

        p1_coords = df_wide[p1_cols].values
        p2_coords = df_wide[p2_cols].values
        
        # 距離計算
        current_lengths = np.linalg.norm(p1_coords - p2_coords, axis=1)
        
        # ひずみ計算
        strains = (current_lengths - natural_length) / natural_length
        
        # 張力計算
        tensions = interp_func(strains)
        
        return pd.Series(tensions, index=df_wide.index)
    except Exception:
        return None

def plot_raw_cwt(time_series: pd.Series, segment_name: str, task_key: str, save_plots: bool):
    """
    時系列データに対してCWTを実行し、時間-周波数プロットを描画する
    """
    if time_series is None or len(time_series) < 100:
        print(f"  スキップ: {segment_name} (データ不足)")
        return

    # サンプリング間隔 (秒)
    dt = 1.0 / config.FRAME_RATE
    
    # スケールの設定 (周波数範囲に対応するように計算)
    # 周波数 f = center_freq / (scale * dt)  => scale = center_freq / (f * dt)
    center_freq = pywt.central_frequency(WAVELET_NAME)
    min_scale = center_freq / (FREQ_RANGE[1] * dt)
    max_scale = center_freq / (FREQ_RANGE[0] * dt)
    
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=64)
    
    try:
        # CWT実行
        # coefs: (n_scales, n_samples)
        coefs, freqs = pywt.cwt(time_series.values, scales, WAVELET_NAME, sampling_period=dt)
        
        # 振幅 (絶対値)
        amplitude = np.abs(coefs)
        
        # --- プロット ---
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 時間軸と周波数軸のメッシュ
        T, F = np.meshgrid(time_series.index, freqs)
        
        # ヒートマップ (pcolormesh)
        # shading='gouraud' で滑らかに
        c = ax.pcolormesh(T, F, amplitude, shading='gouraud', cmap='jet')
        
        fig.colorbar(c, ax=ax, label='Magnitude')
        
        ax.set_title(f'Wavelet Transform (Raw Time Series) - {segment_name}\nTask: {task_key}', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_ylim(FREQ_RANGE) # Y軸の範囲を固定
        
        # 張力の平均的な推移を白線で重ねて表示（参考用）
        # 振幅のスケールに合わせるため正規化
        ax2 = ax.twinx()
        ax2.plot(time_series.index, time_series.values, color='white', alpha=0.3, linewidth=1, label='Tension (N)')
        ax2.set_ylabel('Tension (N)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        plt.tight_layout()

        if save_plots:
            filename = f"{task_key}_{segment_name}_raw_cwt.png"
            save_path = os.path.join(config.RESULT_DIR, filename)
            try:
                os.makedirs(config.RESULT_DIR, exist_ok=True)
                fig.savefig(save_path, dpi=100)
                print(f"  -> 保存: {filename}")
            except Exception as e:
                print(f"  -> 保存エラー: {e}")

        plt.show()
        
    except Exception as e:
        print(f"  CWTエラー ({segment_name}): {e}")

def main():
    # 1. タスク選択
    while True:
        task_key = input("解析するタスク名を入力 (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS[task_key]; break
        else: print(f"エラー: config.py に '{task_key}' が見つかりません。")
    
    print(f"\n--- {task_key} の全時系列ウェーブレット解析を開始します ---")

    # 2. データ読み込み
    df_wide = load_marker_data(cfg)
    interp_func = create_strain_force_interpolator(cfg)
    
    if df_wide is None or interp_func is None:
        print("データ読み込みに失敗しました。終了します。"); return

    # 3. 保存確認
    save_choice = input("\nグラフを画像として保存しますか？ (y/n): ").lower()
    save_plots = (save_choice == 'y')

    # 4. 各ゴムセグメントの解析
    lines_def = cfg.get('LINES_TO_DRAW', {})
    natural_lengths = cfg.get('NATURAL_LENGTHS', {})
    
    print("\n各ゴムの解析を開始します...")
    for name, (p1, p2) in lines_def.items():
        natural_len = natural_lengths.get(name)
        if natural_len is None: continue
        
        # 張力時系列を計算
        tension_series = calculate_tension_series(df_wide, p1, p2, natural_len, interp_func)
        
        if tension_series is not None:
            print(f"Processing: {name} ...")
            plot_raw_cwt(tension_series, name, task_key, save_plots)
        else:
            print(f"Skipping: {name} (マーカーデータ不足)")

    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()