# =============================================================================
# analyze_tension_wavelet.py
#
# 概要:
# strength_visualize.py で生成された張力データ (TENSION_DATA_OUTPUT_PATH) を
# 読み込み、連続ウェーブレット変換 (CWT) を実行します。
#
# 処理フロー:
# 1. config.py から解析対象タスク (task1, task2, or task3) の設定を読み込む。
# 2. TENSION_DATA_OUTPUT_PATH (例: task1_tension_data.csv) を読み込む。
# 3. 各ゴムセグメント ("Front_Upper_In" など) ごとにループ処理。
# 4. 'tension_N' の時系列データ (gait_cycle_% 0-100) を抽出する。
# 5. PyWavelets (pywt.cwt) を使用して CWT を実行する。
# 6. 結果を時間-周波数平面のヒートマップ（スペクトログラム）として描画する。
#    - X軸: 歩行周期 (%)
#    - Y軸: 周波数 (Hz / Harmonics = 周期/歩行周期)
#    - 色: 振幅（張力の変動の強さ）
# 7. 各セグメントのグラフを表示し、保存オプションを提示する。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config # 設定ファイルをインポート
try:
    import pywt # 連続ウェーブレット変換ライブラリ
except ImportError:
    print("エラー: 'pywt' ライブラリが見つかりません。")
    print("ターミナルで `pip install pywavelets` を実行してください。")
    exit()

# --- ▼▼▼ 設定 ▼▼▼ ---
# グラフに表示する高調波の最大次数 (例: 10Hz = 1周期に10回振動)
MAX_HARMONIC_TO_SHOW = 10
# CWTに使用するウェーブレット (Morletが一般的)
WAVELET_NAME = 'morl'
# CWTのスケール（解像度）設定
# (小さいほど高周波、大きいほど低周波を詳細に見る)
SCALES_TO_COMPUTE = np.arange(1, 128) 
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


def load_tension_data(cfg: dict) -> pd.DataFrame | None:
    """config設定に基づき、張力データを読み込む"""
    file_path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not file_path or pd.isna(file_path):
        print(f"エラー: config に TENSION_DATA_OUTPUT_PATH が定義されていません。")
        return None
        
    try:
        df = pd.read_csv(file_path)
        print(f"張力データを読み込みました: {file_path}")
        if 'segment' not in df.columns or 'tension_N' not in df.columns or 'gait_cycle_%' not in df.columns:
            print("エラー: 必要な列 ('segment', 'tension_N', 'gait_cycle_%') が見つかりません。")
            return None
        return df
    except FileNotFoundError:
        print(f"エラー: 張力データファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        print(f"張力データ読み込みエラー: {e}")
        return None

def plot_cwt_for_segment(signal: np.ndarray, segment_name: str, task_key: str, 
                         save_plots: bool, cfg: dict):
    """
    単一セグメントの信号に対して CWT を実行し、ヒートマップをプロットする。
    """
    N = len(signal)
    if N < 10: # データが短すぎる
        print(f"警告: '{segment_name}' のデータが短すぎ ({N}点)。CWT スキップ。")
        return
        
    # サンプリング周期 (dt) の定義
    # 0% から 100% まで N 点 (例: 101点) ある場合、
    # 1歩行周期 (T=1) あたりのサンプル間隔は 1 / (N-1) となる
    dt = 1.0 / (N - 1)
    
    # 時間軸 (Gait Cycle %)
    time_axis = np.linspace(0, 100, N)

    try:
        # 連続ウェーブレット変換を実行
        # cwt_matrix: (n_scales, N)
        # freqs: (n_scales,)
        cwt_matrix, freqs = pywt.cwt(signal, SCALES_TO_COMPUTE, WAVELET_NAME, sampling_period=dt)
        
        # 振幅を計算
        amplitude = np.abs(cwt_matrix)

        # 描画
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # pcolormesh でヒートマップを描画
        # X: 時間 (0-100%), Y: 周波数 (Hz), Z(Color): 振幅
        # shading='gouraud' で滑らかに補間
        c = ax.pcolormesh(time_axis, freqs, amplitude, shading='gouraud', cmap='viridis') # 'viridis' や 'plasma' が見やすい
        fig.colorbar(c, ax=ax, label='Amplitude (N)')
        
        ax.set_title(f'Wavelet Transform (Time-Frequency) - Segment: {segment_name}\nTask: {task_key}', fontsize=14)
        ax.set_xlabel('Gait Cycle (%)', fontsize=12)
        ax.set_ylabel('Frequency (Harmonics / cycles per gait cycle)', fontsize=12)
        
        # Y軸の表示範囲を 0 から MAX_HARMONIC_TO_SHOW に制限
        ax.set_ylim(0, MAX_HARMONIC_TO_SHOW)
        
        plt.tight_layout()

        # 保存
        if save_plots:
            try:
                save_filename = f"{task_key}_{segment_name}_cwt_spectrum.png"
                save_path = os.path.join(config.RESULT_DIR, save_filename)
                os.makedirs(config.RESULT_DIR, exist_ok=True)
                fig.savefig(save_path, dpi=150)
                print(f"  -> グラフを保存しました: {save_path}")
            except Exception as e:
                print(f"  -> グラフ保存エラー: {e}")
        
        plt.show() # グラフを表示

    except Exception as e:
        print(f"エラー: '{segment_name}' の CWT 処理中にエラーが発生しました: {e}")


def main():
    """メイン実行関数"""
    # 1. タスク選択
    while True:
        task_key = input("解析するタスク名を入力 (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS.get(task_key); break
        else: print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")
    if cfg is None: print(f"エラー: {task_key} の設定読み込み失敗。"); return

    print(f"\n--- {task_key} の張力 CWT (ウェーブレット) 解析を開始します ---")

    # 2. データ読み込み
    df_tension = load_tension_data(cfg)
    if df_tension is None:
        print("処理を終了します。"); return

    # 3. 保存確認
    save_choice = input("\nプロットを画像として保存しますか？ (y/n): ").lower()
    save_plots = (save_choice == 'y')
    if save_plots:
        os.makedirs(config.RESULT_DIR, exist_ok=True)
        print(f"画像は {config.RESULT_DIR} に保存されます。")
        
    # 4. CWT 実行とプロット
    segments = sorted(df_tension['segment'].unique())
    print(f"\n{len(segments)}個のセグメントのCWTグラフを順に表示します...")
    
    for segment in segments:
        signal_df = df_tension[df_tension['segment'] == segment].sort_values('gait_cycle_%')
        signal = signal_df['tension_N'].values
        
        plot_cwt_for_segment(signal, segment, task_key, save_plots, cfg)
        
    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()