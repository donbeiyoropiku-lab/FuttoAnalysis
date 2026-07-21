# =============================================================================
# fft.py
#
# 概要:
# strength_visualize.py で生成された張力データ (TENSION_DATA_OUTPUT_PATH) を
# 読み込み、フーリエ変換 (FFT) を実行します。
# 各ゴムセグメントの張力データが、歩行周期の基本周波数の
# 何倍の成分（高調波）で構成されているかを可視化します。
#
# 処理フロー:
# 1. config.py から解析対象タスク (task1, task2, or task3) の設定を読み込む。
# 2. TENSION_DATA_OUTPUT_PATH (例: task1_tension_data.csv) を読み込む。
# 3. 各ゴムセグメント ("Front_Upper_In" など) ごとにループ処理。
# 4. 'tension_N' の時系列データ (gait_cycle_% 0-100) を抽出する。
# 5. NumPy の FFT (np.fft.fft) を適用する。
# 6. 周波数軸を「高調波 (Harmonic)」（周期/歩行）に変換する。
#    - 0 Hz (Harmonic 0): 平均張力 (DC成分)
#    - 1 Hz (Harmonic 1): 歩行周期と同じ周波数 (基本成分)
#    - 2 Hz (Harmonic 2): 歩行周期の2倍の周波数 ...
# 7. 各セグメントの周波数スペクトル（どの高調波がどれだけ強いか）を
#    棒グラフでプロットして表示する。
# 8. プロットを画像として保存するオプションを表示する。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config # 設定ファイルをインポート

# --- ▼▼▼ 設定 ▼▼▼ ---
# グラフに表示する高調波の最大次数
# (例: 10 なら 0Hz から 10Hz まで表示)
MAX_HARMONIC_TO_SHOW = 10
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

def perform_fft(signal: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    単一の信号 (1周期分) に対してFFTを実行し、
    高調波の周波数と振幅を返す。
    """
    N = len(signal) # 信号長 (例: 101)
    if N == 0:
        return None, None

    try:
        # フーリエ変換
        fft_vals = np.fft.fft(signal)
        # 周波数軸を作成 (単位: 周期/サンプル)
        fft_freq = np.fft.fftfreq(N)
        
        # 周波数軸を「高調波 (周期/歩行周期)」に変換
        # (Fs = N と同義)
        harmonics = fft_freq * N

        # 振幅を計算 (片側スペクトル)
        amplitude = np.abs(fft_vals) * (2.0 / N)
        # 0 Hz (DC成分) は2倍しない
        amplitude[0] = amplitude[0] / 2.0
        
        # 正の周波数成分のみ（0Hz から Fs/2 まで）を返す
        positive_mask = harmonics >= 0
        return harmonics[positive_mask], amplitude[positive_mask]
        
    except Exception as e:
        print(f"FFT計算エラー: {e}")
        return None, None

def plot_fft_results(results: dict, task_key: str):
    """FFT結果をサブプロットで描画し、保存を尋ねる"""
    num_segments = len(results)
    if num_segments == 0:
        print("描画するFFT結果がありません。")
        return

    # 描画しやすいようにサブプロットの数を調整 (例: 6列)
    cols = 6
    rows = int(np.ceil(num_segments / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharey=True)
    axes = axes.flatten() # 1次元配列に
    
    fig.suptitle(f'Tension Frequency Spectrum (Harmonics) - Task: {task_key}', fontsize=16)

    max_amp = 0 # 全グラフのY軸を揃えるため

    plot_count = 0
    for i, (segment, (freqs, amps)) in enumerate(results.items()):
        ax = axes[i]
        
        if freqs is None or amps is None:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_title(segment, fontsize=10)
            continue
            
        # 0Hz (DC成分) は除外してプロット (伸縮の振動を見たい場合)
        # ※ 0Hz も見たい場合は mask = (freqs >= 0) & (freqs <= MAX_HARMONIC_TO_SHOW)
        mask = (freqs > 0) & (freqs <= MAX_HARMONIC_TO_SHOW)
        
        if not np.any(mask):
             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(segment, fontsize=10)
             continue
        
        plot_freqs = freqs[mask]
        plot_amps = amps[mask]

        ax.bar(plot_freqs, plot_amps, width=0.8, align='center')
        ax.set_title(segment, fontsize=10)
        ax.set_xlabel("Harmonic (cycles/gait cycle)", fontsize=8)
        ax.set_xticks(range(1, MAX_HARMONIC_TO_SHOW + 1)) # 1Hz から
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        if plot_amps.size > 0: # Y軸の最大値更新
             max_amp = max(max_amp, plot_amps.max())
             
        plot_count += 1

    # 使わなかったサブプロットを非表示
    for i in range(plot_count, len(axes)):
        axes[i].set_visible(False)

    # Y軸を統一
    axes[0].set_ylabel("Tension Amplitude (N)", fontsize=10)
    if max_amp > 0:
        for i in range(plot_count):
            axes[i].set_ylim(0, max_amp * 1.1) # 少し余裕を持たせる

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトルとの重なり回避
    
    # --- 保存確認 ---
    try:
        save_choice = input("\nプロットを画像として保存しますか？ (y/n): ").lower()
        if save_choice == 'y':
            save_filename = f"{task_key}_tension_fft_spectrum.png"
            save_path = os.path.join(config.RESULT_DIR, save_filename)
            os.makedirs(config.RESULT_DIR, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"グラフを保存しました: {save_path}")
        else:
            print("グラフは保存されませんでした。")
    except Exception as e:
        print(f"グラフ保存エラー: {e}")

    plt.show() # 保存後または保存しない場合に表示


def main():
    """メイン実行関数"""
    # 1. タスク選択
    while True:
        task_key = input("解析するタスク名を入力 (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS.get(task_key); break
        else: print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")
    if cfg is None: print(f"エラー: {task_key} の設定読み込み失敗。"); return

    print(f"\n--- {task_key} の張力FFT解析を開始します ---")

    # 2. データ読み込み
    df_tension = load_tension_data(cfg)
    if df_tension is None:
        print("処理を終了します。"); return

    # 3. FFT実行
    segments = sorted(df_tension['segment'].unique())
    fft_results = {} # {segment: (freqs, amps)}
    
    for segment in segments:
        signal_df = df_tension[df_tension['segment'] == segment].sort_values('gait_cycle_%')
        signal = signal_df['tension_N'].values
        
        if len(signal) < 10: # データが短すぎる場合
            print(f"警告: セグメント '{segment}' のデータが短すぎます ({len(signal)}点)。FFTをスキップします。")
            fft_results[segment] = (None, None)
            continue
            
        # 周期データ(例: 101点)であることを確認
        if not (len(signal) > 100 and np.isclose(signal_df['gait_cycle_%'].iloc[0], 0.0) and np.isclose(signal_df['gait_cycle_%'].iloc[-1], 100.0)):
             print(f"警告: セグメント '{segment}' は 0-100% の平均化データではないようです。FFT結果の解釈に注意してください。 (N={len(signal)})")
             
        freqs, amps = perform_fft(signal)
        fft_results[segment] = (freqs, amps)
        
    print("FFT計算完了。")

    # 4. プロット
    plot_fft_results(fft_results, task_key)
    
    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()