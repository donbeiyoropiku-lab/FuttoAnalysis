#optitrackで取得したデータをそのまま3d可視化するためのプログラム
#マーカーの個数の増大や外れ値を確認
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

# --- ▼▼▼ 設定項目 ▼▼▼ ---

# ★ 1. 読み込むCSVファイルのパスを指定
YOUR_CSV_FILE_PATH = r"C:\FuttoAnalysis\opti\20260217\task03.csv"

# ★ 2. アニメーション化したい時間の範囲（秒）を指定
START_TIME_SECONDS = 100# (例: 15秒から)
END_TIME_SECONDS = 110 # (例: 20秒まで)

# ★ 3. 再生速度（1が等倍速。5 を指定すると 5倍速）
PLAYBACK_SPEED_FACTOR = 1
# --- ▲▲▲ 設定はここまで ▲▲▲ ---


def load_special_csv_to_long_format(file_path):
    """
    行ごとに列数が異なる可能性のあるOptiTrack CSVを頑健に読み込む関数。
    """
    print(f"特殊形式CSVの読み込みを開始します: {file_path}")
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
        return pd.DataFrame()

    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダーをスキップ
            for line in f:
                parts = line.strip().split(',')
                try:
                    if len(parts) < 5: continue
                    frame, t, n = int(parts[1]), float(parts[2]), int(parts[4])
                    base = 5
                    if len(parts) >= base + n * 4:
                        for i in range(n):
                            # OptiTrackの標準的な順序 (X, Y, Z) で読み込む
                            x = float(parts[base + 4 * i + 0])
                            y = float(parts[base + 4 * i + 1])
                            z = float(parts[base + 4 * i + 2])
                            marker_id = int(parts[base + 4 * i + 3])
                            rows.append((frame, t, marker_id, x * 1000.0, y * 1000.0, z * 1000.0))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return pd.DataFrame()

    if not rows:
        print("エラー: CSVから有効なデータを1行も読み込めませんでした。")
        return pd.DataFrame()

    final_df = pd.DataFrame(rows, columns=["Frame", "Time", "ID", "X", "Y", "Z"])
    print("✅ データの読み込みと整形が完了しました。")
    return final_df


# --- メインの処理 ---
if __name__ == "__main__":
    
    # 1. データの読み込み
    df_3d_all = load_special_csv_to_long_format(YOUR_CSV_FILE_PATH)

    if df_3d_all.empty:
        print("\nデータ読み込みに失敗したため、プログラムを終了します。")
        sys.exit()

    # 2. 指定した時間範囲でデータをフィルタリング
    print(f"データを {START_TIME_SECONDS} 秒から {END_TIME_SECONDS} 秒の範囲でフィルタリングします。")
    df_filtered = df_3d_all[
        (df_3d_all['Time'] >= START_TIME_SECONDS) & 
        (df_3d_all['Time'] <= END_TIME_SECONDS)
    ].copy()

    if df_filtered.empty:
        print(f"エラー: 指定された時間範囲 ({START_TIME_SECONDS}s - {END_TIME_SECONDS}s) にデータがありません。")
        sys.exit()

    print("✅ データフィルタリングが完了しました。")


    # 3. 3Dアニメーションの作成
    print("3Dアニメーションの作成を開始します...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    marker_ids = sorted(df_filtered['ID'].unique())
    cmap = plt.cm.get_cmap('gist_rainbow', len(marker_ids))
    colors = {mid: cmap(i) for i, mid in enumerate(marker_ids)}
    plots = {mid: ax.plot([], [], [], marker='o', color=colors[mid], label=f'ID {mid}')[0] for mid in marker_ids}
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    def init():
        ax.set_title(f'3D Trajectory Animation ({START_TIME_SECONDS:.1f}s to {END_TIME_SECONDS:.1f}s)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        all_coords = df_filtered[['X', 'Y', 'Z']].values
        x_coords, y_coords, z_coords = all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]
        
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        z_range = z_coords.max() - z_coords.min()
        max_range = np.array([x_range, y_range, z_range]).max() * 1.1

        mid_x = (x_coords.max() + x_coords.min()) / 2
        mid_y = (y_coords.max() + y_coords.min()) / 2
        mid_z = (z_coords.max() + z_coords.min()) / 2
        
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        
        ax.legend(title="Marker IDs", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        return list(plots.values()) + [time_text]

    def update(frame_num):
        current_frame_data = df_filtered[df_filtered['Frame'] == frame_num]
        
        if not current_frame_data.empty:
            current_time = current_frame_data['Time'].iloc[0]
            time_text.set_text(f'Time: {current_time:.2f} s')
            
            for marker_id in marker_ids:
                marker_pos = current_frame_data[current_frame_data['ID'] == marker_id]
                if not marker_pos.empty:
                    pos = marker_pos.iloc[0]
                    plots[marker_id].set_data([pos['X']], [pos['Y']])
                    plots[marker_id].set_3d_properties([pos['Z']])
                else:
                    plots[marker_id].set_data([], [])
                    plots[marker_id].set_3d_properties([])

        return list(plots.values()) + [time_text]

    # 4. アニメーションの実行
    # 4. アニメーションの実行
    frames_to_animate = sorted(df_filtered['Frame'].unique())

    if not frames_to_animate:
        print("エラー: フィルタリング後にアニメーションするフレームがありません。")
        sys.exit()

    # --- ▼▼▼ 変更点 ▼▼▼ ---
    # PLAYBACK_SPEED_FACTOR に応じてフレームを間引く
    try:
        speed_factor = int(PLAYBACK_SPEED_FACTOR)
        if speed_factor < 1: speed_factor = 1
    except (ValueError, NameError):
            speed_factor = 1 # 変数が未定義か数字でない場合は等倍速

    # [::speed_factor] で、リストから speed_factor 個おきに要素を取得
    frames_skipped = frames_to_animate[::speed_factor]

    if not frames_skipped:
        print("エラー: フレーム間引き後にアニメーションするフレームがありません。")
        sys.exit()

    # print文も変更後のフレーム数を表示するように更新
    print(f"アニメーションの総フレーム数: {len(frames_to_animate)} (間引き後: {len(frames_skipped)})")

    # frames引数に間引いたリスト(frames_skipped)を渡す
    # intervalは1フレームあたりのミリ秒。1000 / (フレームレート / 再生速度) で計算
    # 元のデータが約100Hzと仮定して、適切な間隔を設定
    interval_ms = 1000 / (100 / PLAYBACK_SPEED_FACTOR)
    ani = animation.FuncAnimation(fig, update, frames=frames_skipped, init_func=init, blit=False, interval=interval_ms)
    # --- ▲▲▲ 変更ここまで ▲▲▲ ---
    
    # 5. 表示または保存
    while True:
        action = input("\n操作を選択してください (1: 表示, 2: GIF保存): ").lower()
        if action in ['1', '2']:
            break
        print("無効な入力です。'1' または '2' を入力してください。")

    if action == '1':
        print("アニメーションを表示します...")
        plt.show()
        print("アニメーションを終了しました。")
    elif action == '2':
        base_name = os.path.splitext(os.path.basename(YOUR_CSV_FILE_PATH))[0]
        save_filename = f"{base_name}_{START_TIME_SECONDS:.1f}s-{END_TIME_SECONDS:.1f}s.gif"
        save_path = os.path.join(os.path.dirname(YOUR_CSV_FILE_PATH), save_filename)
        
        try:
            writer = animation.PillowWriter(fps=int(100 / PLAYBACK_SPEED_FACTOR)) # 保存するGIFのフレームレート
            print(f"\nアニメーションを保存中: {save_path}")
            print("これには数分かかる場合があります...")
            ani.save(save_path, writer=writer)
            print(f"✅ アニメーションの保存が完了しました: {save_path}")
        except Exception as e:
            print(f"エラー: アニメーションの保存に失敗しました。詳細: {e}")
            print("Pillowがインストールされているか確認してください (`pip install Pillow`)。")


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os # osモジュールを追加

# --- ▼▼▼ 設定項目 ▼▼▼ ---

# ★ 1. 読み込むCSVファイルのパスを指定
YOUR_CSV_FILE_PATH = r"C:\FuttoAnalysis\opti\20251020\task2.csv"

# ★ 2. アニメーション化したい時間の範囲（秒）を指定
START_TIME_SECONDS = 35.0  # (例: 5秒から)
END_TIME_SECONDS = 40.0 # (例: 10秒まで)

# --- ▲▲▲ 設定はここまで ▲▲▲ ---


def load_special_csv_to_long_format(file_path):
    """
    OptiTrackなどが出力する特殊な形式のCSVファイルを読み込み、
    扱いやすい「ロングフォーマット」のDataFrameに変換する関数。
    """
    print(f"特殊形式CSVの読み込みを開始します: {file_path}")
    
    # ファイルの存在確認
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
        return pd.DataFrame() # 空のデータフレームを返す

    try:
        # 実際のデータが始まるまで43行をスキップし、ヘッダーはないものとして読み込む
        df_raw = pd.read_csv(file_path, skiprows=43, header=None)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        return pd.DataFrame()

    processed_rows = []
    # 1行ずつデータを解析
    for index, row in df_raw.iterrows():
        try:
            # 必要な基本情報を抽出
            frame = int(row.iloc[1])
            time = float(row.iloc[2])
            marker_count = int(row.iloc[4]) # このフレームで検出されたマーカーの数
            
            base_index = 5 # マーカーデータの開始列インデックス
            
            # 1行に含まれる全てのマーカーデータを抽出
            for i in range(marker_count):
                # 各マーカーの X, Y, Z 座標と ID を抽出
                x = float(row.iloc[base_index + 4 * i + 2])
                y = float(row.iloc[base_index + 4 * i])
                z = float(row.iloc[base_index + 4 * i + 1])
                marker_id = int(row.iloc[base_index + 4 * i + 3])
                
                # 単位をメートルからミリメートルに変換し、リストに追加
                processed_rows.append((frame, time, marker_id, x * 1000.0, y * 1000.0, z * 1000.0))

        except (ValueError, IndexError):
            # 行のデータが不正で変換に失敗した場合はスキップ
            # print(f"警告: {index+44}行目のデータ形式が不正なためスキップしました。")
            continue
            
    if not processed_rows:
        print("エラー: CSVから有効なデータを1行も読み込めませんでした。")
        return pd.DataFrame()

    # 最後にリストをまとめてDataFrameに変換
    final_df = pd.DataFrame(processed_rows, columns=["Frame", "Time", "ID", "X", "Y", "Z"])
    print("✅ データの読み込みと整形が完了しました。")
    return final_df


# --- メインの処理 ---
if __name__ == "__main__":
    
    # 1. データの読み込み
    df_3d_all = load_special_csv_to_long_format(YOUR_CSV_FILE_PATH)

    # 読み込みに失敗した場合はプログラムを終了
    if df_3d_all.empty:
        print("\nデータ読み込みに失敗したため、プログラムを終了します。")
        sys.exit() # スクリプトを停止

    # 2. ★★★ 指定した時間範囲でデータをフィルタリング ★★★
    print(f"データを {START_TIME_SECONDS} 秒から {END_TIME_SECONDS} 秒の範囲でフィルタリングします。")
    df_filtered = df_3d_all[
        (df_3d_all['Time'] >= START_TIME_SECONDS) & 
        (df_3d_all['Time'] <= END_TIME_SECONDS)
    ].copy()

    if df_filtered.empty:
        print(f"エラー: 指定された時間範囲 ({START_TIME_SECONDS}s - {END_TIME_SECONDS}s) にデータがありません。")
        print("時間範囲またはファイルパスを確認してください。")
        sys.exit()

    print("✅ データフィルタリングが完了しました。")


    # 3. 3Dアニメーションの作成
    print("3Dアニメーションの作成を開始します...")
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # フィルタリング後のデータに存在するマーカーIDのみを対象にする
    marker_ids = df_filtered['ID'].unique()
    plots = {mid: ax.plot([], [], [], marker='o', label=f'ID {mid}')[0] for mid in marker_ids}
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def init():
        ax.set_title(f'3D Trajectory Animation ({START_TIME_SECONDS:.1f}s to {END_TIME_SECONDS:.1f}s)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # データの範囲はフィルタリング後のデータから計算
        x_min, x_max = df_filtered['X'].min(), df_filtered['X'].max()
        y_min, y_max = df_filtered['Y'].min(), df_filtered['Y'].max()
        z_min, z_max = df_filtered['Z'].min(), df_filtered['Z'].max()
        
        # 軸の範囲に少しマージンを持たせる
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        z_margin = (z_max - z_min) * 0.1
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_zlim(z_min - z_margin, z_max + z_margin)
        
        ax.legend()
        ax.set_aspect('equal', adjustable='box') # 各軸のスケールを揃える
        return list(plots.values()) + [time_text]

    def update(frame_num):
        # ★ 現在のフレームに対応するデータを「フィルタリング後のデータ」から取得
        current_frame_data = df_filtered[df_filtered['Frame'] == frame_num]
        
        if not current_frame_data.empty:
            # 時間表示を更新
            current_time = current_frame_data['Time'].iloc[0]
            time_text.set_text(f'Time: {current_time:.2f} s')
            
            # 各マーカーの位置を更新
            for marker_id in marker_ids:
                marker_pos = current_frame_data[current_frame_data['ID'] == marker_id]
                if not marker_pos.empty:
                    pos = marker_pos.iloc[0]
                    plots[marker_id].set_data([pos['X']], [pos['Y']])
                    plots[marker_id].set_3d_properties([pos['Z']])
                else:
                    # このフレームにマーカーがない場合は非表示
                    plots[marker_id].set_data([], [])
                    plots[marker_id].set_3d_properties([])

        return list(plots.values()) + [time_text]

    # 4. ★ アニメーションの実行（フィルタリング後のフレームリストを使用）
    
    # アニメーション化するフレーム番号のリスト（重複なし・昇順）
    frames_to_animate = sorted(df_filtered['Frame'].unique())
    
    if not frames_to_animate:
        print("エラー: フィルタリング後にアニメーションするフレームがありません。")
        sys.exit()

    print(f"アニメーションの総フレーム数: {len(frames_to_animate)}")
    print(f"（元のフレーム {frames_to_animate[0]} から {frames_to_animate[-1]} まで）")

    # FuncAnimation には、実際のフレーム番号のリストを渡す
    ani = animation.FuncAnimation(fig, update, frames=frames_to_animate, init_func=init, blit=False, interval=10)
    
    plt.show()
    print("アニメーションを終了しました。")
'''
