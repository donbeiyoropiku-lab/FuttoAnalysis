#指定したフレームにおけるid確認用のプログラム
#3dマップで表示

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go # Matplotlibの代わりに Plotly を使用

# --- ▼▼▼ 設定 ▼▼▼ ---

# 1. 生データCSVのパス
OPTITRACK_CSV_PATH = r"C:\FuttoAnalysis\opti\20260217\task03.csv"

# 2. ★【重要】確認したい時刻 (秒) ★
#TARGET_TIME = 41.83 #13.5
#TARGET_TIME = 340.98 #13.5
#TARGET_TIME = 340.68 #13.5

#TARGET_TIME = 43.5 #13.5
TARGET_TIME = 20 #13.5

#TARGET_TIME = 341.54 #13.5





# 3. 物理的な座標範囲 (ノイズ除去用)
#task1
#PLAUSIBLE_BOUNDS = {'x': (0, 1000), 'y': (0, 1100), 'z': (-200, 400)}
#task2,3
PLAUSIBLE_BOUNDS = {'x': (-400, 600), 'y': (-400, 1200), 'z': (00, 1600)}

# --- ▲▲▲ 設定ここまで ▲▲▲ ---


def load_opti_data_to_long_robust(file_path):
    """
    行ごとに列数が異なる可能性のあるOptiTrack CSVを頑健に読み込む関数。
    座標をメートル(m)からミリメートル(mm)に変換する。
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return None
    print(f"'{os.path.basename(file_path)}' を読み込んでいます...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダー行をスキップ
            for line in f:
                parts = line.strip().split(',')
                try:
                    if len(parts) < 5: continue
                    frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4])
                    base_col = 5
                    if len(parts) >= base_col + n_markers * 4:
                        for i in range(n_markers):
                            x = float(parts[base_col + 4*i]) * 1000.0
                            y = float(parts[base_col + 4*i + 1]) * 1000.0
                            z = float(parts[base_col + 4*i + 2]) * 1000.0
                            mid = int(parts[base_col + 4*i + 3])
                            rows.append((frame, t, mid, x, y, z))
                except (ValueError, IndexError): continue
        out_df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])
        if out_df.empty:
            print("エラー: データを読み込めませんでした。ファイル形式を確認してください。")
            return None
        print(f"ファイル読み込み成功: {len(out_df)} 行")
        return out_df
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に予期せぬ問題が発生しました。詳細: {e}")
        return None

def filter_plausible_markers(frame_df, bounds):
    """物理的にありえる範囲内のマーカーのみを抽出する"""
    mask = (frame_df['x'].between(*bounds['x']) &
            frame_df['y'].between(*bounds['y']) &
            frame_df['z'].between(*bounds['z']))
    return frame_df[mask]

def visualize_3d_map(df_long):
    """
    TARGET_TIME に最も近いフレームの3Dマップを Plotly でプロットする
    """
    if df_long is None or df_long.empty:
        print("データがありません。")
        return

    # 1. TARGET_TIME に最も近いフレームを検索
    target_frame_data = df_long.iloc[(df_long['Time'] - TARGET_TIME).abs().argmin()]
    target_frame_idx = int(target_frame_data['Frame'])
    actual_time = target_frame_data['Time']
    
    print(f"\nターゲット時刻 {TARGET_TIME:.3f}s に最も近いフレームを検索...")
    print(f"  -> フレーム番号: {target_frame_idx} (実時間: {actual_time:.3f}s)")

    # 2. そのフレームの全マーカーデータを取得
    frame_df = df_long[df_long['Frame'] == target_frame_idx].copy()
    
    # 3. ノイズ除去
    frame_df_filtered = filter_plausible_markers(frame_df, PLAUSIBLE_BOUNDS)
    print(f"  ノイズ除去前: {len(frame_df)} 個 -> 除去後: {len(frame_df_filtered)} 個のマーカーをプロットします。")
    
    if frame_df_filtered.empty:
        print("エラー: このフレームには描画対象のマーカーが見つかりませんでした。")
        return

    # 4. Plotly 3D散布図の準備
    fig = go.Figure()

    # 5. 散布図とテキスト（ID）を同時に描画
    fig.add_trace(go.Scatter3d(
        x=frame_df_filtered['x'],
        y=frame_df_filtered['y'],
        z=frame_df_filtered['z'],
        mode='markers+text',  # ★ マーカーとテキストを両方表示
        marker=dict(
            size=5,
            color='blue'
        ),
        text=frame_df_filtered['id'].astype(int), # ★ IDをテキストとして指定
        textposition='middle right', # テキストの位置
        textfont=dict(
            color='red',
            size=10
        )
    ))

    # 6. プロットのレイアウトと設定
    fig.update_layout(
        title=f"3D Marker Map (Raw ID)<br>Frame: {target_frame_idx} (Time: {actual_time:.3f}s)",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            
            # ★ 軸のスケールをデータに基づいて等しくする (set_aspect('equal') と同等)
            aspectmode='data' 
        ),
        margin=dict(l=0, r=0, b=0, t=40) # 余白を最小化
    )

    print("\nPlotlyの3Dマップウィンドウをブラウザで開きます。")
    print("（ブラウザが自動で開かない場合は、ターミナルに表示されるURL (例: http://127.0.0.1:...) を開いてください）")
    fig.show()


if __name__ == "__main__":
    df_long = load_opti_data_to_long_robust(OPTITRACK_CSV_PATH)
    visualize_3d_map(df_long)