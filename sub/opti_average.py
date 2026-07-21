"""
プログラム名: opti_average.py
概要:
    OptiTrackで計測したマーカー座標データと、床反力計(LabChart)から算出した歩行周期データを統合し、
    歩行動作の「3Dアニメーション」および「歩幅（Step Length）の時系列変化」を可視化するプログラムです。

主な機能:
    1. 歩幅の算出とグラフ化
       - 右脚接地 (HS) 時の「右つま先～左つま先」の距離 (青プロット)
       - 左脚接地 (HS) 時の「左つま先～右つま先」の距離 (赤プロット)
    2. 3D骨格アニメーション
       - 指定されたマーカーを結び、スティックピクチャとして再生

入力ファイル:
    1. マーカー位置データ (.csv)
       - opti_clean.py で補正・出力されたデータ (例: try1_corrected.csv)
    2. 歩行周期データ - 右足 (.csv)
       - gaitcycle_force_labchart.py の出力 (例: task2_gait_cycles_right.csv)
    3. 歩行周期データ - 左足 (.csv)
       - gaitcycle_force_labchart.py の出力 (例: task2_gait_cycles_left.csv)

設定項目 (ユーザー設定エリア):
    - MARKER_MAP: CSV内の実際の列名と、身体部位(R_Toeなど)の対応付け
    - SKELETON_LINKS: アニメーションで線をつなぐ部位の定義

依存ライブラリ:
    - pandas, numpy, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3

# ==========================================
# ▼▼▼ ユーザー設定エリア (ここで定義を変更してください) ▼▼▼
# ==========================================

# 1. ファイルパスの設定
MOCAP_FILE = "try1_corrected.csv"              # opti_clean出力ファイル
RIGHT_CYCLE_FILE = "task2_gait_cycles_right.csv" # gaitcycle_force出力 (右)
LEFT_CYCLE_FILE = "task2_gait_cycles_left.csv"   # gaitcycle_force出力 (左)

# 2. マーカーとCSV列名の対応定義
# corrected.csv のヘッダーにある実際の列名を右側に記述してください
# ※ X, Y, Z の3成分が連続していることを前提としています
# ここでは例として、一般的な命名規則を入れています。実際の名前に書き換えてください。
MARKER_MAP = {
    # 右脚
    'R_Hip':   'R_Hip',    # 右股関節
    'R_Knee':  'R_Knee',   # 右膝関節
    'R_Ankle': 'R_Ankle',  # 右足関節
    'R_Heel':  'R_Heel',   # 右かかと
    'R_Toe':   'R_Toe',    # 右つま先
    
    # 左脚
    'L_Hip':   'L_Hip',    # 左股関節
    'L_Knee':  'L_Knee',   # 左膝関節
    'L_Ankle': 'L_Ankle',  # 左足関節
    'L_Heel':  'L_Heel',   # 左かかと
    'L_Toe':   'L_Toe',    # 左つま先
}

# 3. 骨格の接続定義（アニメーション用: どの点と点を線で結ぶか）
SKELETON_LINKS = [
    ('R_Hip', 'R_Knee'),
    ('R_Knee', 'R_Ankle'),
    ('R_Ankle', 'R_Heel'),
    ('R_Ankle', 'R_Toe'),
    ('R_Heel', 'R_Toe'), # 足部を三角形にする場合
    
    ('L_Hip', 'L_Knee'),
    ('L_Knee', 'L_Ankle'),
    ('L_Ankle', 'L_Heel'),
    ('L_Ankle', 'L_Toe'),
    ('L_Heel', 'L_Toe'),
    
    ('R_Hip', 'L_Hip')   # 骨盤
]

# アニメーションの間引き（1なら全フレーム、2なら2フレームごと）
SKIP_FRAMES = 2 
# 座標軸のスケール調整（OptiTrackはY-upが多いですが、データに合わせて調整）
AXIS_LIMIT = 1500  # 表示範囲 (mm単位などを想定)

# ==========================================
# ▲▲▲ 設定エリア終了 ▲▲▲
# ==========================================

def load_mocap_data(file_path):
    """Mocapデータを読み込み、使いやすい辞書形式に変換する"""
    try:
        df = pd.read_csv(file_path)
        # 列名のクリーニング（前後の空白削除など）
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"Error loading mocap file: {e}")
        return None

def get_marker_pos(df, frame_idx, marker_name):
    """指定フレーム・マーカーの(x,y,z)座標を取得"""
    col_base = MARKER_MAP[marker_name]
    
    # 列名検索: 'Name X', 'Name Y', 'Name Z' のようなパターンを探す
    # 実際のCSVに合わせて調整が必要な場合があります
    cols = [c for c in df.columns if col_base in c]
    
    if len(cols) < 3:
        # 完全一致で探す場合 (例えば列名が "R_Toe:X" などの場合)
        x_col = f"{col_base}:X" # 区切り文字がコロンの場合
        y_col = f"{col_base}:Y"
        z_col = f"{col_base}:Z"
        
        # 区切り文字がない場合や、単純に連番の場合など、
        # CSVの中身に応じてここのロジックは微調整が必要かもしれません
        # ここでは、見つかった列の最初3つをXYZと仮定します
        pass

    # 簡易的な取得ロジック (列名にX, Y, Zが含まれていると仮定)
    x_candidates = [c for c in cols if 'X' in c.upper() or ':0' in c]
    y_candidates = [c for c in cols if 'Y' in c.upper() or ':1' in c]
    z_candidates = [c for c in cols if 'Z' in c.upper() or ':2' in c]

    if not (x_candidates and y_candidates and z_candidates):
         # 単純にその名前から始まる3列を取得
         cols_starts = [c for c in df.columns if c.startswith(col_base)]
         if len(cols_starts) >= 3:
             x_val = df.iloc[frame_idx][cols_starts[0]]
             y_val = df.iloc[frame_idx][cols_starts[1]]
             z_val = df.iloc[frame_idx][cols_starts[2]]
             return np.array([x_val, y_val, z_val])
         else:
            raise ValueError(f"Column for {marker_name} not found.")

    x_val = df.iloc[frame_idx][x_candidates[0]]
    y_val = df.iloc[frame_idx][y_candidates[0]]
    z_val = df.iloc[frame_idx][z_candidates[0]]
    
    return np.array([x_val, y_val, z_val])

def calculate_step_lengths(mocap_df, r_cycle_df, l_cycle_df):
    """歩行周期ごとの歩幅（つま先間距離）を計算"""
    step_data = []

    # --- 右脚 HS (0%) ---
    # 右つま先から左つま先までの距離
    print("Calculating Right Steps...")
    for _, row in r_cycle_df.iterrows():
        frame = int(row['hs_frame'])
        time_sec = row['hs_time']
        
        if frame >= len(mocap_df): continue

        try:
            r_toe = get_marker_pos(mocap_df, frame, 'R_Toe')
            l_toe = get_marker_pos(mocap_df, frame, 'L_Toe')
            
            # 3次元ユークリッド距離 (cm単位に換算が必要ならここで行う。データがmmなら /10)
            dist = np.linalg.norm(r_toe - l_toe) / 10.0 # mm -> cm と仮定
            
            step_data.append({
                'time': time_sec,
                'frame': frame,
                'side': 'Right HS (R->L)',
                'distance_cm': dist,
                'color': 'blue'
            })
        except Exception as e:
            print(f"Skip frame {frame}: {e}")

    # --- 左脚 HS (0%) ---
    # 左つま先から右つま先までの距離
    print("Calculating Left Steps...")
    for _, row in l_cycle_df.iterrows():
        frame = int(row['hs_frame'])
        time_sec = row['hs_time']
        
        if frame >= len(mocap_df): continue

        try:
            l_toe = get_marker_pos(mocap_df, frame, 'L_Toe')
            r_toe = get_marker_pos(mocap_df, frame, 'R_Toe')
            
            dist = np.linalg.norm(l_toe - r_toe) / 10.0 # mm -> cm と仮定
            
            step_data.append({
                'time': time_sec,
                'frame': frame,
                'side': 'Left HS (L->R)',
                'distance_cm': dist,
                'color': 'red'
            })
        except Exception as e:
            print(f"Skip frame {frame}: {e}")

    # 時系列順にソート
    step_df = pd.DataFrame(step_data).sort_values('time')
    return step_df

def update_animation(num, data, lines, title_text):
    """アニメーション更新関数"""
    frame_idx = num * SKIP_FRAMES
    
    if frame_idx >= len(data):
        return lines

    # 各関節の座標を取得
    joints = {}
    for name in MARKER_MAP.keys():
        try:
            joints[name] = get_marker_pos(data, frame_idx, name)
        except:
            return lines

    # 線分の更新
    for line, (start_joint, end_joint) in zip(lines, SKELETON_LINKS):
        p1 = joints[start_joint]
        p2 = joints[end_joint]
        
        # Matplotlib 3D plot data format
        line.set_data([p1[0], p2[0]], [p1[2], p2[2]]) # X, Z (OptiTrack Y-upの場合、Zが奥行き)
        line.set_3d_properties([p1[1], p2[1]])       # Y (高さ)
    
    title_text.set_text(f'Frame: {frame_idx}')
    return lines + [title_text]

def main():
    # 1. データ読み込み
    print("Loading data...")
    mocap_df = load_mocap_data(MOCAP_FILE)
    r_cycle_df = pd.read_csv(RIGHT_CYCLE_FILE)
    l_cycle_df = pd.read_csv(LEFT_CYCLE_FILE)
    
    if mocap_df is None: return

    # 2. 歩幅計算とプロット
    step_df = calculate_step_lengths(mocap_df, r_cycle_df, l_cycle_df)
    
    plt.figure(figsize=(10, 6))
    for side, color in [('Right HS (R->L)', 'blue'), ('Left HS (L->R)', 'red')]:
        subset = step_df[step_df['side'] == side]
        plt.plot(subset['time'], subset['distance_cm'], marker='o', linestyle='-', label=side, color=color)
    
    plt.title("Step Length over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Step Length (cm) [Distance between Toes]")
    plt.grid(True)
    plt.legend()
    
    # グラフを保存または表示（アニメーションと同時に出すとブロックされるため、まずは表示）
    print("Close the graph window to start animation...")
    plt.show()

    # 3. アニメーション作成
    print("Creating animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # 軸の設定
    ax.set_xlim3d([-AXIS_LIMIT, AXIS_LIMIT])
    ax.set_ylim3d([0, AXIS_LIMIT*2]) # Y-up 高さ
    ax.set_zlim3d([-AXIS_LIMIT, AXIS_LIMIT])
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)') # ラベルはMatplotlibの軸定義上Zだが、データ的にはYを使う箇所
    ax.set_zlabel('Y (Height)')

    # 初期化: 空のラインを作成
    lines = []
    for _ in SKELETON_LINKS:
        line, = ax.plot([], [], [], 'o-', lw=2)
        lines.append(line)
    
    title_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # アニメーション実行
    frames = len(mocap_df) // SKIP_FRAMES
    ani = FuncAnimation(fig, update_animation, frames=frames, fargs=(mocap_df, lines, title_text),
                        interval=30, blit=False)
    
    plt.show()

if __name__ == "__main__":
    main()