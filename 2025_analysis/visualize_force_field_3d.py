# =============================================================================
# visualize_force_field_3d.py (v3.0 - ボーン表示 & 矢印拡大版)
#
# 目的:
#   平均化された「歩行周期 (0-100%)」データを用いて、
#   ゴムの「瞬時エネルギー（仕事）」と「関節トルク」を3D空間上でアニメーション表示する。
#
# 修正点 (v3.0):
#   1. 関節中心 (Hip->Knee->Ankle) を結ぶボーン線を描画に追加。
#   2. トルク矢印のスケールを拡大 (50.0 -> 150.0)。
#   3. main関数を整理。
#   4. X軸反転を廃止 (生座標を表示)。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import os
import sys
import config

# --- 設定 ---
ANIMATION_INTERVAL = 100 # ms
# ★ 変更: トルク矢印を大きくする
TORQUE_SCALE = 150.0 
WORK_SENSITIVITY = 0.05
SMOOTHING_SIGMA = 2.0 

def apply_coordinate_transform(df_mean_cycle: pd.DataFrame, task_key: str) -> pd.DataFrame:
    """task2/task3の座標をtask1と同じ向きに変換する。

    task1  : Y負 = 進行方向 (前方)
    task2/3: X負 = 進行方向 (前方)

    変換式 (X↔Y 入れ替え):
        x_new = y_old
        y_new = x_old   → X負(旧前方) が Y負(新前方) にマッピングされる
        z_new = z_old

    注意: X↔Y 入れ替えは行列式 = -1 の鏡像変換になる。
    左右が反転する場合は、どちらか一方に符号を加える (例: x_new = -y_old)。
    実データで確認しながら調整すること。

    変換後、トルクのクロス積も自動的に新座標系で計算されるため、
    全タスクで矢状面トルク (X成分) を使って表示できる。
    """
    if task_key == 'task1':
        return df_mean_cycle
    df = df_mean_cycle.copy()
    old_x = df['x'].values.copy()
    old_y = df['y'].values.copy()
    df['x'] = -old_y  # 旧Y → 新X (符号反転で左右鏡像を補正、正規90°回転)
    df['y'] = old_x   # 旧X → 新Y (旧前後 → 新前後; X負→Y負で前方が一致)
    return df


def load_data(cfg):
    """平均化マーカーデータと張力データを読み込む"""
    mean_csv_path_ranged = cfg.get('MEAN_CYCLE_RANGED_OUTPUT_PATH')
    mean_csv_path_all = cfg.get('MEAN_CYCLE_OUTPUT_PATH')
    mean_csv_path = None
    
    if mean_csv_path_ranged and os.path.exists(mean_csv_path_ranged):
        mean_csv_path = mean_csv_path_ranged
    elif mean_csv_path_all and os.path.exists(mean_csv_path_all):
        mean_csv_path = mean_csv_path_all
    else:
        print(f"エラー: 平均化データファイルが見つかりません。")
        return None, None
        
    tension_csv_path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not tension_csv_path or not os.path.exists(tension_csv_path):
        print(f"エラー: 張力データファイルが見つかりません: {tension_csv_path}")
        return None, None

    try:
        df_mean_cycle = pd.read_csv(mean_csv_path)
        print(f"平均化データを読み込みました: {os.path.basename(mean_csv_path)}")
        df_tension = pd.read_csv(tension_csv_path)
        print(f"張力データを読み込みました: {os.path.basename(tension_csv_path)}")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None, None

    return df_mean_cycle, df_tension

def calculate_physics_per_step(df_mean_cycle, df_tension, cfg):
    """歩行周期の各ステップ(%)ごとに物理量(座標, トルク, 仕事)を計算する"""
    print("物理量(トルク・仕事)を一括計算中...")
    
    joint_defs = cfg.get('JOINT_CENTER_DEFS', {})
    segment_map = cfg.get('SEGMENT_MAP', {})
    lines_def = cfg['LINES_TO_DRAW']
    
    # ピボット操作
    marker_pivot = df_mean_cycle.pivot(index='gait_cycle_%', columns='id', values=['x', 'y', 'z'])
    marker_pivot.columns = [f"{col[1]}_{col[0]}" for col in marker_pivot.columns]
    tension_pivot = df_tension.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
    
    steps = sorted(marker_pivot.index.unique())
    frames_data = []
    prev_lengths = {} 
    
    # セグメントIDリスト
    foot_ids = segment_map.get('Foot', [])
    shank_ids = segment_map.get('Shank', [])
    thigh_ids = segment_map.get('Thigh', [])

    for i, step in enumerate(steps):
        row_coord = marker_pivot.loc[step]
        row_tension = tension_pivot.loc[step] if step in tension_pivot.index else pd.Series()
        
        pos_map = {}
        for col in marker_pivot.columns:
            mid, axis = col.split('_')
            mid = int(mid)
            if mid not in pos_map: pos_map[mid] = [0,0,0]
            pos_map[mid][['x','y','z'].index(axis)] = row_coord[col]
        for k in pos_map: pos_map[k] = np.array(pos_map[k])
        
        # --- 関節中心計算 ---
        joint_centers = {}
        for jname, jdef in joint_defs.items():
            m_ids = jdef['markers']
            pts = [pos_map[m] for m in m_ids if m in pos_map]
            j_type = jdef['type']

            if not pts or len(pts) != len(m_ids): continue

            if j_type == 'single':
                joint_centers[jname] = pts[0]
            elif j_type == 'midpoint':
                joint_centers[jname] = np.mean(pts, axis=0)
            elif j_type == 'ratio_1_3_between_mids':
                if len(pts) >= 4:
                    mid1 = (pts[0] + pts[1]) / 2.0
                    mid2 = (pts[2] + pts[3]) / 2.0
                    joint_centers[jname] = (3.0 * mid1 + 1.0 * mid2) / 4.0
            elif j_type == 'mid_of_ratio_2_1':
                if len(pts) >= 4:
                    p1 = (1.0 * pts[0] + 2.0 * pts[1]) / 3.0 
                    p2 = (1.0 * pts[2] + 2.0 * pts[3]) / 3.0 
                    joint_centers[jname] = (p1 + p2) / 2.0

        # --- トルク計算 (累積方式) ---
        marker_forces = {mid: np.zeros(3) for mid in pos_map}
        for seg_name, (p1, p2) in lines_def.items():
            if seg_name not in row_tension: continue
            tension = row_tension[seg_name]
            if pd.isna(tension) or tension <= 0: continue
            if p1 not in pos_map or p2 not in pos_map: continue
            
            vec = pos_map[p2] - pos_map[p1]
            norm = np.linalg.norm(vec)
            if norm < 1e-6: continue
            
            force_on_p1 = (vec / norm) * tension
            marker_forces[p1] += force_on_p1
            force_on_p2 = (-vec / norm) * tension
            marker_forces[p2] += force_on_p2

        joint_torques = {j: np.zeros(3) for j in joint_centers}
        
        # セグメント定義に基づく集計
        if 'Ankle' in joint_centers:
            center = joint_centers['Ankle']
            for mid in foot_ids:
                if mid in pos_map:
                    joint_torques['Ankle'] += np.cross(pos_map[mid] - center, marker_forces[mid]) / 1000.0
        if 'Knee' in joint_centers:
            center = joint_centers['Knee']
            for mid in foot_ids + shank_ids:
                if mid in pos_map:
                    joint_torques['Knee'] += np.cross(pos_map[mid] - center, marker_forces[mid]) / 1000.0
        if 'Hip' in joint_centers:
            center = joint_centers['Hip']
            for mid in foot_ids + shank_ids + thigh_ids:
                if mid in pos_map:
                    joint_torques['Hip'] += np.cross(pos_map[mid] - center, marker_forces[mid]) / 1000.0

        # --- 仕事量計算 ---
        rubber_states = {}
        for seg_name, (p1, p2) in lines_def.items():
            if p1 in pos_map and p2 in pos_map:
                curr_len = np.linalg.norm(pos_map[p1] - pos_map[p2]) / 1000.0
                tension = row_tension.get(seg_name, 0)
                dL = 0.0
                if i > 0 and seg_name in prev_lengths:
                    dL = curr_len - prev_lengths[seg_name]
                prev_lengths[seg_name] = curr_len
                raw_work = tension * dL
                
                rubber_states[seg_name] = {
                    'len': curr_len, 'ten': tension, 'dL': dL,
                    'raw_work': raw_work, 'p1': pos_map[p1], 'p2': pos_map[p2]
                }

        frames_data.append({
            'step': step,
            'pos_map': pos_map,
            'joint_centers': joint_centers,
            'joint_torques': joint_torques,
            'rubber_states': rubber_states
        })
    
    # 仕事量の平滑化
    rubber_names = lines_def.keys()
    for name in rubber_names:
        work_series = []
        indices = []
        for idx, frame in enumerate(frames_data):
            if name in frame['rubber_states']:
                work_series.append(frame['rubber_states'][name]['raw_work'])
                indices.append(idx)
        if not work_series: continue
        smoothed_work = gaussian_filter1d(work_series, sigma=SMOOTHING_SIGMA, mode='wrap')
        for i, val in zip(indices, smoothed_work):
            frames_data[i]['rubber_states'][name]['smoothed_work'] = val

    return frames_data

def get_gradient_color(work_val, max_val):
    """白ベース、透明度高めの色を返す"""
    base_color = np.array([1.0, 1.0, 1.0]) 
    red_color = np.array([1.0, 0.0, 0.0])
    blue_color = np.array([0.0, 0.0, 1.0])
    
    intensity = min(abs(work_val) / max_val, 1.0)
    visual_intensity = intensity ** 0.5 

    if work_val < 0: color = (1 - visual_intensity) * base_color + visual_intensity * red_color
    else: color = (1 - visual_intensity) * base_color + visual_intensity * blue_color
    
    alpha = 0.8 + 0.2 * visual_intensity
    return tuple(color), alpha

def animate_force_field(frames_data, cfg, task_key):
    """3Dアニメーション生成"""
    print("アニメーション生成中...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    all_pos = []
    for f in frames_data:
        for p in f['pos_map'].values(): all_pos.append(p)
    all_pos = np.array(all_pos)
    
    # 軸範囲 (入力座標そのまま)
    x_vals, y_vals, z_vals = all_pos[:, 0], all_pos[:, 1], all_pos[:, 2]
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    
    max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
    mid_x, mid_y, mid_z = (x_max+x_min)*0.5, (y_max+y_min)*0.5, (z_max+z_min)*0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'{task_key} Force Field\nRed=Assist, Blue=Brake, Arrow=Torque')

    # --- 描画オブジェクト ---
    lines_def = cfg['LINES_TO_DRAW']
    rubber_lines = {name: ax.plot([], [], [], linewidth=4)[0] for name in lines_def}
    quivers = []
    scat = ax.scatter([], [], [], c='gray', s=10, alpha=0.4)
    
    # ★ 追加: 関節ボーン (Hip-Knee-Ankle)
    skeleton_line, = ax.plot([], [], [], color='black', linewidth=5, marker='o', markersize=6, zorder=10)

    def update(frame_idx):
        data = frames_data[frame_idx]
        step = data['step']
        ax.set_title(f"Gait Cycle: {int(step)}%")
        
        # 1. マーカー
        xs = [p[0] for p in data['pos_map'].values()]
        ys = [p[1] for p in data['pos_map'].values()]
        zs = [p[2] for p in data['pos_map'].values()]
        scat._offsets3d = (xs, ys, zs)
        
        # 2. ゴム
        for name, line in rubber_lines.items():
            if name in data['rubber_states']:
                state = data['rubber_states'][name]
                p1, p2 = state['p1'], state['p2']
                work_val = state.get('smoothed_work', 0.0)
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
                color, alpha = get_gradient_color(work_val, WORK_SENSITIVITY)
                line.set_color(color); line.set_alpha(alpha)

        # ★ 3. 関節ボーン更新
        jx, jy, jz = [], [], []
        # 定義順に接続 (Hip -> Knee -> Ankle)
        for jname in ['Hip', 'Knee', 'Ankle']:
            if jname in data['joint_centers']:
                pt = data['joint_centers'][jname]
                jx.append(pt[0]); jy.append(pt[1]); jz.append(pt[2])
        if jx:
            skeleton_line.set_data(jx, jy)
            skeleton_line.set_3d_properties(jz)

        # 4. トルク矢印更新
        while quivers: q = quivers.pop(); q.remove()
        for jname, torque_vec in data['joint_torques'].items():
            if jname in data['joint_centers']:
                center = data['joint_centers'][jname]
                tx, _, _ = torque_vec
                
                # 全タスク統一: 座標変換後はY負が進行方向なので矢状面トルク = X成分
                # (task2/3 は apply_coordinate_transform で X↔Y 変換済み)
                u, v, w = tx, 0, 0

                draw_mag = np.linalg.norm([u, v, w])
                if draw_mag > 0.05:
                    x, y, z = center[0], center[1], center[2]
                    # スケール適用
                    u, v, w = u*TORQUE_SCALE, v*TORQUE_SCALE, w*TORQUE_SCALE
                    q = ax.quiver(x, y, z, u, v, w, color='gold', linewidth=4, arrow_length_ratio=0.3)
                    quivers.append(q)

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=ANIMATION_INTERVAL)
    
    save_dir = os.path.join(config.RESULT_DIR, "2025", task_key)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_key}_force_field_3d_v3.mp4")
    try: ani.save(save_path, writer='ffmpeg', fps=10); print(f"保存完了: {save_path}")
    except: 
        try: ani.save(save_path.replace('.mp4','.gif'), writer='pillow', fps=10); print("GIF保存")
        except: pass
    plt.show()

# --- ★ main関数の修正 ---
def main():
    print("\n=== Force Field 3D Visualizer (v3.0) ===")
    
    # 1. タスク入力
    while True:
        try:
            task_key = input("解析するタスク名を入力 (task1, task2, or task3): ").strip().lower()
            if not task_key: continue
            if task_key in config.TASK_CONFIGS:
                cfg = config.TASK_CONFIGS[task_key]
                break
            else:
                print(f"エラー: config.py に '{task_key}' の定義が見つかりません。")
        except KeyboardInterrupt:
            print("\n終了します。")
            sys.exit()

    print(f"\n--- {task_key} の処理を開始 ---")

    # 2. データロード
    df_mean_cycle, df_tension = load_data(cfg)
    if df_mean_cycle is None or df_tension is None:
        print("エラー: データの読み込みに失敗しました。終了します。")
        return

    # task2/task3: X↔Y 入れ替えでtask1と同じ座標系に変換
    df_mean_cycle = apply_coordinate_transform(df_mean_cycle, task_key)
    if task_key != 'task1':
        print(f"座標変換を適用しました (X↔Y 入れ替え): {task_key}")

    # 3. 物理量計算
    try:
        frames_data = calculate_physics_per_step(df_mean_cycle, df_tension, cfg)
        if not frames_data:
            print("エラー: フレームデータが生成されませんでした。")
            return
    except Exception as e:
        print(f"エラー: 物理量計算中に問題が発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. アニメーション生成
    try:
        animate_force_field(frames_data, cfg, task_key)
    except Exception as e:
        print(f"エラー: アニメーション生成中に問題が発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()