# =============================================================================
# calculate_rubber_torque.py (v3.3 - 可読性向上版)
#
# 目的:
#   平均化されたマーカーデータ (mean_cycle) とゴム張力データ (tension_data) から、
#   各関節（Hip, Knee, Ankle）にかかるゴムのトルクを計算し、グラフ化する。
#
# 処理フロー:
# 1. config.py からタスク設定 (task1, task2, or task3) を読み込む。
# 2. 該当タスクの平均化マーカーデータと張力データをロードする。
# 3. 座標系を統一する (unify_coordinate_system):
#    - create_anime_grad_D.py の出力はタスクによって座標の意味が異なるため、
#      全タスクを (X: 逆進行, Y: 左右, Z: 上下) の統一座標系に変換する。
# 4. トルクを計算する (calculate_all_torques):
#    - 統一座標系において、張力ベクトルとレバーアームベクトルの外積 (τ = r x F)
#      から、各関節の3軸トルク (Nm) を計算する。
# 5. トルクデータを保存する (..._torque_data.csv)。
# 6. 結果をグラフ化する (plot_joint_torques):
#    - 関節ごと (Hip, Knee, Ankle の順) に3軸トルクの平均±SDをプロット。
#    - グラフ下部に、歩行周期10%刻みのスティック図 (進行方向) を描画する。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config # 設定ファイルをインポート

# =============================================================================
# データ読み込み・座標系統一
# =============================================================================

def load_data(cfg: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    config設定に基づき、平均化マーカーデータと張力データを読み込む。
    _ranged (モード3) のCSVがあればそちらを優先する。

    Args:
        cfg (dict): config.TASK_CONFIGS[task_key] の設定辞書

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (df_mean_cycle, df_tension)
    """
    mean_csv_path_ranged = cfg.get('MEAN_CYCLE_RANGED_OUTPUT_PATH')
    mean_csv_path_all = cfg.get('MEAN_CYCLE_OUTPUT_PATH')
    mean_csv_path = None
    
    if mean_csv_path_ranged and os.path.exists(mean_csv_path_ranged):
        mean_csv_path = mean_csv_path_ranged
    elif mean_csv_path_all and os.path.exists(mean_csv_path_all):
        mean_csv_path = mean_csv_path_all
    else:
        print(f"エラー: 平均化データファイルが見つかりません。")
        if mean_csv_path_ranged: print(f"  (試行1: {mean_csv_path_ranged})")
        if mean_csv_path_all: print(f"  (試行2: {mean_csv_path_all})")
        return None, None
        
    tension_csv_path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not tension_csv_path or not os.path.exists(tension_csv_path):
        print(f"エラー: 張力データファイルが見つかりません: {tension_csv_path}")
        return None, None

    try:
        df_mean_cycle = pd.read_csv(mean_csv_path)
        print(f"平均化データを読み込みました: {mean_csv_path}")
    except Exception as e:
        print(f"平均化データの読み込みエラー: {e}"); return None, None
        
    try:
        df_tension = pd.read_csv(tension_csv_path)
        print(f"張力データを読み込みました: {tension_csv_path}")
    except Exception as e:
        print(f"張力データの読み込みエラー: {e}"); return None, None

    return df_mean_cycle, df_tension

def unify_coordinate_system(df_mean_cycle: pd.DataFrame, task_key: str) -> pd.DataFrame | None:
    """
    create_anime_grad_D.py が出力したマーカー座標を、
    統一座標系 (X:進行逆, Y:左右, Z:上下) に変換する。

    Args:
        df_mean_cycle (pd.DataFrame): create_anime_grad_D.py の出力CSV
        task_key (str): 'task1', 'task2', 'task3'

    Returns:
        pd.DataFrame | None: 統一座標系に変換されたDataFrame
    """
    print(f"'{task_key}' の座標系を統一中 (X:進行逆, Y:左右, Z:上下)...")
    df_unified = df_mean_cycle.copy()
    
    if 'x' not in df_mean_cycle.columns:
        print("エラー: 入力データに 'x' 列がありません。")
        return None

    if task_key == 'task1':
        # task1入力: (x=左右, y=進行, z=垂直)
        print("  -> Task1: X_final = -y_mid, Y_final = x_mid, Z_final = z_mid")
        df_unified['x_new'] = -df_mean_cycle['y'] # 逆進行
        df_unified['y_new'] =  df_mean_cycle['x'] # 左右
        df_unified['z_new'] =  df_mean_cycle['z'] # 上下
    elif task_key in ['task2', 'task3']:
        # task2/3入力: (x=逆進行, y=左右, z=垂直)
        print("  -> Task2/3: 座標系は既に統一済み (x, y, z をそのまま使用)。")
        df_unified['x_new'] =  df_mean_cycle['x'] # 逆進行
        df_unified['y_new'] =  df_mean_cycle['y'] # 左右
        df_unified['z_new'] =  df_mean_cycle['z'] # 上下
    else:
        print(f"警告: 不明なタスクキー '{task_key}'。座標系変換をスキップします。")
        return df_mean_cycle

    # 元の列を削除し、新しい列名に変更
    df_unified = df_unified.drop(columns=['x', 'y', 'z'])
    df_unified = df_unified.rename(columns={'x_new': 'x', 'y_new': 'y', 'z_new': 'z'})
    return df_unified

# =============================================================================
# トルク計算関数
# =============================================================================

def _calculate_joint_center(joint_def: dict, positions: dict) -> np.ndarray:
    """単一の関節中心を計算するヘルパー関数"""
    m_type = joint_def.get('type')
    m_ids = joint_def.get('markers')
    if not m_type or not m_ids: return np.full(3, np.nan)
    
    # 必要なIDがpositionsにあるか確認
    points = [np.array(positions[mid]) for mid in m_ids if mid in positions]
    if len(points) != len(m_ids): return np.full(3, np.nan) # マーカー不足
    
    if m_type == 'single': return points[0]
    if m_type == 'midpoint': return np.mean(points[:2], axis=0)
    if m_type == 'centroid': return np.mean(points, axis=0)
    return np.full(3, np.nan)

def calculate_all_torques(df_mean_cycle: pd.DataFrame, tension_df: pd.DataFrame, cfg: dict, task_key: str) -> pd.DataFrame | None:
    """
    統一座標系 (X:逆進行, Y:左右, Z:上下) で、各関節にかかるゴムのトルクを計算する。
    """
    print("ゴムによる関節トルクの計算を開始...")
    joint_center_defs = cfg.get('JOINT_CENTER_DEFS')
    rubber_torque_map = cfg.get('RUBBER_TORQUE_MAP')
    if not joint_center_defs or not rubber_torque_map:
        print("エラー: configに 'JOINT_CENTER_DEFS' または 'RUBBER_TORQUE_MAP' が未定義です。")
        return None
    try:
        # 張力データをピボット (Index: 周期%, Columns: セグメント名, Values: 張力N)
        tension_pivot = tension_df.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
    except Exception as e:
         print(f"張力データのピボット失敗: {e}"); return None

    torque_records = []
    gait_cycle_perc_list = sorted(df_mean_cycle['gait_cycle_%'].unique())
    if not gait_cycle_perc_list: print("エラー: gait_cycle_% が見つかりません。"); return None
    
    # 浮動小数点数の比較誤差を避けるため、丸めてグループ化
    grouped = df_mean_cycle.groupby(df_mean_cycle['gait_cycle_%'].apply(lambda x: np.round(x, 5)))
    
    for cycle_perc in gait_cycle_perc_list:
        try: 
            current_data = grouped.get_group(np.round(cycle_perc, 5))
        except KeyError: 
            continue # その%のデータがない場合はスキップ
             
        # その周期%での全マーカーの絶対座標 {id: (x,y,z)}
        positions = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()}
        # その周期%での全関節中心の絶対座標 {name: (x,y,z)}
        joint_centers = {name: _calculate_joint_center(joint_def, positions) for name, joint_def in joint_center_defs.items()}
        
        # configのトルクマップに従って各ゴムのトルクを計算
        for segment_name, (joint_name, p_attach_id, p_origin_id) in rubber_torque_map.items():
            try:
                F_magnitude = tension_pivot.loc[cycle_perc, segment_name]
                P_attach = positions.get(p_attach_id)
                P_origin = positions.get(p_origin_id)
                Joint_center = joint_centers.get(joint_name)

                # データ欠損チェック
                if pd.isna(F_magnitude) or P_attach is None or P_origin is None or Joint_center is None or np.isnan(Joint_center).any():
                    continue
                
                P_attach, P_origin = np.array(P_attach), np.array(P_origin)
                
                # 力ベクトル (F_vec)
                vec_direction = P_origin - P_attach
                norm = np.linalg.norm(vec_direction)
                if norm < 1e-6: continue # ゼロベクトル回避
                F_vec = (vec_direction / norm) * F_magnitude
                
                # レバーアームベクトル (r_vec)
                r_vec = P_attach - Joint_center
                
                # トルク (τ = r x F) (単位: N*mm)
                torque_vec = np.cross(r_vec, F_vec)
                
                # N*mm -> N*m に単位換算
                torque_vec_Nm = torque_vec / 1000.0
                
                torque_records.append({
                    "gait_cycle_%": cycle_perc, "segment": segment_name, "joint": joint_name,
                    "torque_x_Nm": torque_vec_Nm[0], # X軸回り (内/外転)
                    "torque_y_Nm": torque_vec_Nm[1], # Y軸回り (屈曲/伸展)
                    "torque_z_Nm": torque_vec_Nm[2]  # Z軸回り (内/外旋)
                })

            except (KeyError, IndexError, TypeError): 
                 continue # ループ内のエラーはスキップして継続
                 
    if not torque_records:
        print("警告: トルク計算結果が0件です。config定義を確認してください。"); return None
        
    print("ゴムトルクの計算完了。")
    return pd.DataFrame(torque_records)

# =============================================================================
# 可視化関数
# =============================================================================

def plot_stick_figure(ax: plt.Axes, df_mean_cycle_unified: pd.DataFrame, cfg: dict):
    """
    XZ平面 (サジタル面) のスティック図を10%刻みで描画する。
    描画時はX座標(逆進行)を反転(-X)し、左から右へ前向きに歩くように見せる。
    """
    print("スティック図を生成中...")
    
    # --- X座標を反転 (トルク計算とは独立した描画用処理) ---
    df_plot = df_mean_cycle_unified.copy()
    df_plot['x'] = -df_plot['x'] # X(逆進行) -> X(進行) に反転
    
    try:
        joint_defs = cfg.get('JOINT_CENTER_DEFS', {})
        foot_marker_ids = cfg.get('SEGMENTS', {}).get('Foot', [])
        if not foot_marker_ids:
             print("警告: config に 'SEGMENTS' -> 'Foot' が未定義。つま先なしで描画します。")
        
        # 描画する体節 (関節中心キーを使用)
        bones = [('Hip', 'Knee'), ('Knee', 'Ankle')]
        if foot_marker_ids:
            bones.append(('Ankle', 'Toe')) # つま先を追加

        target_percs = np.linspace(0, 100, 11) # 0, 10, ..., 100
        
        # X軸方向のオフセットを計算
        all_x_coords = df_plot['x']
        x_min, x_max = all_x_coords.min(), all_x_coords.max()
        total_width = (x_max - x_min) * 1.2 # 1周期の移動幅
        offsets = np.linspace(0, total_width * (len(target_percs) - 1), len(target_percs)) # 0, 1200, 2400...

        ax.set_title('Sagittal Plane Stick Figure (10% intervals)')
        ax.set_xlabel('Horizontal Position (Progression) (mm)') # X軸は進行方向
        ax.set_ylabel('Z (Vertical) (mm)') # Y軸はZ (上下)
        ax.set_aspect('equal')

        grouped = df_plot.groupby(df_plot['gait_cycle_%'].apply(lambda x: np.round(x, 5)))

        all_plot_x, all_plot_z = [], [] # 軸範囲設定用
        
        # 0% の Hip X 座標を取得 (オフセット基準用)
        base_hip_x = 0
        try:
            base_data = grouped.get_group(np.round(0.0, 5))
            base_pos = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in base_data.iterrows()}
            base_centers = {name: _calculate_joint_center(joint_def, base_pos) for name, joint_def in joint_defs.items()}
            base_hip_x = base_centers.get('Hip', np.zeros(3))[0] # 0%のHip X(進行)座標
        except KeyError:
            print("警告: 0% のデータが見つからないため、スティック図のXオフセットが不正確な可能性があります。")

        for i, cycle_perc in enumerate(target_percs):
            try: current_data = grouped.get_group(np.round(cycle_perc, 5))
            except KeyError: continue
            
            positions = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()}
            joint_centers = {name: _calculate_joint_center(joint_def, positions) for name, joint_def in joint_defs.items()}
            
            if foot_marker_ids:
                foot_points = [positions[mid] for mid in foot_marker_ids if mid in positions]
                joint_centers['Toe'] = np.mean(foot_points, axis=0) if foot_points else np.full(3, np.nan)
            
            # Xオフセット: 各周期のHip位置を基準0%Hip位置に揃え、そこから等間隔オフセット
            current_hip_x = joint_centers.get('Hip', np.zeros(3))[0]
            x_shift = (current_hip_x - base_hip_x) # 0%からの相対移動量
            offset_vec = np.array([offsets[i] - x_shift, 0, 0])
            
            for (p1_name, p2_name) in bones:
                p1, p2 = joint_centers.get(p1_name), joint_centers.get(p2_name)
                if p1 is not None and p2 is not None and not np.isnan(p1).any() and not np.isnan(p2).any():
                    p1_shifted, p2_shifted = p1 + offset_vec, p2 + offset_vec
                    line_x = [p1_shifted[0], p2_shifted[0]]
                    line_z = [p1_shifted[2], p2_shifted[2]] # Z軸(上下)を描画
                    ax.plot(line_x, line_z, color='black', marker='o', markersize=3)
                    all_plot_x.extend(line_x); all_plot_z.extend(line_z)
            
            p_hip = joint_centers.get('Hip')
            if p_hip is not None and not np.isnan(p_hip).any():
                 ax.text(p_hip[0] + offset_vec[0], p_hip[2] + 50, f"{int(cycle_perc)}%", ha='center', fontsize=9)

        # 軸範囲を自動調整
        if all_plot_z:
            z_min = min(np.min(all_plot_z), 0)
            z_max = np.max(all_plot_z) * 1.1 + 50
            ax.set_ylim(z_min, z_max)
        if all_plot_x:
            ax.set_xlim(np.min(all_plot_x) - 50, np.max(all_plot_x) + 50)

        # X軸の反転は不要 (描画用データ df_plot['x'] = -df_plot['x'] で対応済み)
        print("スティック図生成完了。")
    except Exception as e:
        print(f"スティック図の描画中にエラーが発生しました: {e}")

def plot_joint_torques(torque_df: pd.DataFrame, task_key: str, cfg: dict, df_mean_cycle_unified: pd.DataFrame):
    """
    関節ごと、軸ごとにゴムトルクの合計をグラフ化し、スティック図を追加する。
    """
    if torque_df is None or torque_df.empty:
        print("トルクデータがないため、グラフを描画できません。"); return
    print("ゴムトルクのグラフを生成中...")
    
    total_torque_df = torque_df.groupby(['gait_cycle_%', 'joint'])[['torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm']].sum().reset_index()
    
    # configの定義順 (Hip, Knee, Ankle)
    defined_joints = list(cfg.get('JOINT_CENTER_DEFS', {}).keys())
    joints_in_data = total_torque_df['joint'].unique()
    joints = [j for j in defined_joints if j in joints_in_data] # 定義順にソート
    if not joints: print("エラー: プロット対象の関節データがありません。"); return

    # 統一座標系 (X:逆進行, Y:左右, Z:上下) に基づく軸ラベル
    highlight_axis = 'torque_y_Nm' # Y軸回り (屈曲/伸展)
    highlight_label = 'Flexion(+)/Extension(-) Torque'
    other_axes = {
        'torque_x_Nm': 'Adduction(+)/Abduction(-) Torque', # X軸回り
        'torque_z_Nm': 'Internal(+)/External(-) Rotation Torque', # Z軸回り
    }
    
    num_joints = len(joints)
    height_ratios = [3] * num_joints + [2] # トルク:3, スティック:2 の高さ比
    
    fig, axes = plt.subplots(
        num_joints + 1, 1, 
        figsize=(12, 4 * num_joints + 3),
        sharex=False, # ★ X軸の共有は解除
        gridspec_kw={'height_ratios': height_ratios}
    )
    axes_list = np.atleast_1d(axes)
    ax_torque = axes_list[:-1] # トルク用 (Hip, Knee, Ankle)
    ax_stick = axes_list[-1]  # スティック図用
        
    fig.suptitle(f"Total Rubber Torque at Joints (Task: {task_key}, Coords: X=AP_inv, Y=ML, Z=V)", fontsize=16)

    # トルクグラフ間でのみX軸を共有
    if len(ax_torque) > 1:
        for i in range(len(ax_torque) - 1):
            ax_torque[i].sharex(ax_torque[i+1])

    # トルクグラフを描画
    for i, (ax, joint_name) in enumerate(zip(ax_torque, joints)):
        joint_data = total_torque_df[total_torque_df['joint'] == joint_name]
        
        if highlight_axis in joint_data.columns:
            ax.plot(joint_data['gait_cycle_%'], joint_data[highlight_axis], 
                    label=highlight_label, linewidth=3.0, alpha=1.0, color='red')
        for axis_col, label in other_axes.items():
            if axis_col in joint_data.columns:
                 ax.plot(joint_data['gait_cycle_%'], joint_data[axis_col], 
                         label=label, linewidth=1.5, alpha=0.7, linestyle='--')
        
        ax.set_title(f"Joint: {joint_name}")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True); ax.legend(); ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlim(0, 100) # X軸範囲を 0-100% に固定
        
        # 最後のトルクグラフ以外は X軸ラベルを非表示
        if i < len(ax_torque) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    # 最後のトルクグラフにのみ X軸ラベル
    ax_torque[-1].set_xlabel("Gait Cycle (%)")
    
    # スティック図を描画
    try:
        plot_stick_figure(ax_stick, df_mean_cycle_unified, cfg)
    except Exception as e:
        print(f"スティック図の描画中にエラーが発生しました: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存確認
    try:
        save_choice = input("\nトルクグラフを画像として保存しますか？ (y/n): ").lower()
        if save_choice == 'y':
            save_filename = f"{task_key}_torque_graph_with_stick.png"
            save_path = os.path.join(config.RESULT_DIR, save_filename)
            os.makedirs(config.RESULT_DIR, exist_ok=True)
            fig.savefig(save_path, dpi=150); print(f"グラフを保存しました: {save_path}")
        else: print("グラフは保存されませんでした。")
    except Exception as e: print(f"グラフ保存エラー: {e}")
        
    plt.show()
    print("トルクグラフ生成完了。")

# =============================================================================
# メイン実行ブロック
# =============================================================================
def main():
    """メイン実行関数"""
    while True:
        task_key = input("解析するタスク名を入力してください (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS.get(task_key); break
        else: print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")
    if cfg is None: print(f"エラー: {task_key} の設定読み込み失敗。"); return

    print(f"\n--- {task_key} のゴムトルク解析を開始します ---")
    
    # 1. データ読み込み
    # create_anime_grad_D.py の出力 (mean_cycle) を読み込む
    df_mean_cycle, df_tension = load_data(cfg)
    if df_mean_cycle is None or df_tension is None:
        print("エラー: データ読み込みに失敗しました。処理を終了します。"); return

    # 2. 座標系を統一
    df_mean_cycle_unified = unify_coordinate_system(df_mean_cycle, task_key)
    if df_mean_cycle_unified is None:
         print("エラー: 座標系統一に失敗しました。"); return

    # 3. トルク計算
    torque_data = calculate_all_torques(df_mean_cycle_unified, df_tension, cfg, task_key)
    
    if torque_data is not None and not torque_data.empty:
        # 4. トルクデータ保存
        try:
            output_path_base = cfg.get('TENSION_DATA_OUTPUT_PATH', f"{task_key}_torque_data.csv")
            output_path = output_path_base.replace('_tension_data.csv', '_torque_data.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torque_data.to_csv(output_path, index=False, float_format='%.6f')
            print(f"ゴムトルクデータを保存しました: {output_path}")
        except Exception as e: print(f"トルクCSVの保存エラー: {e}")

        # 5. トルクグラフ表示
        # スティック図用に統一座標系データを渡す
        plot_joint_torques(torque_data, task_key, cfg, df_mean_cycle_unified)
    else:
        print("トルク計算に失敗したか、結果が空でした。グラフ表示をスキップします。")

    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()