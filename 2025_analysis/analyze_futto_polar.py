# =============================================================================
# analyze_futto_polar.py
#
# 目的:
#   Futtoの機械的特性・力学的特性を評価する統合ツール。
#   以下の2つの解析モードを搭載し、実行時に選択可能。
#
#   Mode 1: Work Loop Analyzer (ワークループ解析)
#     - ゴムの「伸び vs 張力」をプロットし、剛性(Stiffness)やヒステリシスを評価。
#
#   Mode 2: Polar Force Analyzer (ポーラーチャート解析)
#     - 下腿(Shank)に作用するゴムの合力ベクトル(Assist Vector)を計算し、
#       その「方向」と「強さ」を極座標グラフで可視化。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config

# --- 設定: ワークループ解析で表示するゴム ---
WORK_LOOP_TARGETS = [
    'Front_Upper_In', 'Back_Upper_In',      # Hip
    'Front_Knee_Upper_In', 'Back_Thigh_In', # Knee
    'Front_Shin', 'Back_Shin_In'            # Ankle
]

# --- 共通関数 ---
def load_data(cfg):
    mean_path = cfg.get('MEAN_CYCLE_OUTPUT_PATH')
    tension_path = cfg.get('TENSION_DATA_OUTPUT_PATH')

    if not mean_path or not os.path.exists(mean_path):
        print(f"エラー: 平均座標ファイルなし: {mean_path}")
        return None, None
    if not tension_path or not os.path.exists(tension_path):
        print(f"エラー: 張力データファイルなし: {tension_path}")
        return None, None

    return pd.read_csv(mean_path), pd.read_csv(tension_path)

# =============================================================================
# Mode 1: Work Loop Logic
# =============================================================================
def calculate_work_loops(df_coord, df_tension, cfg):
    lines_def = cfg['LINES_TO_DRAW']
    
    # Pivot
    coord_pivot = df_coord.pivot(index='gait_cycle_%', columns='id', values=['x', 'y', 'z'])
    coord_pivot.columns = [f"{col[1]}_{col[0]}" for col in coord_pivot.columns]
    tension_pivot = df_tension.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
    
    cycles = coord_pivot.index.values
    results = {}

    for name in WORK_LOOP_TARGETS:
        # 名前マッチング (In/Out含む処理)
        target_name = name
        if name not in lines_def:
            candidates = [k for k in lines_def.keys() if name in k]
            if candidates: target_name = candidates[0]
            else: continue

        if target_name not in tension_pivot.columns: continue
        p1_id, p2_id = lines_def[target_name]
        
        try:
            p1_xyz = coord_pivot[[f"{p1_id}_x", f"{p1_id}_y", f"{p1_id}_z"]].values
            p2_xyz = coord_pivot[[f"{p2_id}_x", f"{p2_id}_y", f"{p2_id}_z"]].values
            lengths = np.linalg.norm(p1_xyz - p2_xyz, axis=1) # Length
            tensions = tension_pivot[target_name].values       # Tension
            results[target_name] = {'len': lengths, 'ten': tensions}
        except KeyError: continue
            
    return results, cycles

def run_work_loop_mode(df_coord, df_tension, cfg, task_key):
    print("\n--- Running Work Loop Analysis ---")
    results, cycles = calculate_work_loops(df_coord, df_tension, cfg)
    
    if not results:
        print("描画可能なデータがありません。")
        return

    num_plots = len(results)
    cols = 3
    rows = (num_plots // cols) + (1 if num_plots % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_plots > 1: axes = axes.flatten()
    else: axes = [axes]
    
    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        L = data['len']
        T = data['ten']
        
        # Plot
        sc = ax.scatter(L, T, c=cycles, cmap='hsv', s=15, alpha=0.8)
        ax.plot(L, T, 'k-', alpha=0.3, linewidth=1)
        ax.plot(L[0], T[0], 'ko', markersize=8, markerfacecolor='white', label='Start')
        
        # Stiffness近似
        if np.max(L) - np.min(L) > 5:
            z = np.polyfit(L, T, 1)
            ax.text(0.05, 0.9, f'k ≈ {z[0]:.2f} N/mm', transform=ax.transAxes, fontsize=9, color='blue')

        ax.set_title(name)
        ax.set_xlabel('Length (mm)'); ax.set_ylabel('Tension (N)')
        ax.grid(True, linestyle='--')

    # 余白削除
    for j in range(i+1, len(axes)): fig.delaxes(axes[j])
        
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Gait Cycle (%)')
    plt.suptitle(f'Work Loops (Stiffness & Hysteresis) - {task_key}', fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.4)
    
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_work_loops.png")
    plt.savefig(save_path)
    print(f"保存完了: {save_path}")
    plt.show()

# =============================================================================
# Mode 2: Polar Force Logic
# =============================================================================
def get_shank_related_rubbers(cfg):
    seg_map = cfg.get('SEGMENT_MAP', {})
    shank_ids = set(seg_map.get('Shank', []))
    lines = cfg['LINES_TO_DRAW']
    target_rubbers = []
    
    for name, (p1, p2) in lines.items():
        p1_in = p1 in shank_ids
        p2_in = p2 in shank_ids
        # XOR: 片方だけがShank (外部からの力)
        if p1_in ^ p2_in:
            if p1_in: origin, target = p1, p2
            else:     origin, target = p2, p1
            target_rubbers.append({'name': name, 'oid': origin, 'tid': target})
    return target_rubbers

def calculate_net_force(df_coord, df_tension, rubbers):
    coord_pivot = df_coord.pivot(index='gait_cycle_%', columns='id', values=['x', 'y', 'z'])
    coord_pivot.columns = [f"{col[1]}_{col[0]}" for col in coord_pivot.columns]
    tension_pivot = df_tension.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
    
    cycles = coord_pivot.index.values
    n_points = len(cycles)
    net_force = np.zeros((n_points, 3))
    
    for item in rubbers:
        name, oid, tid = item['name'], item['oid'], item['tid']
        if name not in tension_pivot.columns: continue
        try:
            o_pos = coord_pivot[[f"{oid}_x", f"{oid}_y", f"{oid}_z"]].values
            t_pos = coord_pivot[[f"{tid}_x", f"{tid}_y", f"{tid}_z"]].values
        except KeyError: continue
            
        vec = t_pos - o_pos
        dist = np.linalg.norm(vec, axis=1)[:, np.newaxis]
        dist[dist < 1e-6] = 1.0
        unit_vec = vec / dist
        
        ten = tension_pivot[name].values
        if len(ten) != n_points:
             ten = np.interp(np.linspace(0, 100, n_points), 
                             np.linspace(0, 100, len(ten)), ten)
        
        net_force += unit_vec * ten[:, np.newaxis]
        
    return net_force, cycles

def run_polar_force_mode(df_coord, df_tension, cfg, task_key):
    print("\n--- Running Polar Force Analysis (Shank Assist) ---")
    rubbers = get_shank_related_rubbers(cfg)
    print(f"抽出されたゴム: {[r['name'] for r in rubbers]}")
    
    if not rubbers:
        print("Shankに関連するゴムが見つかりません。")
        return

    net_force, cycles = calculate_net_force(df_coord, df_tension, rubbers)
    
    # ==========================================
    # 修正後 (After)
    # ==========================================
    # 軸判定 (簡易ロジック)
    Fx, Fy, Fz = net_force[:, 0], net_force[:, 1], net_force[:, 2]
    
    
    # Progression判定
    if 'task1' in task_key:
        # ★ マイナスをつけてみる (逆向きに歩いている場合)
        P_vec = -Fz
        V_vec = -Fy # Vertical (Y-Up仮定) 
    else:
        # Task2, 3 ももし逆なら -Fx にする
        P_vec = Fx
        V_vec = Fy # Vertical (Y-Up仮定)
        
    r = np.sqrt(V_vec**2 + P_vec**2)
    theta = np.arctan2(V_vec, P_vec)
    
    # 向き補正 (平均が前上に来るように)
    mean_theta = np.mean(theta)
    if -np.pi < mean_theta < -np.pi/2: theta += np.pi

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    sc = ax.scatter(theta, r, c=cycles, cmap='hsv', s=30, alpha=0.8)
    ax.plot(theta, r, 'k-', alpha=0.4, linewidth=1)
    
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(f"Assist Vector Polar Chart - {task_key}\n(0°=Front, 90°=Up)", va='bottom', fontsize=14)
    plt.colorbar(sc, label='Gait Cycle (%)', fraction=0.03, pad=0.04)
    
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_polar_force.png")
    plt.savefig(save_path)
    print(f"保存完了: {save_path}")
    plt.show()

# =============================================================================
# Main
# =============================================================================
def main():
    print("\n=== Futto Mechanics Analyzer ===")
    
    # 1. Task Selection
    while True:
        try:
            task_key = input("解析するタスク名 (task1, task2, task3): ").strip().lower()
            if task_key in config.TASK_CONFIGS:
                cfg = config.TASK_CONFIGS[task_key]
                break
            else: print("無効なタスク名です。")
        except: sys.exit()

    # 2. Data Loading
    print("データを読み込んでいます...")
    df_coord, df_tension = load_data(cfg)
    if df_coord is None: return

    # 3. Mode Selection
    while True:
        print("\n--- Analysis Menu ---")
        print("1: Work Loop (Length vs Tension)")
        print("2: Polar Force (Assist Vector on Shank)")
        print("q: Quit")
        
        mode = input("Select Mode >> ").strip().lower()
        
        if mode == '1':
            run_work_loop_mode(df_coord, df_tension, cfg, task_key)
        elif mode == '2':
            run_polar_force_mode(df_coord, df_tension, cfg, task_key)
        elif mode == 'q':
            print("終了します。")
            break
        else:
            print("無効な選択です。")

if __name__ == "__main__":
    main()