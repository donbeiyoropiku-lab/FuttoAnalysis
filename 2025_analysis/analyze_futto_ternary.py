"""
=============================================================================
Program: Futto Ternary EP Analyzer (Integrated Proximal & Distal)
Version: 1.6 (Integrated + Baseline Removal)

【変更点 v1.6】
  - 各ゴムの張力データに対し、「最小値（ベースライン）」を減算する処理を追加。
  - これにより、静的な初期張力(Pre-tension)の影響を除外し、
    歩行動作に伴う「動的な貢献度(Dynamic Contribution)」のみを評価します。

【プログラムの目的】
本プログラムは、Futtoのゴム張力データを「筋シナジー仮説」および「平衡点(EP)仮説」に基づいて解析し、
歩行動作における「制御戦略（Control Strategy）」を三角図（Ternary Plot）として可視化します。
「近位（股関節・膝）」と「遠位（足首・下腿）」の2つの視点から、Futtoがどの機能を代行・強調しているかを評価します。

【1. 近位 (Proximal) EP解析の定義】
股関節と膝関節の協調性を評価します。二関節筋相当のゴムは加重配分(0.5ずつ)して計算します。

  [頂点と機能]
  ● Top (上): Hip EP (Stability / 固定・安定)
    - 役割: 体幹・骨盤の支持、単関節伸展・屈曲。
    - ゴム: Back_Upper (100%) + Front_Upper (50%)
  ● Bottom-Right (右下): Hip-Knee EP (Coordination / 協調・リーチ)
    - 役割: 股関節と膝の連動、脚の振り出し、歩幅の調整。
    - ゴム: Front_Upper (50%) + Back_Thigh (50%)
  ● Bottom-Left (左下): Knee EP (Support / 荷重支持)
    - 役割: 着地時の膝折れ防止、純粋な膝の安定化。
    - ゴム: Front_Knee_Upper (100%) + Back_Thigh (50%)

【2. 遠位 (Distal) EP解析の定義】
下腿と足部の制御戦略を評価します。

  [頂点と機能]
  ● Top (上): Dorsiflexion (Clearance / 背屈・クリアランス)
    - 役割: つま先を引き上げ、つまずきを防止する。
    - ゴム: Front_Shin + Toe
  ● Bottom-Right (右下): Plantarflexion (Propulsion / 底屈・推進)
    - 役割: 地面を蹴り出し、前方への推進力を生む。
    - ゴム: Back_Shin
  ● Bottom-Left (左下): Stability (Knee-Shank Coupling / 安定・固定)
    - 役割: 脛骨を前後から挟み込み、足首と膝下のブレを防ぐ。
    - ゴム: Back_Knee + Front_Knee_Lower

【グラフの見方】
  - 軸の方向: 時計回り (0 → 1)
  - 色 (Color): 歩行周期 (0% = 着地, 60% = 遊脚初期, 100% = 次の着地)
  - 重心の位置:
      - 重心が「上」にある時 → 安定性(Hip) や クリアランス(DF) を重視している。
      - 重心が「右下」にある時 → 協調(HK) や 推進(PF) を重視している。
      - 重心が「左下」にある時 → 荷重支持(Knee) や 固定(Stab) を重視している。
  - ループの大きさ:
      - 大きい場合: メリハリのある動的なアシスト（理想的）。
      - 小さい場合: ギプスのような静的な固定アシスト。

【出力ファイル】
  1. {task}_ternary_proximal.png      (近位・全体図)
  2. {task}_ternary_proximal_zoom.png (近位・拡大図)
  3. {task}_ternary_distal.png        (遠位・全体図)
  4. {task}_ternary_distal_zoom.png   (遠位・拡大図)
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config
import math

# --- ゴムの分類定義 ---
RUBBER_GROUPS = {
    # --- Proximal用 (加重計算に使用) ---
    'Front_Upper': ['Front_Upper_In', 'Front_Upper_Out'],           
    'Back_Upper': ['Back_Upper_In', 'Back_Upper_Out'],              
    'Back_Thigh': ['Back_Thigh_In', 'Back_Thigh_Out'],              
    'Front_Knee_Upper': ['Front_Knee_Upper_In', 'Front_Knee_Upper_Out'],
    
    # --- Distal用 (単純合計に使用) ---
    # A: Dorsiflexion (Clearance)
    'DF_Group': ['Front_Shin', 'Toe_In', 'Toe_Out'],
    # B: Plantarflexion (Propulsion)
    'PF_Group': ['Back_Shin_In', 'Back_Shin_Out'],
    # C: Stability (Knee-Shank Coupling)
    'Stab_Group': ['Back_Knee_In', 'Back_Knee_Out', 'Front_Knee_Lower_In', 'Front_Knee_Lower_Out']
}

def load_tension_data(cfg):
    path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not path or not os.path.exists(path):
        print(f"エラー: ファイルなし {path}")
        return None
    return pd.read_csv(path)

def get_rubber_sum(df_tension, rubber_names, n_points):
    """
    指定されたゴムリストの合計張力を取得 (補間付き)
    ★修正(v1.6): 各ゴムごとに「最小値(ベースライン)」を引き、動的成分のみを抽出して加算する
    """
    total_vals = np.zeros(n_points)
    
    for name in rubber_names:
        seg_data = df_tension[df_tension['segment'] == name].sort_values('gait_cycle_%')
        if seg_data.empty: continue
        
        vals = seg_data['tension_N'].values
        
        # 1. データ長合わせ (補間)
        if len(vals) != n_points:
            vals = np.interp(np.linspace(0, 100, n_points), 
                             np.linspace(0, 100, len(vals)), vals)
        
        # 2. ★ベースライン除去 (Baseline Removal)
        # そのゴムの最小値を引き、「変動分」だけを取り出す
        current_min = np.min(vals)
        vals_dynamic = vals - current_min
        
        # 念のため負の値にならないようクリップ (数値計算誤差対策)
        vals_dynamic = np.maximum(vals_dynamic, 0)
        
        total_vals += vals_dynamic
        
    return total_vals

# --- 計算ロジック: Proximal ---
def calculate_proximal_balance(df_tension, n_points, cycles):
    # 生データ取得 (get_rubber_sum内でベースライン除去済み)
    vals = {}
    for key in ['Front_Upper', 'Back_Upper', 'Back_Thigh', 'Front_Knee_Upper']:
        vals[key] = get_rubber_sum(df_tension, RUBBER_GROUPS[key], n_points)
    
    # 重み付け計算 (User Logic)
    # Hip EP (Top): Back_Upper(1.0) + Front_Upper(0.5)
    val_a = vals['Back_Upper'] + 0.5 * vals['Front_Upper']
    
    # HK EP (Right): Front_Upper(0.5) + Back_Thigh(0.5)
    val_b = 0.5 * vals['Front_Upper'] + 0.5 * vals['Back_Thigh']
    
    # Knee EP (Left): Front_Knee_Upper(1.0) + Back_Thigh(0.5)
    val_c = vals['Front_Knee_Upper'] + 0.5 * vals['Back_Thigh']
    
    df = pd.DataFrame({'Val_A': val_a, 'Val_B': val_b, 'Val_C': val_c}, index=cycles)
    df['Total'] = (df['Val_A'] + df['Val_B'] + df['Val_C']).replace(0, 1e-6)
    
    df['Ratio_A'] = df['Val_A'] / df['Total'] # Top
    df['Ratio_B'] = df['Val_B'] / df['Total'] # Right
    df['Ratio_C'] = df['Val_C'] / df['Total'] # Left
    
    return df

# --- 計算ロジック: Distal ---
def calculate_distal_balance(df_tension, n_points, cycles):
    # 生データ取得 (get_rubber_sum内でベースライン除去済み)
    val_a = get_rubber_sum(df_tension, RUBBER_GROUPS['DF_Group'], n_points)   # Top: DF
    val_b = get_rubber_sum(df_tension, RUBBER_GROUPS['PF_Group'], n_points)   # Right: PF
    val_c = get_rubber_sum(df_tension, RUBBER_GROUPS['Stab_Group'], n_points) # Left: Stab
    
    df = pd.DataFrame({'Val_A': val_a, 'Val_B': val_b, 'Val_C': val_c}, index=cycles)
    df['Total'] = (df['Val_A'] + df['Val_B'] + df['Val_C']).replace(0, 1e-6)
    
    df['Ratio_A'] = df['Val_A'] / df['Total']
    df['Ratio_B'] = df['Val_B'] / df['Total']
    df['Ratio_C'] = df['Val_C'] / df['Total']
    
    return df

# --- 描画関連 ---
def draw_ternary_axis_labels(ax, labels):
    """軸ラベルと目盛りを描画 (Clockwise 0->1)"""
    sqrt3 = math.sqrt(3)
    ticks = [0.2, 0.4, 0.6, 0.8]
    
    # 1. Left Axis (Axis A): Bottom-Left -> Top (0 -> 1)
    for t in ticks:
        px, py = t * 0.5, t * sqrt3/2
        ax.plot([px, px-0.015], [py, py], 'k-', linewidth=0.8)
        ax.text(px-0.02, py, f'{t:.1f}', fontsize=8, ha='right', va='center')
        # Grid
        qx, qy = t*0.5 + (1-t)*1.0, t*sqrt3/2
        ax.plot([px, qx], [py, qy], 'k:', linewidth=0.5, alpha=0.3)
    
    ax.text(0.15, sqrt3/2 * 0.5, f'{labels["left_axis"]} (0→1)', rotation=60, ha='right', va='center', fontsize=9, fontweight='bold', color='gray')

    # 2. Right Axis (Axis B): Top -> Bottom-Right (0 -> 1)
    for t in ticks:
        px, py = (1-t)*0.5 + t*1.0, (1-t)*sqrt3/2
        ax.plot([px, px+0.015], [py, py], 'k-', linewidth=0.8)
        ax.text(px+0.02, py, f'{t:.1f}', fontsize=8, ha='left', va='center')
        # Grid
        qx, qy = t, 0
        ax.plot([px, qx], [py, qy], 'k:', linewidth=0.5, alpha=0.3)
        
    ax.text(0.85, sqrt3/2 * 0.5, f'{labels["right_axis"]} (0→1)', rotation=-60, ha='left', va='center', fontsize=9, fontweight='bold', color='gray')

    # 3. Bottom Axis (Axis C): Bottom-Right -> Bottom-Left (0 -> 1)
    for t in ticks:
        px, py = (1-t)*1.0, 0
        ax.plot([px, px], [py, py-0.02], 'k-', linewidth=0.8)
        ax.text(px, py-0.03, f'{t:.1f}', fontsize=8, ha='center', va='top')
        # Grid
        qx, qy = (1-t)*0.5, (1-t)*sqrt3/2
        ax.plot([px, qx], [py, qy], 'k:', linewidth=0.5, alpha=0.3)
        
    ax.text(0.5, -0.08, f'{labels["bottom_axis"]} (0→1)', ha='center', va='top', fontsize=9, fontweight='bold', color='gray')

def draw_ternary_base(ax, x_vals, y_vals, cycles, labels):
    sqrt3 = math.sqrt(3)
    
    # 枠線
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, sqrt3/2, 0]
    ax.plot(tri_x, tri_y, 'k-', linewidth=1.5, zorder=1)
    
    # グリッドと軸
    draw_ternary_axis_labels(ax, labels)
    
    # 頂点ラベル
    ax.text(0.5, sqrt3/2 + 0.02, labels['top_vertex'], ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.text(1.05, 0, labels['right_vertex'], ha='left', va='center', fontsize=12, fontweight='bold')
    ax.text(-0.05, 0, labels['left_vertex'], ha='right', va='center', fontsize=12, fontweight='bold')
    
    # 軌跡プロット
    sc = ax.scatter(x_vals, y_vals, c=cycles, cmap='hsv', s=30, alpha=0.8, edgecolor='none', zorder=3)
    ax.plot(x_vals, y_vals, 'k-', alpha=0.4, linewidth=0.8, zorder=2)
    
    # キーポイント注釈
    key_percs = [0, 10, 30, 50, 60, 80, 100]
    for kp in key_percs:
        # np.abs(cycles - kp) で cycles が numpy array である必要がある
        idx = np.abs(cycles - kp).argmin()
        ax.annotate(f'{kp}%', (x_vals[idx], y_vals[idx]), 
                    xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')
        ax.plot(x_vals[idx], y_vals[idx], 'ko', markersize=3, zorder=4)
        
    return sc

def save_plot(x_vals, y_vals, cycles, labels, title, filename_suffix, task_key):
    # --- 1. 全体図 ---
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)
    sc = draw_ternary_base(ax, x_vals, y_vals, cycles, labels)
    plt.colorbar(sc, label='Gait Cycle (%)', fraction=0.03, pad=0.04)
    plt.title(f'{title} (Full) - {task_key}', fontsize=14)
    ax.axis('off'); ax.set_aspect('equal')
    
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_{filename_suffix}.png")
    plt.savefig(save_path)
    print(f"  保存完了: {os.path.basename(save_path)}")
    plt.close(fig)

    # --- 2. 拡大図 ---
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
    span = max(x_max - x_min, y_max - y_min) * 1.5
    if span < 0.1: span = 0.1

    fig_z = plt.figure(figsize=(10, 9))
    ax_z = fig_z.add_subplot(111)
    sc_z = draw_ternary_base(ax_z, x_vals, y_vals, cycles, labels)
    
    ax_z.set_xlim(cx - span/2, cx + span/2)
    ax_z.set_ylim(cy - span/2, cy + span/2)
    ax_z.axis('on')
    ax_z.set_xticks([]); ax_z.set_yticks([])
    for spine in ax_z.spines.values():
        spine.set_visible(True); spine.set_color('gray')

    plt.colorbar(sc_z, label='Gait Cycle (%)', fraction=0.03, pad=0.04)
    plt.title(f'{title} (Zoomed) - {task_key}', fontsize=14)
    ax_z.set_aspect('equal')
    
    save_path_zoom = os.path.join(config.RESULT_DIR, f"{task_key}_{filename_suffix}_zoom.png")
    plt.savefig(save_path_zoom)
    print(f"  保存完了: {os.path.basename(save_path_zoom)}")
    plt.close(fig_z)

def main():
    print("\n=== Futto Integrated EP Analyzer (Proximal & Distal) ===")
    while True:
        try:
            task_key = input("解析するタスク名 (task1, task2, task3): ").strip().lower()
            if task_key in config.TASK_CONFIGS:
                cfg = config.TASK_CONFIGS[task_key]; break
        except: sys.exit()

    df_tension = load_tension_data(cfg)
    if df_tension is None: return

    # numpy array として取得する
    cycles = np.array(sorted(df_tension['gait_cycle_%'].unique()))
    n_points = len(cycles)

    # ==========================================
    # 1. Proximal EP (Hip / Hip-Knee / Knee)
    # ==========================================
    print("\n--- Generating Proximal EP Plot ---")
    df_prox = calculate_proximal_balance(df_tension, n_points, cycles)
    
    sqrt3 = math.sqrt(3)
    a_prox = df_prox['Ratio_A'].values
    b_prox = df_prox['Ratio_B'].values
    c_prox = df_prox['Ratio_C'].values
    x_prox = 0.5 * a_prox + 1.0 * b_prox
    y_prox = (sqrt3 / 2) * a_prox
    
    labels_prox = {
        'top_vertex': 'Hip EP\n(Stability)',
        'right_vertex': 'Hip-Knee EP\n(Coordination)',
        'left_vertex': 'Knee EP\n(Support)',
        'left_axis': 'Hip Axis',
        'right_axis': 'Hip&Knee Axis',
        'bottom_axis': 'Knee Axis'
    }
    
    save_plot(x_prox, y_prox, cycles, labels_prox, 
              "Proximal EP Balance", "ternary_proximal", task_key)

    # ==========================================
    # 2. Distal EP (DF / PF / Stability)
    # ==========================================
    print("\n--- Generating Distal EP Plot ---")
    df_dist = calculate_distal_balance(df_tension, n_points, cycles)
    
    a_dist = df_dist['Ratio_A'].values # Top
    b_dist = df_dist['Ratio_B'].values # Right
    c_dist = df_dist['Ratio_C'].values # Left
    x_dist = 0.5 * a_dist + 1.0 * b_dist
    y_dist = (sqrt3 / 2) * a_dist
    
    labels_dist = {
        'top_vertex': 'Dorsiflexion\n(Clearance)',
        'right_vertex': 'Plantarflexion\n(Propulsion)',
        'left_vertex': 'Knee-Shank\n(Stability)',
        'left_axis': 'DF Axis',
        'right_axis': 'PF Axis',
        'bottom_axis': 'Stability Axis'
    }
    
    save_plot(x_dist, y_dist, cycles, labels_dist, 
              "Distal EP Balance", "ternary_distal", task_key)
              
    print("\nAll plots generated successfully.")

if __name__ == "__main__":
    main()