# =============================================================================
# analyze_rubber_synergy.py
#
# 目的:
#   非負値行列因子分解 (NNMF) を用いて、多数のゴムの張力データから
#   少数の「協調パターン (Synergy)」を抽出する。
#   これにより、「どのような組み合わせ」が「いつ」活動しているかを明らかにする。
#
# 出力:
#   1. VAF (Variance Accounted For) グラフ: 最適なシナジー数を決定するためのグラフ
#   2. シナジー詳細図: 各シナジーの「ゴムの構成比(W)」と「活動タイミング(H)」
# =============================================================================

# =============================================================================
# analyze_rubber_synergy.py (v1.1 - 描画バグ修正版)
#
# 修正点:
#   - plot_synergy_details で n=1 の時にエラーになる問題を修正 (squeeze=False)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import os
import sys
import config

# --- 設定 ---
MIN_SYNERGIES = 1
MAX_SYNERGIES = 6

# 解析対象のゴム順序
SORT_ORDER = [
    'Front_Upper_In', 'Front_Upper_Out',
    'Back_Upper_In', 'Back_Upper_Out',
    'Back_Thigh_In', 'Back_Thigh_Out',
    'Front_Knee_Upper_In', 'Front_Knee_Upper_Out',
    'Front_Knee_Lower_In', 'Front_Knee_Lower_Out',
    'Back_Knee_In', 'Back_Knee_Out',
    'Front_Shin',
    'Back_Shin_In', 'Back_Shin_Out',
    'Toe_In', 'Toe_Out'
]

def load_tension_data(cfg):
    path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not path or not os.path.exists(path):
        print(f"エラー: ファイルなし {path}")
        return None
    return pd.read_csv(path)

def preprocess_for_nmf(df_tension):
    cycles = sorted(df_tension['gait_cycle_%'].unique())
    segments = df_tension['segment'].unique()
    
    ordered_segments = [s for s in SORT_ORDER if s in segments]
    remaining = [s for s in segments if s not in ordered_segments]
    feature_names = ordered_segments + remaining
    
    matrix_list = []
    
    for seg in feature_names:
        seg_data = df_tension[df_tension['segment'] == seg].sort_values('gait_cycle_%')
        vals = seg_data['tension_N'].values
        
        if len(vals) != len(cycles):
            vals = np.interp(np.linspace(0, 100, len(cycles)), 
                             np.linspace(0, 100, len(vals)), vals)
        
        if np.max(vals) > 0:
            vals = vals / np.max(vals)
        
        matrix_list.append(vals)
    
    X = np.array(matrix_list)
    return X.T, feature_names, cycles

def calculate_vaf(X, W, H):
    X_reconstructed = np.dot(W, H)
    error = X - X_reconstructed
    sse = np.sum(error ** 2)
    sst = np.sum(X ** 2)
    vaf = 1 - (sse / sst)
    return vaf

def analyze_synergies(X, n_components):
    # init='nndsvd' 推奨
    model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=2000)
    W = model.fit_transform(X) 
    H = model.components_      
    return W, H

def plot_vaf_curve(X, task_key):
    vafs = []
    ns = range(MIN_SYNERGIES, MAX_SYNERGIES + 1)
    
    print("シナジー数の検討中 (VAF計算)...")
    for n in ns:
        W, H = analyze_synergies(X, n)
        vaf = calculate_vaf(X, W, H)
        vafs.append(vaf)
        print(f"  n={n}: VAF = {vaf:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(ns, vafs, 'o-', linewidth=2)
    plt.axhline(0.90, color='r', linestyle='--', label='90% Threshold')
    plt.title(f'VAF Curve (Synergy Selection) - {task_key}')
    plt.xlabel('Number of Synergies')
    plt.ylabel('VAF (Variance Accounted For)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    suggested_n = next((n for n, v in zip(ns, vafs) if v >= 0.90), 3)
    return suggested_n

def plot_synergy_details(W, H, feature_names, cycles, task_key, n_synergies):
    # Hの正規化
    for i in range(n_synergies):
        scale = np.max(H[i, :])
        if scale > 0:
            H[i, :] /= scale
            W[:, i] *= scale

    # ★修正箇所: squeeze=False で常に2次元配列として受け取る
    fig, axes = plt.subplots(n_synergies, 2, figsize=(15, 3 * n_synergies), squeeze=False)
    fig.suptitle(f'Extracted Synergies (n={n_synergies}) - {task_key}', fontsize=16)
    
    colors = []
    for name in feature_names:
        if 'Back' in name: colors.append('royalblue')
        elif 'Toe' in name: colors.append('orange')
        else: colors.append('tomato')

    for i in range(n_synergies):
        # 1. 時間パターン (W)
        ax_time = axes[i, 0]
        ax_time.plot(cycles, W[:, i], linewidth=2, color='black')
        ax_time.fill_between(cycles, W[:, i], alpha=0.2, color='gray')
        ax_time.set_title(f'Synergy #{i+1} : Temporal Activation', fontsize=12)
        ax_time.set_ylabel('Activation')
        ax_time.set_xlim(0, 100)
        ax_time.grid(True, linestyle='--')
        
        # 2. 空間パターン (H)
        ax_space = axes[i, 1]
        y_pos = np.arange(len(feature_names))
        ax_space.barh(y_pos, H[i, :], color=colors)
        ax_space.set_yticks(y_pos)
        ax_space.set_yticklabels(feature_names, fontsize=9)
        ax_space.set_title(f'Synergy #{i+1} : Rubber Weights (Spatial)', fontsize=12)
        ax_space.set_xlim(0, 1.1)
        ax_space.grid(axis='x', linestyle='--')
        ax_space.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_synergy_n{n_synergies}.png")
    plt.savefig(save_path)
    print(f"結果保存: {save_path}")
    plt.show()

def main():
    print("\n=== Rubber Synergy Analyzer (NNMF) ===")
    while True:
        try:
            task_key = input("解析するタスク名 (task1, task2, task3): ").strip().lower()
            if task_key in config.TASK_CONFIGS:
                cfg = config.TASK_CONFIGS[task_key]; break
        except: sys.exit()

    df_tension = load_tension_data(cfg)
    if df_tension is None: return

    X, feature_names, cycles = preprocess_for_nmf(df_tension)
    
    suggested_n = plot_vaf_curve(X, task_key)
    print(f"\n推奨されるシナジー数: {suggested_n}")
    
    # 手動入力受付
    try:
        user_input = input(f"解析するシナジー数を入力してください (Default: {suggested_n}): ")
        n_synergies = int(user_input) if user_input.strip() else suggested_n
    except:
        n_synergies = suggested_n

    W, H = analyze_synergies(X, n_synergies)
    plot_synergy_details(W, H, feature_names, cycles, task_key, n_synergies)

if __name__ == "__main__":
    main()