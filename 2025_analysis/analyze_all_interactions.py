# =============================================================================
# analyze_all_interactions.py
#
# 目的:
#   表側(Front)と裏側(Back)を区別せず、全ゴム間の相互作用(相関・遅延)を解析する。
#   これにより「表が縮むと裏が伸びる(拮抗)」などの連動性を可視化する。
#
# 出力:
#   1. Correlation Matrix: 同時収縮(正)か拮抗(負)か
#   2. Lag Matrix: タイミングのズレ
#   3. Unified Hovmöller: 高さ順に並べた全身の波動図
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import config

# --- 設定: 解析順序 (身体の上から下へ、表裏を混ぜて配置) ---
# この順序でグラフのY軸が並びます
SORT_ORDER = [
    # --- Hip / Thigh Level ---
    ['Front_Upper_In', 'Front_Upper_Out'], # 表・腰
    ['Back_Upper_In', 'Back_Upper_Out'],   # 裏・腰
    ['Back_Thigh_In', 'Back_Thigh_Out'],   # 裏・ハム
    
    # --- Knee Level ---
    ['Front_Knee_Upper_In', 'Front_Knee_Upper_Out'], # 表・膝上
    ['Front_Knee_Lower_In', 'Front_Knee_Lower_Out'], # 表・膝下
    ['Back_Knee_In', 'Back_Knee_Out'],               # 裏・膝
    
    # --- Shank / Ankle Level ---
    ['Front_Shin'],                      # 表・すね
    ['Back_Shin_In', 'Back_Shin_Out'],   # 裏・ふくらはぎ
    ['Toe_In', 'Toe_Out']                # 表・つま先
]

def load_tension_data(cfg):
    """張力データ読み込み"""
    tension_path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not tension_path or not os.path.exists(tension_path):
        print(f"エラー: ファイルなし {tension_path}")
        return None
    return pd.read_csv(tension_path)

def preprocess_all_rubbers(df_tension):
    """
    全ゴムのデータを抽出し、SORT_ORDERに従って並べたDataFrameを作成
    """
    cycles = sorted(df_tension['gait_cycle_%'].unique())
    matrix_data = []
    labels = []

    for group in SORT_ORDER:
        # グループ内の平均張力を計算
        level_tensions = np.zeros(len(cycles))
        valid_count = 0
        
        for name in group:
            seg_data = df_tension[df_tension['segment'] == name].sort_values('gait_cycle_%')
            values = seg_data['tension_N'].values
            if len(values) == len(cycles):
                level_tensions += values
                valid_count += 1
        
        if valid_count > 0:
            avg_tension = level_tensions / valid_count
            
            # 正規化 (0-1)
            if np.max(avg_tension) > np.min(avg_tension):
                norm_tension = (avg_tension - np.min(avg_tension)) / (np.max(avg_tension) - np.min(avg_tension))
            else:
                norm_tension = avg_tension

            matrix_data.append(norm_tension)
            # ラベル作成 (Front/Backがわかるように)
            label = group[0].replace('_In','').replace('_Out','')
            labels.append(label)

    return pd.DataFrame(matrix_data, index=labels, columns=cycles)

def calculate_cross_correlation_matrix(df_matrix):
    """
    全ペアの最大相関係数とラグ(遅れ)を計算
    """
    n_vars = len(df_matrix)
    labels = df_matrix.index
    n_points = df_matrix.shape[1]
    
    corr_mat = np.zeros((n_vars, n_vars))
    lag_mat = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            sig1 = df_matrix.iloc[i].values
            sig2 = df_matrix.iloc[j].values
            
            # 相互相関
            # 循環相関として計算 (-50% ~ +50%)
            best_corr = -1
            best_lag = 0
            
            for lag in range(-n_points//2, n_points//2):
                shifted_sig2 = np.roll(sig2, -lag)
                corr = np.corrcoef(sig1, shifted_sig2)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            
            # 対角成分(自分自身)は corr=1.0, lag=0
            if i == j:
                best_corr = 1.0
                best_lag = 0
            
            corr_mat[i, j] = best_corr
            lag_mat[i, j] = best_lag

    return corr_mat, lag_mat, labels

def plot_interaction_matrices(corr_mat, lag_mat, labels, task_key):
    """相関行列とラグ行列を描画"""
    
    # 1. 相関行列 (Correlation)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, xticklabels=labels, yticklabels=labels, 
                cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt=".1f")
    plt.title(f'Interaction Matrix (Correlation) - {task_key}\nRed=Co-contraction, Blue=Antagonistic')
    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_all_correlation.png")
    plt.savefig(save_path)
    print(f"保存: {save_path}")
    plt.show()
    
    # 2. ラグ行列 (Time Lag)
    plt.figure(figsize=(10, 8))
    # ラグは -50 ~ +50 の範囲。0付近(同期)を白、プラス(遅れ)を赤、マイナス(先行)を青で表現
    sns.heatmap(lag_mat, xticklabels=labels, yticklabels=labels, 
                cmap='PuOr', center=0, annot=True, fmt=".0f")
    plt.title(f'Interaction Matrix (Phase Lag %) - {task_key}\nRow leads Column (Blue) vs Row follows Column (Red)')
    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_all_lag.png")
    plt.savefig(save_path)
    print(f"保存: {save_path}")
    plt.show()

def plot_unified_hovmoller(df_matrix, task_key):
    """統合型時空間プロット"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_matrix, cmap='magma', cbar_kws={'label': 'Normalized Tension'})
    plt.title(f'Unified Spatiotemporal Plot (Front & Back) - {task_key}')
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Segment (Proximal -> Distal)')
    plt.tight_layout()
    save_path = os.path.join(config.RESULT_DIR, f"{task_key}_unified_hovmoller.png")
    plt.savefig(save_path)
    print(f"保存: {save_path}")
    plt.show()

def main():
    print("\n=== All-to-All Interaction Analyzer ===")
    while True:
        try:
            task_key = input("解析するタスク名 (task1, task2, task3): ").strip().lower()
            if task_key in config.TASK_CONFIGS:
                cfg = config.TASK_CONFIGS[task_key]
                break
        except: sys.exit()

    df_tension = load_tension_data(cfg)
    if df_tension is None: return

    # 1. データ整形 (表裏混合)
    df_matrix = preprocess_all_rubbers(df_tension)

    # 2. 統合型 時空間プロット
    plot_unified_hovmoller(df_matrix, task_key)

    # 3. 相関・ラグ行列計算
    corr_mat, lag_mat, labels = calculate_cross_correlation_matrix(df_matrix)
    
    # 4. 行列描画
    plot_interaction_matrices(corr_mat, lag_mat, labels, task_key)

if __name__ == "__main__":
    main()