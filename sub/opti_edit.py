'''
# opti_edit_C.py
#hurui
#kabsch
# ### 解析の全体像

この解析は、以下の3つの大きなステップで構成されます。各ステップで1つのPythonスクリプトを実行し、その結果を次のステップで使用します。

**Step 1: 生データの前処理**

  * **目的**: OptiTrackから出力された生のマーカーデータ（IDが安定しない）を、15個のマーカーIDが安定して追跡されたクリーンなデータに変換します。
  * **入力**: `task1.csv` (OptiTrack生データ)
  * **使用スクリプト**: `opti_edit_A.py`
  * **出力**: `task1_corrected_A.csv` (クリーンなマーカーデータ)

-----

**Step 2: 歩行周期の平均化と可視化**

  * **目的**: クリーンなマーカーデータと歩行周期の定義ファイルから、平均的な1歩行周期の動きを計算し、アニメーションで可視化します。
  * **入力**:
      * `task1_corrected_A.csv` (Step 1の出力)
      * `task1_gait_cycles.csv` (歩行周期の定義ファイル)
  * **使用スクリプト**: `create_anime_grad.py`
  * **出力**:
      * `task1_mean_cycle.csv` (平均化されたマーカーデータ)
      * 平均歩行周期のアニメーション表示
      * 部位ごとの長さ変化のグラフ表示

-----

**Step 3: 張力の計算と可視化**

  * **目的**: 平均化された歩行周期データとゴムの物性データから、各ゴム部分にかかる張力を計算し、アニメーションとグラフで可視化します。
  * **入力**:
      * `task1_mean_cycle.csv` (Step 2の出力)
      * `rubber_strength.xlsx` (ゴムの物性データ)
  * **使用スクリプト**: `strength_visualize.py`
  * **出力**:
      * 張力を色で表現した3Dアニメーション表示
      * 部位ごとの張力変化のグラフ表示

<!-- end list -->
'''

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- ▼▼▼ 設定 ▼▼▼ ---
# 新しいデータに合わせて、この部分をご自身の環境に合わせて更新してください。
OPTITRACK_CSV_PATH = r"C:\FuttoAnalysis\opti\20251027\task2.csv"
OUTPUT_CSV_PATH    = r"C:\FuttoAnalysis\opti\20251027\task2_corrected_A1.csv"

# pre_analysis_checker.pyで確認した安定区間の開始・終了時刻
STATIC_START = 3.00
STATIC_END   = 10.0

# pre_analysis_checker.pyで確認した安定マーカーの中から、
# 最も動きが少ないと思われる腰部のマーカーID
ANCHOR_ID = 7264 

# 物理的にありえる座標範囲（必要に応じて調整）
PLAUSIBLE_BOUNDS = {'x': (-300, 150), 'y': (0, 1100), 'z': (-1000, 100)}
#ここ大事
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


def load_opti_data_to_long_robust(file_path):
    """
    行ごとに列数が異なる可能性のあるOptiTrack CSVを頑健に読み込む関数。
    """
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return None
    
    print(f"'{os.path.basename(file_path)}' を読み込んでいます...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # ヘッダー行をスキップ
            for _ in range(43):
                next(f)
            
            for line in f:
                parts = line.strip().split(',')
                try:
                    if len(parts) < 5: continue
                    frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4])
                    base_col = 5
                    if len(parts) >= base_col + n_markers * 4:
                        for i in range(n_markers):
                            x = float(parts[base_col + 4*i + 0])
                            y = float(parts[base_col + 4*i + 1])
                            z = float(parts[base_col + 4*i + 2])
                            mid = int(parts[base_col + 4*i + 3])
                            rows.append((frame, t, mid, x * 1000.0, y * 1000.0, z * 1000.0))
                except (ValueError, IndexError):
                    continue
        out_df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])
        print(f"ファイル読み込み成功: {len(out_df)} 行")
        return out_df
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に予期せぬ問題が発生しました。詳細: {e}")
        return None

def kabsch_solve(A, B):
    """Kabschアルゴリズムで最適な回転行列Rと移動ベクトルtを計算する"""
    A, B = np.asarray(A), np.asarray(B)
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t_vec = cB - R @ cA
    return R, t_vec

def build_static_template(df_long):
    """
    静止立位区間から、最も安定した15個のマーカーでテンプレートを作成する。
    """
    static_df = df_long[(df_long['Time'] >= STATIC_START) & (df_long['Time'] <= STATIC_END)]
    
    # 静止区間で最も頻繁に出現する15個のマーカーIDを特定
    top_15_ids = static_df['id'].value_counts().nlargest(15).index
    if len(top_15_ids) < 15:
        print(f"警告: 静止区間で安定したマーカーが {len(top_15_ids)} 個しか見つかりませんでした。")
        return {}, [], None

    mean_pos = static_df[static_df['id'].isin(top_15_ids)].groupby('id')[['x','y','z']].mean()
    
    template = {int(mid): row.to_numpy() for mid, row in mean_pos.iterrows()}
    template_ids = sorted(template.keys())
    
    print(f"テンプレート作成完了。マーカー数: {len(template_ids)}")
    return template, template_ids

def process(df_long):
    """メインの追跡処理"""
    template_g, templ_ids = build_static_template(df_long)
    if not templ_ids:
        print("エラー: テンプレートを作成できませんでした。処理を中断します。")
        return pd.DataFrame()

    corrected_rows = []
    last_known_R = np.eye(3)
    last_known_t = np.zeros(3)

    for frame, g in df_long.groupby('Frame'):
        time_scalar = float(g['Time'].iloc[0])
        
        obs_df = g.copy()
        for axis, (min_val, max_val) in PLAUSIBLE_BOUNDS.items():
            obs_df = obs_df[(obs_df[axis] >= min_val) & (obs_df[axis] <= max_val)]
        
        if obs_df.empty:
            final_coords = {mid: last_known_R @ template_g[mid] + last_known_t for mid in templ_ids}
        else:
            obs_coords = obs_df[['x','y','z']].values
            
            predicted_template = {mid: last_known_R @ template_g[mid] + last_known_t for mid in templ_ids}
            pred_coords_arr = np.array([predicted_template[mid] for mid in templ_ids])
            
            cost_matrix = np.linalg.norm(pred_coords_arr[:, None, :] - obs_coords[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            reliable_templ_pts, reliable_obs_pts = [], []
            final_assignment = {}
            
            reliable_pairs_dist_threshold = 75.0
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < reliable_pairs_dist_threshold:
                    mid = templ_ids[r]
                    reliable_templ_pts.append(template_g[mid])
                    reliable_obs_pts.append(obs_coords[c])
                    final_assignment[mid] = obs_coords[c]

            if len(reliable_obs_pts) >= 4:
                R, t_vec = kabsch_solve(reliable_templ_pts, reliable_obs_pts)
                last_known_R, last_known_t = R, t_vec
            else:
                R, t_vec = last_known_R, last_known_t

            final_coords = {
                mid: final_assignment.get(mid, R @ template_g[mid] + t_vec)
                for mid in templ_ids
            }
        
        for mid in templ_ids:
            p = final_coords[mid]
            corrected_rows.append((frame, time_scalar, mid, p[0], p[1], p[2]))

    return pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])

if __name__ == "__main__":
    df_long = load_opti_data_to_long_robust(OPTITRACK_CSV_PATH)
    
    if df_long is not None and not df_long.empty:
        print("マーカー軌道の補正処理を開始します...")
        corrected_df = process(df_long)
        print(f"処理完了。補正後のデータ: {len(corrected_df)} 行")

        out_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        corrected_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"補正済みデータを保存しました: {OUTPUT_CSV_PATH}")
    else:
        print("データが読み込めなかったため、処理を終了します。")
