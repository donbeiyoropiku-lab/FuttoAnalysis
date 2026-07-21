'''
# opti_edit_A.py

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
OPTITRACK_CSV_PATH = r"C:\FuttoAnalysis\opti\20251020\task2.csv"
OUTPUT_CSV_PATH    = r"C:\FuttoAnalysis\opti\20251020\task2_corrected_A1.csv"

# 静止立位区間
STATIC_START = 2.045
STATIC_END   = 4.336

# 物理的にありえる座標範囲（修正済み）
#task1
#PLAUSIBLE_BOUNDS = {'x': (0, 1000), 'y': (0, 1100), 'z': (-100, 400)}

#task2,3
PLAUSIBLE_BOUNDS = {'x': (-300, 150), 'y': (0, 1100), 'z': (-1000, 100)}

# マッチングする際の最大許容誤差 (mm)
MATCHING_THRESHOLD_MM = 75.0
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


def load_opti_data_to_long_robust(file_path):
    """行ごとに列数が異なる可能性のあるOptiTrack CSVを頑健に読み込む関数。"""
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
                            x, y, z, mid = float(parts[base_col + 4*i]), float(parts[base_col + 4*i + 1]), float(parts[base_col + 4*i + 2]), int(parts[base_col + 4*i + 3])
                            rows.append((frame, t, mid, x * 1000.0, y * 1000.0, z * 1000.0))
                except (ValueError, IndexError): continue
        out_df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])
        print(f"ファイル読み込み成功: {len(out_df)} 行")
        return out_df
    except Exception as e:
        print(f"エラー: ファイルの読み込み中に予期せぬ問題が発生しました。詳細: {e}")
        return None

def kabsch_solve(A, B):
    """Kabschアルゴリズムで A から B への最適な回転行列Rと移動ベクトルtを計算する"""
    A, B = np.asarray(A), np.asarray(B)
    # 点が少なすぎる場合は単位行列とゼロベクトルを返す（動きなしと仮定）
    if A.shape[0] < 3: return np.identity(3), np.zeros(3)

    cA, cB = A.mean(axis=0), B.mean(axis=0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    # 反転を防ぐためのチェック
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    t_vec = cB - R_mat @ cA
    return R_mat, t_vec

def build_static_template(df_long):
    """静止立位区間からテンプレートを作成する"""
    static_df = df_long[(df_long['Time'] >= STATIC_START) & (df_long['Time'] <= STATIC_END)]
    # 静止区間で最も頻繁に出現する15個のマーカーIDを特定
    top_15_ids = static_df['id'].value_counts().nlargest(15).index
    if len(top_15_ids) < 15:
        print(f"警告: 静止区間で安定したマーカーが {len(top_15_ids)} 個しか見つかりませんでした。")
        return None, None
    # テンプレートを構築
    mean_pos = static_df[static_df['id'].isin(top_15_ids)].groupby('id')[['x','y','z']].mean()
    template_ids = sorted(mean_pos.index.astype(int).tolist())
    # テンプレートを辞書形式 {id: np.array([x,y,z])} で保存
    template_geometry = {int(mid): row.to_numpy() for mid, row in mean_pos.iterrows()}
    print(f"テンプレート作成完了。マーカー数: {len(template_ids)}")
    return template_geometry, template_ids

def filter_plausible_markers(frame_df, bounds):
    """物理的にありえる範囲内のマーカーのみを抽出する"""
    mask = (frame_df['x'].between(*bounds['x']) &
            frame_df['y'].between(*bounds['y']) &
            frame_df['z'].between(*bounds['z']))
    return frame_df[mask]

def process(df_long):
    """メインの追跡処理 (以前の成功ロジックに基づく)"""
    template_g, template_ids = build_static_template(df_long)
    if not template_ids:
        print("エラー: テンプレートを作成できませんでした。処理を中断します。")
        return pd.DataFrame()

    corrected_rows = []
    # 状態変数: 静的テンプレートから直前のフレームの姿勢への剛体変換
    last_known_R = np.eye(3)
    last_known_t = np.zeros(3)

    # フレーム番号順に処理
    unique_frames = sorted(df_long['Frame'].unique())
    for frame in unique_frames:
        g = df_long[df_long['Frame'] == frame]
        time_scalar = float(g['Time'].iloc[0])
        obs_df = filter_plausible_markers(g, PLAUSIBLE_BOUNDS)

        if obs_df.empty:
            # 観測点がない場合: 直前の変換を使い、静的テンプレートから位置を予測
            final_coords = {mid: last_known_R @ template_g[mid] + last_known_t for mid in template_ids}
        else:
            obs_coords = obs_df[['x','y','z']].values # 現フレームの観測点群

            # 1. 予測: 直前の変換(last_R, last_t)を使い、静的テンプレートから現在の姿勢を予測
            predicted_template_coords = {mid: last_known_R @ template_g[mid] + last_known_t for mid in template_ids}
            pred_coords_arr = np.array([predicted_template_coords[mid] for mid in template_ids])

            # 2. マッチング: 予測位置と観測位置を比較し、最も近いペアを見つける
            cost_matrix = np.linalg.norm(pred_coords_arr[:, np.newaxis, :] - obs_coords[np.newaxis, :, :], axis=2)
            # ハンガリアン法で最適な割り当てを計算
            template_indices, obs_indices = linear_sum_assignment(cost_matrix)

            # 3. 信頼できるペアを抽出 (Kabsch計算用)
            reliable_template_points_static = [] # 静的テンプレート上の点
            reliable_observed_points = []      # 対応する観測点
            final_assignment = {}              # {template_id: observed_coord} の辞書

            for i, j in zip(template_indices, obs_indices):
                if cost_matrix[i, j] < MATCHING_THRESHOLD_MM:
                    template_id = template_ids[i]
                    reliable_template_points_static.append(template_g[template_id])
                    reliable_observed_points.append(obs_coords[j])
                    final_assignment[template_id] = obs_coords[j] # 観測座標を記録

            # 4. Kabsch計算: 静的テンプレートから現在の観測点への *新しい* 剛体変換を計算
            if len(reliable_observed_points) >= 3:
                # Kabschには静的テンプレートの点と、それに対応する観測点を与える
                current_R, current_t = kabsch_solve(reliable_template_points_static, reliable_observed_points)
                # 計算した新しい変換を、次のフレームの予測のために保存
                last_known_R, last_known_t = current_R, current_t
            else:
                # 信頼できるペアが少ない場合、直前の変換を維持
                current_R, current_t = last_known_R, last_known_t
                # print(f"フレーム {frame}: 信頼できるマッチングが少なすぎるため、前回の変換を使用します。")


            # 5. 最終座標の決定:
            #    - マッチングした点 (final_assignmentにある点) は、観測座標をそのまま使う (変形を反映)
            #    - マッチングしなかった点 (欠損点) は、*新しく計算した変換(current_R, current_t)* を
            #      *静的テンプレート*に適用して予測する
            final_coords = {}
            for tid in template_ids:
                if tid in final_assignment:
                    final_coords[tid] = final_assignment[tid] # 観測値を採用
                else:
                    final_coords[tid] = current_R @ template_g[tid] + current_t # 予測値で補完

        # 6. 結果を格納
        for mid in template_ids:
            pos = final_coords.get(mid, [np.nan, np.nan, np.nan]) # 念のため存在確認
            corrected_rows.append((frame, time_scalar, int(mid), *pos))

    df = pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])

    # 最終的な短い欠損を線形補間
    if df['x'].isnull().any():
        print("警告: 処理結果にNaNが含まれています。補間を試みます...")
        df_interpolated = []
        for tid in template_ids:
            marker_df = df[df['id'] == tid].copy()
            marker_df = marker_df.sort_values('Time')
            marker_df[['x', 'y', 'z']] = marker_df[['x', 'y', 'z']].interpolate(method='linear', limit_direction='both', axis=0, limit_area='inside')
            df_interpolated.append(marker_df)
        df = pd.concat(df_interpolated).sort_values(['Frame', 'id']).reset_index(drop=True)
        if df.isnull().values.any():
             print("警告: 補間後もNaNが残っています。データ欠損が大きい可能性があります。")

    return df

if __name__ == "__main__":
    df_long = load_opti_data_to_long_robust(OPTITRACK_CSV_PATH)

    if df_long is not None and not df_long.empty:
        print("マーカー軌道の補正処理を開始します...")
        corrected_df = process(df_long)

        if corrected_df.isnull().values.any():
             print("警告: 最終出力データにNaNが含まれています。")
        elif corrected_df.empty:
             print("エラー: 処理結果が空になりました。")
        else:
            # 結果の最初の数フレームと最後の数フレームを比較して動きがあるか簡易チェック
            first_pose_df = corrected_df[corrected_df['Frame'] == corrected_df['Frame'].min()]
            last_pose_df = corrected_df[corrected_df['Frame'] == corrected_df['Frame'].max()]
            # Make sure both frames have the same set of IDs before comparing
            common_ids = sorted(list(set(first_pose_df['id']) & set(last_pose_df['id'])))
            if common_ids:
                 first_pose = first_pose_df[first_pose_df['id'].isin(common_ids)].sort_values('id')[['x','y','z']].values
                 last_pose = last_pose_df[last_pose_df['id'].isin(common_ids)].sort_values('id')[['x','y','z']].values
                 if first_pose.shape == last_pose.shape and np.allclose(first_pose, last_pose, atol=1e-1): # 許容誤差を少し大きく
                     print("警告: 処理結果の最初と最後のフレームの形状が非常に近いです。動きが小さいか、追跡に問題がある可能性があります。")
                 else:
                      print("処理完了。形状の変化が検出されました。")
            else:
                 print("警告：最初と最後のフレームで共通のマーカーIDが見つからず、比較できませんでした。")


            print(f"補正後のデータ: {len(corrected_df)} 行")
            out_dir = os.path.dirname(OUTPUT_CSV_PATH)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            corrected_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
            print(f"補正済みデータを保存しました: {OUTPUT_CSV_PATH}")

    else:
        print("データが読み込めなかったため、処理を終了します。")

