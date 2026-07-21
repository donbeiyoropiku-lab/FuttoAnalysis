# =============================================================================
# opti_clean.py (v23 - Persistent ID Map)
#
# 概要:
#   n点剛体マーカーを追跡する。
#
# アプローチ: "生ID/IDマップ優先" + "近傍点探索(マップ登録)" + "剛体補完"
#
# 処理フロー:
# 1. テンプレート作成
# 2. キーフレーム特定 (追跡起点)
# 3. 歩行区間の追跡 (process_rigid_tracking_v3):
#    - id_map = {} (生ID->TIDの永続マップ) を初期化
#    - (A) テンプレートID優先: 観測点の生IDがテンプレートIDなら採用。
#    - (B) IDマップ優先: 観測点の生IDがid_mapにあれば採用。
#    - (C) 近傍点探索: (A)(B)以外の「ID不明観測点」と
#          「未発見テンプレートID」をマッチング。
#    - (D) IDマップ登録: (C)でマッチしたら、id_map[生ID] = テンプレートID を登録。
#    - (E) 剛体補完: (A)～(D)でも見つからないIDを剛体補完。
#    - (F) 状態更新: 最終ポーズを次フレームの近傍探索用に保存。
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- ▼▼▼ 設定 ▼▼▼ ---
# 1. ファイルパス

OPTITRACK_CSV_PATH = r"C:\FuttoAnalysis\opti\20251111\try1.csv"

OUTPUT_CSV_PATH    = r"C:\FuttoAnalysis\opti\20251111\try_corrected.csv"



# 2. 時刻区間 (秒)

STANDING_START_TIME = 0.0 # 静止立位の開始

WALKING_START_TIME  = 10.0 # 歩行開始 (テンプレート終了 & 追跡開始時刻)

WALKING_STOP_TIME   = 120.0 # 歩行終了 (この時刻まで処理)



# 3. 処理設定

NUM_MARKERS = 12 # 追跡対象のマーカー数

MATCHING_THRESHOLD_MM = 75.0 # マッチング許容誤差 (mm)



# 4. 物理的にありえる座標範囲（ノイズ除去用）

PLAUSIBLE_BOUNDS = {'x': (-400, 400), 'y': (0, 1100), 'z': (-400, 1200)}

# =============================================================================
# ヘルパー関数群 (v22と同様のため省略)
# =============================================================================
def load_opti_data_to_long_robust(file_path: str) -> pd.DataFrame | None:
    # (... v22と同様 ...)
    if not os.path.exists(file_path): print(f"エラー: ファイルが見つかりません: {file_path}"); return None
    print(f"'{os.path.basename(file_path)}' を読み込み中..."); rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f)
            for line_num, line in enumerate(f, 44):
                parts = line.strip().split(',');
                try:
                    if len(parts) < 5: continue
                    frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4]); base_col = 5
                    if len(parts) >= base_col + n_markers * 4:
                        for i in range(n_markers):
                            x = float(parts[base_col + 4*i]) * 1000.0; y = float(parts[base_col + 4*i + 1]) * 1000.0
                            z = float(parts[base_col + 4*i + 2]) * 1000.0; mid = int(parts[base_col + 4*i + 3])
                            rows.append((frame, t, mid, x, y, z))
                except (ValueError, IndexError): continue
        if not rows: print("エラー: 有効なデータ行が読み込めませんでした。"); return None
        df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"]); print(f"読み込み成功: {len(df)} 行"); return df
    except Exception as e: print(f"ファイル読み込みエラー: {e}"); return None

def filter_plausible_markers(frame_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    # (... v22と同様 ...)
    if frame_df.empty: return frame_df
    try: mask = (frame_df['x'].between(*bounds['x']) & frame_df['y'].between(*bounds['y']) & frame_df['z'].between(*bounds['z'])); return frame_df[mask]
    except KeyError: print(f"エラー: filter_plausible - 列 x, y, z が見つかりません。"); return pd.DataFrame()

def kabsch_solve(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (... v22と同様 ...)
    A_arr, B_arr = np.asarray(A), np.asarray(B)
    if A_arr.ndim != 2 or B_arr.ndim != 2 or A_arr.shape[1] != 3 or B_arr.shape[1] != 3 or \
       A_arr.shape[0] < 3 or B_arr.shape[0] < 3 or A_arr.shape != B_arr.shape or \
       np.isnan(A_arr).any() or np.isnan(B_arr).any(): return np.identity(3), np.zeros(3)
    try:
        cA, cB = A_arr.mean(axis=0), B_arr.mean(axis=0); H = (A_arr - cA).T @ (B_arr - cB); U, _, Vt = np.linalg.svd(H); R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0: Vt[-1, :] *= -1; R_mat = Vt.T @ U.T
        t_vec = cB - R_mat @ cA; return R_mat, t_vec
    except Exception as e: print(f"エラー: Kabsch計算エラー: {e}"); return np.identity(3), np.zeros(3)

def build_static_template(df_long: pd.DataFrame, start_time: float, end_time: float, num_markers: int) -> tuple[dict | None, list | None]:
    # (... v22と同様 ...)
    print(f"静止区間 ({start_time}s - {end_time}s) からテンプレートを作成..."); static_df = df_long[(df_long['Time'] >= start_time) & (df_long['Time'] <= end_time)]
    if static_df.empty: print("エラー: 指定された静止区間にデータがありません。"); return None, None
    top_n_ids = static_df['id'].value_counts().nlargest(num_markers).index
    if len(top_n_ids) < num_markers: print(f"警告: 安定マーカー不足 ({len(top_n_ids)}/{num_markers})。"); return None, None
    mean_pos_df = static_df[static_df['id'].isin(top_n_ids)].groupby('id')[['x','y','z']].mean()
    template_geometry = {int(mid): row.to_numpy() for mid, row in mean_pos_df.iterrows()}; template_ids = sorted(template_geometry.keys())
    print(f"テンプレート作成完了。マーカー数: {len(template_ids)}"); print(f"  テンプレートID: {template_ids}"); return template_geometry, template_ids

def get_keyframe_pose(df_long: pd.DataFrame, keyframe_time: float, template_geometry: dict, template_ids: list) -> tuple[int, dict, dict]:
    """
    追跡開始時刻 (キーフレーム) のマーカー位置とIDを特定する。
    生ID -> テンプレートID の初期マップも返す。
    """
    print(f"キーフレーム (t~{keyframe_time:.3f}s) の初期ポーズを特定中...")
    initial_pose_map = {} # {tid: pos}
    id_map = {} # {raw_id: tid}
    
    try:
        key_frame_row = df_long.iloc[(df_long['Time'] - keyframe_time).abs().argmin()]
        key_frame_index = int(key_frame_row['Frame']); actual_time = key_frame_row['Time']
    except (IndexError, KeyError): print("エラー: キーフレーム時刻付近データなし。"); return -1, {}, {}
    
    key_frame_df = df_long[df_long['Frame'] == key_frame_index]; obs_df = filter_plausible_markers(key_frame_df, PLAUSIBLE_BOUNDS)
    if obs_df.empty: print(f"エラー: キーフレーム {key_frame_index} に観測点がありません。"); return key_frame_index, {}, {}
    
    obs_rows = list(obs_df.itertuples(index=False, name=None)) # [(frame, t, raw_id, x, y, z), ...]
    obs_coords = np.array([row[3:] for row in obs_rows])
    obs_raw_ids = [int(row[2]) for row in obs_rows]
    
    template_coords_arr = np.array([template_geometry[tid] for tid in template_ids])
    
    if len(obs_coords) > len(template_ids):
        template_centroid = template_coords_arr.mean(axis=0); distances = np.linalg.norm(obs_coords - template_centroid, axis=1)
        closest_n_indices = np.argsort(distances)[:len(template_ids)]
        # 観測データ側を絞り込む
        obs_coords = obs_coords[closest_n_indices, :]
        obs_raw_ids = [obs_raw_ids[i] for i in closest_n_indices]
        print(f"  警告: キーフレーム観測点多すぎ ({len(obs_df)}個)。近い {len(template_ids)} 点を選択。")
    
    if obs_coords.shape[0] < 3: print(f"エラー: キーフレーム観測点少なすぎ ({len(obs_coords)}個)。"); return key_frame_index, {}, {}
    
    try: cost_matrix = cdist(template_coords_arr, obs_coords); template_indices, obs_indices = linear_sum_assignment(cost_matrix)
    except ValueError as e: print(f"エラー: キーフレームのマッチング失敗: {e}"); return key_frame_index, {}, {}

    found_count = 0
    for r_idx, c_idx in zip(template_indices, obs_indices):
        if cost_matrix[r_idx, c_idx] < MATCHING_THRESHOLD_MM:
            tid = template_ids[r_idx]
            raw_id = obs_raw_ids[c_idx]
            pos = obs_coords[c_idx]
            
            initial_pose_map[tid] = pos
            if tid != raw_id: # テンプレートIDと生IDが異なる場合のみマップ登録
                 id_map[raw_id] = tid
            found_count += 1
            
    if found_count < 3: print(f"エラー: キーフレームの特定マーカー不足 ({found_count}個)。"); return key_frame_index, {}, {}
    print(f"  -> フレーム {key_frame_index} (実時間 {actual_time:.3f}s) を追跡起点とします。({found_count}/{NUM_MARKERS} 個特定)")
    print(f"  初期IDマップ: {id_map}")
    return key_frame_index, initial_pose_map, id_map


# --- ▼▼▼【関数修正】生ID優先ロジック + 永続IDマップ (v23) ▼▼▼ ---
def process_rigid_tracking_v3(df_long_walk: pd.DataFrame, keyframe_frame: int, initial_pose_map: dict,
                              template_geometry: dict, template_ids: list, initial_id_map: dict) -> pd.DataFrame:
    """
    生ID/IDマップ優先 + 近傍点探索(マップ登録) + 剛体補完 で追跡
    """
    print("歩行区間の処理を開始 (生ID優先 + 近傍点探索 + 剛体補完)...")

    # --- 状態変数初期化 ---
    last_known_pose_map = initial_pose_map.copy() # {tid: pos}
    id_map = initial_id_map.copy() # ★ 永続IDマップ
    template_id_set = set(template_ids) # 高速検索用
    
    corrected_rows = []
    
    unique_frames = sorted(df_long_walk['Frame'].unique())
    start_index = 0
    if keyframe_frame in unique_frames:
        start_index = unique_frames.index(keyframe_frame)
    else:
        print(f"警告: キーフレーム {keyframe_frame} が歩行区間データに見つかりません。最初から処理します。")

    print(f"  {len(unique_frames) - start_index} フレームを追跡します...")
    
    # --- フレームループ ---
    for frame_idx, frame in enumerate(unique_frames[start_index:], start=start_index):
        if frame_idx % 1000 == 0:
            print(f"  Processing frame {frame_idx}/{len(unique_frames)}...")

        frame_group = df_long_walk[df_long_walk['Frame'] == frame]
        if frame_group.empty: continue
        time_scalar = frame_group['Time'].iloc[0]
        obs_df = filter_plausible_markers(frame_group, PLAUSIBLE_BOUNDS)

        final_pose_map = {} # {tid: pos} このフレームの最終座標
        
        # 1. テンプレートID / IDマップ優先マッチング
        unassigned_obs_rows = [] # (raw_id, pos)
        
        if not obs_df.empty:
            for row in obs_df.itertuples():
                raw_id = int(row.id)
                pos = np.array([row.x, row.y, row.z])
                
                if raw_id in template_id_set:
                    # ステップA: 生IDがテンプレートIDと一致
                    final_pose_map[raw_id] = pos
                elif raw_id in id_map:
                    # ステップB: 生IDが既知のマップにある
                    tid = id_map[raw_id]
                    if tid not in final_pose_map: # 複数の生IDが同じTIDにマップされるのを防止
                         final_pose_map[tid] = pos
                    # else: 既にこのTIDは埋まっているので、この観測点は無視
                else:
                    # ステップC: ID不明の観測点
                    unassigned_obs_rows.append((raw_id, pos))

        # 2. 近傍点探索 (ID不明マーカーの割り当て)
        missing_template_ids = [tid for tid in template_ids if tid not in final_pose_map]
        
        if missing_template_ids and unassigned_obs_rows:
            last_pose_tids_missing = []
            last_pose_array_list = []
            for tid in missing_template_ids:
                if tid in last_known_pose_map and not np.isnan(last_known_pose_map[tid]).any():
                    last_pose_tids_missing.append(tid)
                    last_pose_array_list.append(last_known_pose_map[tid])

            if last_pose_tids_missing:
                last_pose_array = np.array(last_pose_array_list)
                unknown_obs_ids, unknown_obs_coords_list = zip(*unassigned_obs_rows)
                unknown_obs_coords = np.array(unknown_obs_coords_list)
                
                if last_pose_array.ndim == 2 and unknown_obs_coords.ndim == 2:
                    try:
                        cost_matrix = cdist(last_pose_array, unknown_obs_coords)
                        last_indices, obs_indices = linear_sum_assignment(cost_matrix)
                        
                        used_obs_indices = set() # 複数のTIDが同じ観測点にマッピングされるのを防ぐ
                        for last_idx, obs_idx in zip(last_indices, obs_indices):
                            if obs_idx not in used_obs_indices and \
                               obs_idx < unknown_obs_coords.shape[0] and \
                               cost_matrix[last_idx, obs_idx] < MATCHING_THRESHOLD_MM:
                                
                                tid = last_pose_tids_missing[last_idx]
                                raw_id = unknown_obs_ids[obs_idx]
                                pos = unknown_obs_coords[obs_idx]
                                
                                final_pose_map[tid] = pos # 最終ポーズに採用
                                id_map[raw_id] = tid # ★ ステップD: IDマップに登録
                                used_obs_indices.add(obs_idx) # この観測点は使用済み
                                
                    except ValueError as e:
                        print(f"警告: 近傍点探索マッチングエラー @F{frame}: {e}")

        # 3. 剛体補完 (残りの欠損マーカー)
        current_R, current_t = np.identity(3), np.zeros(3)
        kabsch_input_template, kabsch_input_final = [], []
        
        # (1)と(2)で確定したマーカーを使って R, t を計算
        for tid, obs_pos in final_pose_map.items():
             static_pos = template_geometry.get(tid)
             if static_pos is not None and not np.isnan(static_pos).any():
                  kabsch_input_template.append(static_pos)
                  kabsch_input_final.append(obs_pos)
        
        if len(kabsch_input_final) >= 3:
            new_R, new_t = kabsch_solve(kabsch_input_template, kabsch_input_final)
            if np.isfinite(new_R).all() and np.isfinite(new_t).all():
                current_R, current_t = new_R, new_t
        else: # 観測点が少ない場合 -> 前回のポーズマップから R, t を推定
            last_pose_items_valid = [(tid, pos) for tid, pos in last_known_pose_map.items() if pos is not None and not np.isnan(pos).any() and tid in template_geometry]
            if len(last_pose_items_valid) >= 3:
                 last_tids_valid, last_pos_valid = zip(*last_pose_items_valid)
                 static_pos_valid = [template_geometry[tid] for tid in last_tids_valid]
                 new_R, new_t = kabsch_solve(static_pos_valid, last_pos_valid)
                 if np.isfinite(new_R).all() and np.isfinite(new_t).all():
                      current_R, current_t = new_R, new_t

        # 4. 最終座標決定 & 格納
        for tid in template_ids:
            if tid in final_pose_map:
                # (1) または (2) で見つかった
                final_pos = final_pose_map[tid]
            else: # (3) 剛体補完
                static_pos = template_geometry.get(tid)
                if static_pos is not None and not np.isnan(static_pos).any():
                     final_pos = current_R @ static_pos + current_t
                else:
                     final_pos = np.full(3, np.nan)
                final_pose_map[tid] = final_pos # 補完した値もマップに追加
            
            corrected_rows.append((frame, time_scalar, int(tid), *final_pos))

        # 5. 状態更新
        last_known_pose_map = final_pose_map

    print("歩行区間の追跡完了。")
    print(f"  最終的なIDマップ ({len(id_map)}件): {id_map}") # ★ 最終マップを出力
    if not corrected_rows: return pd.DataFrame()
    return pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])
# --- ▲▲▲ 関数修正ここまで ▲▲▲ ---


def create_full_scaffold(df_long: pd.DataFrame, all_template_ids: list) -> pd.DataFrame:
    """全フレーム x 全ID の抜け殻DF作成。"""
    if df_long.empty: return pd.DataFrame()
    all_frames_df = df_long[['Frame', 'Time']].drop_duplicates().sort_values('Frame').reset_index(drop=True)
    all_ids_df = pd.DataFrame({'id': all_template_ids})
    df_scaffold = pd.merge(all_frames_df, all_ids_df, how='cross')
    return df_scaffold.sort_values(['Frame', 'id']).reset_index(drop=True)

def fill_static_zones(df_full_scaffold: pd.DataFrame, static_template_geometry: dict, cfg: dict) -> pd.DataFrame:
    """静止区間をテンプレートで埋め、歩行区間・隙間区間をNaNにする。"""
    print("静止区間をテンプレート座標で補完...")
    if df_full_scaffold.empty: return df_full_scaffold

    template_df = pd.DataFrame.from_dict(static_template_geometry, orient='index', columns=['x_template', 'y_template', 'z_template'])
    template_df['id'] = template_df.index.astype(int)
    template_df = template_df.dropna()

    df_full_scaffold['id'] = df_full_scaffold['id'].astype(int)
    df_filled = pd.merge(df_full_scaffold, template_df, on='id', how='left', suffixes=('', '_tpl'))

    t_static_end = cfg.get('STANDING_END_TIME', 0.0)
    t_walk_start = cfg.get('WALKING_START_TIME', 0.0)
    
    is_static = df_filled['Time'] <= t_static_end
    
    df_filled['x'] = np.where(is_static, df_filled['x_template'], np.nan)
    df_filled['y'] = np.where(is_static, df_filled['y_template'], np.nan)
    df_filled['z'] = np.where(is_static, df_filled['z_template'], np.nan)

    df_filled = df_filled[['Frame', 'Time', 'id', 'x', 'y', 'z']]
    print("静止区間の補完完了。歩行区間と隙間区間はNaNです。")
    return df_filled


def final_interpolate_gaps(df_merged: pd.DataFrame, all_template_ids: list) -> pd.DataFrame:
    """隙間区間と残ったNaNを時間軸で線形補間。"""
    print("最終線形補間を実行...")
    if df_merged.empty: return df_merged
    df_interpolated_list = []
    df_merged = df_merged.sort_values('Time')
    for tid in all_template_ids:
        marker_df = df_merged[df_merged['id'] == tid].copy()
        if marker_df.empty: continue
        marker_df = marker_df.drop_duplicates(subset='Time').set_index('Time')
        valid_points = marker_df[['x','y','z']].dropna()
        if len(valid_points) >= 2:
            try: marker_df[['x', 'y', 'z']] = marker_df[['x', 'y', 'z']].interpolate(method='index', limit_direction='both')
            except ValueError as e: print(f"警告: ID {tid} の補間エラー: {e}")
        elif len(valid_points) == 1:
             marker_df[['x', 'y', 'z']] = marker_df[['x', 'y', 'z']].fillna(value=valid_points.iloc[0].to_dict())
        df_interpolated_list.append(marker_df.reset_index())
    if not df_interpolated_list: return pd.DataFrame()
    df_final = pd.concat(df_interpolated_list).sort_values(['Frame', 'id']).reset_index(drop=True)
    if df_final[['x','y','z']].isnull().values.any():
        print("警告: 最終補間後もNaNあり。")
    else: print("最終補間完了。")
    return df_final


# =============================================================================
# メイン実行ブロック
# =============================================================================

if __name__ == "__main__":
    # --- 1. データ読み込み ---
    df_long = load_opti_data_to_long_robust(OPTITRACK_CSV_PATH)
    if df_long is None: exit("エラー: データ読み込み失敗。")

    # --- 2. テンプレート作成 ---
    template_geometry, template_ids = build_static_template(
        df_long, STANDING_START_TIME, WALKING_START_TIME, NUM_MARKERS
    )
    if template_geometry is None: exit("エラー: テンプレート作成失敗。")

    # --- 3. キーフレーム特定 (自動) ---
    key_frame_index, initial_pose_map, initial_id_map = get_keyframe_pose( # ★ id_map を受け取る
        df_long, WALKING_START_TIME, template_geometry, template_ids
    )
    if initial_pose_map is None: exit("エラー: キーフレーム処理失敗。")

    # --- 4. 歩行区間データ抽出 ---
    df_long_walk = df_long[
        (df_long['Time'] >= WALKING_START_TIME) & (df_long['Time'] <= WALKING_STOP_TIME)
    ].copy()
    if df_long_walk.empty: exit(f"エラー: 歩行区間 ({WALKING_START_TIME}s-{WALKING_STOP_TIME}s) データが空。")

    # --- 5. 追跡実行 (v3: 生ID優先) ---
    df_walk_processed = process_rigid_tracking_v3( # ★ 関数名変更
        df_long_walk, key_frame_index, initial_pose_map,
        template_geometry, template_ids, initial_id_map # ★ id_map を渡す
    )
    if df_walk_processed.empty: exit("エラー: 歩行区間の追跡処理結果が空。")
    if df_walk_processed[['x','y','z']].isnull().values.any(): print("警告: 歩行区間の追跡結果にNaNあり。")

    # --- 6. 全区間の結合と補完 ---
    print("全フレームデータの結合と静止/隙間区間補完を開始...")
    df_scaffold = create_full_scaffold(df_long, template_ids)
    
    temp_cfg = { # fill_static_zones 用の簡易辞書
        'STANDING_END_TIME': WALKING_START_TIME,
        'WALKING_START_TIME': WALKING_START_TIME,
        'WALKING_STOP_TIME': WALKING_STOP_TIME
    }
    df_filled = fill_static_zones(df_scaffold, template_geometry, temp_cfg)
    
    df_merged = pd.merge(
        df_filled, df_walk_processed[['Frame', 'id', 'x', 'y', 'z']],
        on=['Frame', 'id'], how='left', suffixes=('_filled', '_walk')
    )
    df_merged['x'] = df_merged['x_walk'].fillna(df_merged['x_filled'])
    df_merged['y'] = df_merged['y_walk'].fillna(df_merged['y_filled'])
    df_merged['z'] = df_merged['z_walk'].fillna(df_merged['z_filled'])
    df_merged = df_merged[['Frame', 'Time', 'id', 'x', 'y', 'z']]

    # --- 7. 最終補間 ---
    df_final = final_interpolate_gaps(df_merged, template_ids)

    # --- 8. 保存 ---
    if not df_final.empty:
        try:
            output_dir = os.path.dirname(OUTPUT_CSV_PATH)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            df_final.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
            print(f"\n処理結果を保存しました: {OUTPUT_CSV_PATH} ({len(df_final)} 行)")
        except Exception as e: print(f"エラー: 結果の保存に失敗。 {e}")
    else: print("\nエラー: 最終結果が空。ファイルは保存されません。")
    print(f"\n--- 処理完了 ---")