# =============================================================================
# opti_edit_E.py (v20.2 - Fix KeyError in fill_static_zones)
#
# 概要:
# 歩行位相に応じて追跡・補完アルゴリズムを切り替える。
# - Rigid Phase (configで定義): opti_edit_C (Kabschベース) 方式。
# - Flexible Phase (configで定義): opti_edit_D (近傍探索ベース) 方式。
# fill_static_zones 関数の列名参照を修正。
# =============================================================================

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import config # 設定ファイルをインポート

# --- 定数 ---
FLEXIBLE_COMPLETION_RANGE = config.PHASE_WEIGHTING['flexible_phase_range']

# =============================================================================
# ヘルパー関数群 (v20と同様のため省略)
# =============================================================================
def load_opti_data(file_path: str) -> pd.DataFrame | None:
    # (... v20 と同様 ...)
    if not os.path.exists(file_path): print(f"エラー: ファイルが見つかりません: {file_path}"); return None
    print(f"'{os.path.basename(file_path)}' を読み込み中..."); rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダー
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
                except (ValueError, IndexError): continue # スキップ
        if not rows: print("エラー: 有効なデータ行が読み込めませんでした。"); return None
        df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"]); print(f"読み込み成功: {len(df)} 行"); return df
    except Exception as e: print(f"ファイル読み込みエラー: {e}"); return None

def load_gait_cycles(file_path: str, time_offset: float) -> pd.DataFrame | None:
    # (... v20 と同様 ...)
    try:
        df_cycles = pd.read_csv(file_path); required_cols = ['hs_time', 'next_hs_time', 'hs_frame']
        if not all(col in df_cycles.columns for col in required_cols): print(f"エラー: 歩行周期ファイル列不足: {file_path}"); return None
        df_cycles['opti_start_time'] = df_cycles['hs_time'] - time_offset; df_cycles['opti_end_time'] = df_cycles['next_hs_time'] - time_offset
        df_cycles['duration'] = df_cycles['opti_end_time'] - df_cycles['opti_start_time']
        initial_count = len(df_cycles); valid_cycles = df_cycles[df_cycles['duration'] > 1e-6].copy(); removed_count = initial_count - len(valid_cycles)
        if removed_count > 0: print(f"警告: {removed_count} 個の異常な歩行周期を除外。")
        if valid_cycles.empty: print("エラー: 有効な歩行周期なし。"); return None
        print(f"歩行周期データ読み込み成功: {len(valid_cycles)} 周期")
        valid_cycles = valid_cycles.loc[~valid_cycles['hs_frame'].duplicated(keep='first')]
        if valid_cycles['hs_frame'].isnull().any(): print("警告: hs_frame 欠損値あり。除外します。"); valid_cycles = valid_cycles.dropna(subset=['hs_frame'])
        if valid_cycles.empty: print("エラー: hs_frame 欠損除外後、有効周期なし。"); return None
        return valid_cycles.set_index('hs_frame')
    except Exception as e: print(f"歩行周期ファイルの読み込み/処理エラー: {e}"); return None


def load_mean_cycle(file_path: str) -> pd.DataFrame | None:
    # (... v20 と同様 ...)
    if not file_path or pd.isna(file_path): print("情報: 平均周期ファイル指定なし。スキップ。"); return None
    try:
        df_mean = pd.read_csv(file_path); print(f"平均歩行周期データ読み込み成功: {file_path}")
        if 'gait_cycle_%' not in df_mean.columns or 'id' not in df_mean.columns: print("エラー: 平均周期ファイル列不足。"); return None
        return df_mean
    except Exception as e: print(f"平均歩行周期ファイルの読み込みエラー: {e}"); return None


def filter_plausible(frame_df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    # (... v20 と同様 ...)
    if frame_df.empty: return frame_df
    try: mask = (frame_df['x'].between(*bounds['x']) & frame_df['y'].between(*bounds['y']) & frame_df['z'].between(*bounds['z'])); return frame_df[mask]
    except KeyError: print(f"エラー: filter_plausible - 列 x, y, z 不足。"); return pd.DataFrame()


def kabsch_solve(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (... v20 と同様 ...)
    A_arr, B_arr = np.asarray(A), np.asarray(B)
    if A_arr.ndim != 2 or B_arr.ndim != 2 or A_arr.shape[1] != 3 or B_arr.shape[1] != 3: return np.identity(3), np.zeros(3)
    if A_arr.shape[0] < 3 or B_arr.shape[0] < 3 or A_arr.shape != B_arr.shape: return np.identity(3), np.zeros(3)
    if np.isnan(A_arr).any() or np.isnan(B_arr).any(): return np.identity(3), np.zeros(3)
    try:
        cA, cB = A_arr.mean(axis=0), B_arr.mean(axis=0); H = (A_arr - cA).T @ (B_arr - cB); U, _, Vt = np.linalg.svd(H); R_mat = Vt.T @ U.T
        if np.linalg.det(R_mat) < 0: Vt[-1, :] *= -1; R_mat = Vt.T @ U.T
        t_vec = cB - R_mat @ cA; return R_mat, t_vec
    except Exception as e: print(f"エラー: Kabsch計算エラー: {e}"); return np.identity(3), np.zeros(3)


def _calculate_segment_properties(template_geometry: dict, segments: dict, hierarchy: dict, all_ids: list) -> tuple | None:
    # (... v20 と同様 ...)
    centroids, marker_rel_vecs, centroid_rel_vecs = {}, {}, {}; segment_of_tid = {tid: seg for seg, tids in segments.items() for tid in tids}
    try:
        for seg_name, tids in segments.items():
            valid_pos = [p for tid in tids if (p := template_geometry.get(tid)) is not None and not np.isnan(p).any()]; centroids[seg_name] = np.mean(valid_pos, axis=0) if valid_pos else np.zeros(3)
        for tid in all_ids:
            pos = template_geometry.get(tid); seg_name = segment_of_tid.get(tid)
            if pos is not None and not np.isnan(pos).any() and seg_name and seg_name in centroids: marker_rel_vecs[tid] = pos - centroids[seg_name]
        for seg_name, parent_name in hierarchy.items():
            if parent_name and seg_name in centroids and parent_name in centroids: centroid_rel_vecs[seg_name] = centroids[seg_name] - centroids[parent_name]
            elif seg_name in centroids: centroid_rel_vecs[seg_name] = np.zeros(3)
        return centroids, marker_rel_vecs, centroid_rel_vecs, segment_of_tid
    except Exception as e: print(f"セグメントプロパティ計算エラー: {e}"); return None

def build_hybrid_templates(df_long: pd.DataFrame, df_mean_cycle: pd.DataFrame | None, cfg: dict) -> tuple[dict | None, tuple | None, tuple | None]:
    # (... v20 と同様 ...)
    print(f"静止区間 ({cfg['STATIC_START']}s - {cfg['STATIC_END']}s) から静止テンプレートを作成..."); all_template_ids = [tid for sublist in cfg['SEGMENTS'].values() for tid in sublist]
    if not all_template_ids: print("エラー: configにSEGMENTS未定義。"); return None, None, None
    static_df = df_long[(df_long['Time'] >= cfg['STATIC_START']) & (df_long['Time'] <= cfg['STATIC_END'])]
    if static_df.empty: print(f"エラー: 静止区間データなし。"); return None, None, None
    mean_pos_static_df = static_df[static_df['id'].isin(all_template_ids)].groupby('id')[['x','y','z']].mean()
    if len(mean_pos_static_df) < len(all_template_ids) * 0.7: print(f"エラー: 静止マーカー不足。"); return None, None, None
    template_geometry_static = {int(mid): row.to_numpy() for mid, row in mean_pos_static_df.iterrows()}; [template_geometry_static.setdefault(tid, np.full(3, np.nan)) for tid in all_template_ids]
    static_templates_tuple = _calculate_segment_properties(template_geometry_static, cfg['SEGMENTS'], config.CHAIN_HIERARCHY, all_template_ids)
    if static_templates_tuple is None: print("エラー: 静止テンプレートプロパティ計算失敗。"); return None, None, None
    print("静止テンプレート作成完了。")
    swing_templates_tuple = None
    if df_mean_cycle is not None:
        print("平均歩行周期データからSwingテンプレートを作成...")
        try:
            flex_start, flex_end = config.PHASE_WEIGHTING['flexible_phase_range']; swing_phase_target = (flex_start + flex_end) / 2
            target_perc_row = df_mean_cycle.iloc[(df_mean_cycle['gait_cycle_%'] - swing_phase_target).abs().argsort()[:1]]
            if not target_perc_row.empty:
                target_perc = target_perc_row['gait_cycle_%'].values[0]; swing_df = df_mean_cycle[np.isclose(df_mean_cycle['gait_cycle_%'], target_perc)]
                if not swing_df.empty:
                    template_geometry_swing = {int(row['id']): row[['x','y','z']].to_numpy() for _, row in swing_df.iterrows()}; [template_geometry_swing.setdefault(tid, np.full(3, np.nan)) for tid in all_template_ids]
                    swing_templates_tuple = _calculate_segment_properties(template_geometry_swing, cfg['SEGMENTS'], config.CHAIN_HIERARCHY, all_template_ids)
                    if swing_templates_tuple: print("Swingテンプレート作成完了。")
                    else: print(f"警告: Swingテンプレートプロパティ計算失敗。")
                else: print(f"警告: Swingテンプレート - データなし @ {swing_phase_target:.1f}% 付近。")
            else: print("警告: Swingテンプレート - 平均周期データ空または不正。")
        except Exception as e: print(f"Swingテンプレート作成エラー: {e}")
    else: print("情報: 平均周期データなし。Swingテンプレート作成不可。")
    return template_geometry_static, static_templates_tuple, swing_templates_tuple


def get_keyframe_data(df_long: pd.DataFrame, cfg: dict) -> tuple[int | None, dict | None]:
    # (... v20 と同様 ...)
    print(f"キーフレーム (t~{cfg['KEYFRAME_TIME']:.3f}s) のデータを取得...");
    try: key_frame_row = df_long.iloc[(df_long['Time'] - cfg['KEYFRAME_TIME']).abs().argmin()]; key_frame_index = int(key_frame_row['Frame']); actual_time = key_frame_row['Time']
    except (IndexError, KeyError): print("エラー: キーフレーム時刻付近データなし。"); return None, None
    key_frame_df = df_long[df_long['Frame'] == key_frame_index]
    if key_frame_df.empty: print(f"エラー: フレーム {key_frame_index} データなし。"); return None, None
    print(f"  -> フレーム {key_frame_index} (実時間 {actual_time:.3f}s) を使用。")
    key_frame_pos_map = {int(row.id): (row.x, row.y, row.z) for row in key_frame_df.itertuples()}; initial_pose_map = {}
    found_count = 0; all_template_ids = [tid for sublist in cfg['SEGMENTS'].values() for tid in sublist]
    for template_id in all_template_ids:
        raw_id_found = next((r_id for r_id, t_id in cfg['KEYFRAME_MAP'].items() if t_id == template_id), None)
        if raw_id_found and raw_id_found in key_frame_pos_map: initial_pose_map[template_id] = np.array(key_frame_pos_map[raw_id_found]); found_count += 1
    if found_count == 0: print("エラー: キーフレームマーカーなし。"); return None, None
    print(f"  {found_count}/{len(all_template_ids)} 個のマーカーで追跡開始。不足分は補完。"); return key_frame_index, initial_pose_map


def calculate_gait_phase(current_time: float, df_cycles_indexed: pd.DataFrame | None) -> tuple[pd.Series | None, float]:
    # (... v20 と同様 ...)
    if df_cycles_indexed is None or df_cycles_indexed.empty: return None, 0.0
    relevant_cycles = df_cycles_indexed[df_cycles_indexed['opti_start_time'] <= current_time]
    if relevant_cycles.empty: return None, 0.0
    current_cycle = relevant_cycles.iloc[-1]; cycle_duration = current_cycle['duration']
    if cycle_duration <= 1e-6: return current_cycle, 0.0
    time_in_cycle = current_time - current_cycle['opti_start_time']; phase_perc = np.clip((time_in_cycle / cycle_duration) * 100.0, 0.0, 100.0)
    return current_cycle, phase_perc


def get_tracking_mode(phase_perc: float) -> str:
    """歩行位相(%)に基づいて追跡モード ('rigid' or 'flexible') を返す。"""
    for r_start, r_end in config.PHASE_WEIGHTING['rigid_phase_ranges']:
        if r_start <= phase_perc <= r_end:
            return 'rigid'
    return 'flexible' # rigidでなければflexible

def predict_missing_marker_pos_adaptive(tid: int, current_centroids: dict, phase_perc: float,
                                        static_templates: tuple | None, swing_templates: tuple | None) -> np.ndarray:
    # (... v20 と同様 ...)
    flex_s, flex_e = FLEXIBLE_COMPLETION_RANGE; is_flexible_phase = flex_s <= phase_perc <= flex_e
    templates_to_use = swing_templates if (is_flexible_phase and swing_templates) else static_templates
    if templates_to_use is None: return np.full(3, np.nan)
    _, marker_rel_vecs, _, segment_of_tid = templates_to_use
    seg_name = segment_of_tid.get(tid); rel_vector = marker_rel_vecs.get(tid); centroid = current_centroids.get(seg_name)
    if seg_name and rel_vector is not None and centroid is not None and not np.isnan(centroid).any(): return centroid + rel_vector
    else: return np.full(3, np.nan)


# =============================================================================
# メイン処理関数 (位相別アルゴリズム切り替え)
# =============================================================================

def process_phase_switched_tracking(df_long_walk: pd.DataFrame, df_cycles_indexed: pd.DataFrame | None,
                                    initial_pose_map: dict, template_geometry_static: dict,
                                    static_templates: tuple, swing_templates: tuple | None, cfg: dict) -> pd.DataFrame:
    """歩行位相に応じて opti_edit_C (rigid) と opti_edit_D (flexible) のロジックを切り替える。"""
    print("歩行区間の処理を開始 (位相別アルゴリズム切り替え)...")
    if static_templates is None: print("エラー: 静止テンプレートが無効。"); return pd.DataFrame()
    all_template_ids = list(initial_pose_map.keys())
    if not all_template_ids: print("エラー: 初期ポーズIDが空。"); return pd.DataFrame()

    last_known_pose_map = initial_pose_map.copy(); last_known_R, last_known_t = np.eye(3), np.zeros(3)
    current_centroids = static_templates[0].copy() if static_templates[0] else {}
    corrected_rows = []
    unique_frames = sorted(df_long_walk['Frame'].unique()); start_index = 0
    try:
        key_frame_row = df_long_walk.iloc[(df_long_walk['Time'] - cfg['KEYFRAME_TIME']).abs().argmin()]
        keyframe_frame = int(key_frame_row['Frame'])
        if keyframe_frame in unique_frames: start_index = unique_frames.index(keyframe_frame)
    except (IndexError, KeyError): print(f"警告: キーフレーム付近データなし。")
    print(f"  {len(unique_frames) - start_index} フレームを追跡...")

    for frame_idx, frame in enumerate(unique_frames[start_index:], start=start_index):
        if frame_idx % 500 == 0: print(f"  Processing frame {frame_idx}/{len(unique_frames)}...")
        frame_group = df_long_walk[df_long_walk['Frame'] == frame];
        if frame_group.empty: continue
        time_scalar = frame_group['Time'].iloc[0]; obs_df = filter_plausible(frame_group, cfg['PLAUSIBLE_BOUNDS'])
        obs_coords = obs_df[['x','y','z']].values if not obs_df.empty else np.empty((0,3))
        _, phase_perc = calculate_gait_phase(time_scalar, df_cycles_indexed); tracking_mode = get_tracking_mode(phase_perc)
        observed_matches_map = {}; final_pose_map = {}; kabsch_input_template, kabsch_input_final = [], []

        if tracking_mode == 'rigid':
            pred_coords_list, pred_ids_list = [], []
            for tid in all_template_ids:
                static_pos = template_geometry_static.get(tid)
                if static_pos is not None and not np.isnan(static_pos).any(): pred_pos = last_known_R @ static_pos + last_known_t; pred_coords_list.append(pred_pos); pred_ids_list.append(tid)
            if pred_coords_list and obs_coords.shape[0] > 0:
                pred_coords_arr = np.array(pred_coords_list)
                if pred_coords_arr.ndim == 2 and obs_coords.ndim == 2:
                    try:
                        cost_matrix = cdist(pred_coords_arr, obs_coords); pred_indices, obs_indices = linear_sum_assignment(cost_matrix)
                        for r, c in zip(pred_indices, obs_indices):
                            if c < obs_coords.shape[0] and cost_matrix[r, c] < config.MATCHING_THRESHOLD_MM: tid = pred_ids_list[r]; observed_matches_map[tid] = obs_coords[c]
                    except ValueError as e: print(f"警告: Rigidマッチングエラー @F{frame}: {e}")
            for tid in all_template_ids: final_pose_map[tid] = observed_matches_map.get(tid, np.full(3, np.nan))
            for tid in all_template_ids:
                static_pos = template_geometry_static.get(tid); target_pos = observed_matches_map.get(tid)
                if static_pos is not None and not np.isnan(static_pos).any() and target_pos is not None and not np.isnan(target_pos).any():
                    kabsch_input_template.append(static_pos); kabsch_input_final.append(target_pos)
            if len(kabsch_input_final) >= 3:
                new_R, new_t = kabsch_solve(kabsch_input_template, kabsch_input_final)
                if np.isfinite(new_R).all() and np.isfinite(new_t).all(): last_known_R, last_known_t = new_R, new_t
        else: # tracking_mode == 'flexible'
            last_pose_items = [(tid, pos) for tid, pos in last_known_pose_map.items() if pos is not None and not np.isnan(pos).any()]
            if last_pose_items and obs_coords.shape[0] > 0:
                last_pose_tids, last_pose_array_list = zip(*last_pose_items); last_pose_array = np.array(last_pose_array_list)
                if last_pose_array.ndim == 2 and obs_coords.ndim == 2:
                     try:
                          cost_matrix = cdist(last_pose_array, obs_coords); last_indices, obs_indices = linear_sum_assignment(cost_matrix)
                          for last_idx, obs_idx in zip(last_indices, obs_indices):
                               if obs_idx < obs_coords.shape[0] and cost_matrix[last_idx, obs_idx] < config.MATCHING_THRESHOLD_MM: tid = last_pose_tids[last_idx]; observed_matches_map[tid] = obs_coords[obs_idx]
                     except ValueError as e: print(f"警告: Flexibleマッチングエラー @F{frame}: {e}")
            for tid in all_template_ids: final_pose_map[tid] = observed_matches_map.get(tid, np.full(3, np.nan))
            for tid in all_template_ids:
                static_pos = template_geometry_static.get(tid); final_pos_for_kabsch = final_pose_map.get(tid)
                if static_pos is not None and not np.isnan(static_pos).any() and final_pos_for_kabsch is not None and not np.isnan(final_pos_for_kabsch).any():
                    kabsch_input_template.append(static_pos); kabsch_input_final.append(final_pos_for_kabsch)
            if len(kabsch_input_final) >= 3:
                new_R, new_t = kabsch_solve(kabsch_input_template, kabsch_input_final)
                if np.isfinite(new_R).all() and np.isfinite(new_t).all(): last_known_R, last_known_t = new_R, new_t
        temp_centroids = {}
        for seg_name in config.PROCESSING_ORDER:
            found_markers_pos = [observed_matches_map[tid] for tid in cfg['SEGMENTS'].get(seg_name, []) if tid in observed_matches_map]
            if found_markers_pos: temp_centroids[seg_name] = np.mean(found_markers_pos, axis=0)
            else:
                parent_name = config.CHAIN_HIERARCHY.get(seg_name)
                if parent_name and parent_name in temp_centroids and static_templates:
                    parent_centroid = temp_centroids[parent_name]; rel_vec = static_templates[2].get(seg_name, np.zeros(3))
                    if parent_centroid is not None and not np.isnan(parent_centroid).any(): temp_centroids[seg_name] = parent_centroid + rel_vec
                    else: temp_centroids[seg_name] = current_centroids.get(seg_name, np.zeros(3))
                else: temp_centroids[seg_name] = current_centroids.get(seg_name, np.zeros(3))
        current_centroids = temp_centroids
        for tid in all_template_ids:
            final_pos = final_pose_map.get(tid)
            if final_pos is None or np.isnan(final_pos).any(): final_pos = predict_missing_marker_pos_adaptive(tid, current_centroids, phase_perc, static_templates, swing_templates)
            corrected_rows.append((frame, time_scalar, tid, *final_pos))
            final_pose_map[tid] = final_pos # 補完後の値で更新
        last_known_pose_map = final_pose_map

    print("歩行区間の追跡完了。")
    if not corrected_rows: return pd.DataFrame()
    return pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])

# =============================================================================
# 出力用関数群
# =============================================================================
def create_full_scaffold(df_long: pd.DataFrame, all_template_ids: list) -> pd.DataFrame:
    """全フレーム x 全ID の抜け殻DF作成。"""
    if df_long.empty: return pd.DataFrame()
    all_frames_df = df_long[['Frame', 'Time']].drop_duplicates().sort_values('Frame').reset_index(drop=True)
    all_ids_df = pd.DataFrame({'id': all_template_ids})
    df_scaffold = pd.merge(all_frames_df, all_ids_df, how='cross')
    return df_scaffold.sort_values(['Frame', 'id']).reset_index(drop=True)

# --- ▼▼▼【関数修正】KeyError対策 ▼▼▼ ---
def fill_static_zones(df_full_scaffold: pd.DataFrame, static_template_geometry: dict, cfg: dict) -> pd.DataFrame:
    """静止区間と隙間区間を特定し、静止区間のみテンプレートで埋める。"""
    print("静止区間をテンプレート座標で補完...")
    if df_full_scaffold.empty: return df_full_scaffold

    template_df = pd.DataFrame.from_dict(static_template_geometry, orient='index', columns=['x_template', 'y_template', 'z_template']) # 列名を変更
    template_df['id'] = template_df.index.astype(int)
    template_df = template_df.dropna() # NaNテンプレートを除外

    df_full_scaffold['id'] = df_full_scaffold['id'].astype(int)
    # left merge: scaffold にテンプレート座標を追加 (存在しないIDはNaNのまま)
    df_filled = pd.merge(df_full_scaffold, template_df, on='id', how='left')

    # 各区間の Time を取得 (存在しない場合はデフォルト値)
    t_static_end = cfg.get('T1_STATIC_END', 0.0)
    t_walk_start = cfg.get('T1_WALK_START', 0.0)
    t_walk_end = cfg.get('T2_WALK_END', float('inf'))
    t_static_start = cfg.get('T2_STATIC_START', float('inf'))

    # 各区間フラグ
    is_static1 = df_filled['Time'] <= t_static_end
    is_gap1 = (df_filled['Time'] > t_static_end) & (df_filled['Time'] < t_walk_start)
    is_walk = (df_filled['Time'] >= t_walk_start) & (df_filled['Time'] <= t_walk_end)
    is_gap2 = (df_filled['Time'] > t_walk_end) & (df_filled['Time'] < t_static_start)
    is_static2 = df_filled['Time'] >= t_static_start

    # 静止区間にテンプレート座標を適用 (NaNでない場合のみ)
    df_filled['x'] = np.where(is_static1 | is_static2, df_filled['x_template'], np.nan)
    df_filled['y'] = np.where(is_static1 | is_static2, df_filled['y_template'], np.nan)
    df_filled['z'] = np.where(is_static1 | is_static2, df_filled['z_template'], np.nan)

    # 不要なテンプレート列を削除
    df_filled = df_filled[['Frame', 'Time', 'id', 'x', 'y', 'z']]

    print("静止区間の補完完了。歩行区間と隙間区間はNaNです。")
    return df_filled
# --- ▲▲▲ 関数修正ここまで ▲▲▲ ---

def final_interpolate_gaps(df_merged: pd.DataFrame, all_template_ids: list) -> pd.DataFrame:
    """隙間区間と残ったNaNを時間軸で線形補間。"""
    print("最終線形補間を実行...")
    if df_merged.empty: return df_merged
    df_interpolated_list = []
    df_merged = df_merged.sort_values('Time') # Timeでソート
    for tid in all_template_ids:
        marker_df = df_merged[df_merged['id'] == tid].copy()
        if marker_df.empty: continue
        # Timeの重複を除去
        marker_df = marker_df.drop_duplicates(subset='Time').set_index('Time')
        # NaNでない点が2つ以上あるか確認
        valid_points = marker_df[['x','y','z']].dropna()
        if len(valid_points) >= 2:
            try: # interpolate のエラーハンドリング
                 marker_df[['x', 'y', 'z']] = marker_df[['x', 'y', 'z']].interpolate(method='index', limit_direction='both')
            except ValueError as e: print(f"警告: ID {tid} の補間エラー: {e}")
        elif len(valid_points) == 1: # 点が1つだけならそれで埋める
             marker_df[['x', 'y', 'z']] = marker_df[['x', 'y', 'z']].fillna(value=valid_points.iloc[0].to_dict())
        # else: NaNのみならそのまま
        df_interpolated_list.append(marker_df.reset_index())
    if not df_interpolated_list: return pd.DataFrame() # 結果が空の場合
    df_final = pd.concat(df_interpolated_list).sort_values(['Frame', 'id']).reset_index(drop=True)
    if df_final[['x','y','z']].isnull().values.any():
        print("警告: 最終補間後もNaNあり。")
    else: print("最終補間完了。")
    return df_final


# =============================================================================
# メイン実行ブロック
# =============================================================================

if __name__ == "__main__":
    # --- 1. タスク選択 ---
    while True:
        task_key = input("解析するタスク名を入力 (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS: cfg = config.TASK_CONFIGS[task_key]; break
        else: print(f"エラー: config.py に '{task_key}' が見つかりません。")
    print(f"\n--- {task_key} の処理を開始 ---")

    # --- 2. データ読み込み ---
    df_long = load_opti_data(cfg['OPTI_CSV_PATH'])
    df_cycles_indexed = load_gait_cycles(cfg['LABCHART_CYCLES_PATH'], config.TIME_OFFSET)
    df_mean_cycle = load_mean_cycle(cfg['MEAN_CYCLE_CSV_PATH'])
    if df_long is None: exit("エラー: OptiTrack データ読み込み失敗。")

    # --- 3. テンプレート作成 ---
    all_template_ids = [tid for sublist in cfg['SEGMENTS'].values() for tid in sublist]
    template_geometry_static, static_templates, swing_templates = build_hybrid_templates(df_long, df_mean_cycle, cfg)
    if template_geometry_static is None or static_templates is None: exit("エラー: テンプレート作成失敗。")

    # --- 4. キーフレーム取得 ---
    key_frame_index, initial_pose_map = get_keyframe_data(df_long, cfg)
    if initial_pose_map is None: exit("エラー: キーフレーム処理失敗。")

    # --- 5. 歩行区間データ抽出 ---
    walk_start = cfg.get('T1_WALK_START'); walk_end = cfg.get('T2_WALK_END')
    if walk_start is None or walk_end is None: exit("エラー: configに歩行区間未定義。")
    df_long_walk = df_long[(df_long['Time'] >= walk_start) & (df_long['Time'] <= walk_end)].copy()
    if df_long_walk.empty: exit(f"エラー: 歩行区間 ({walk_start}s-{walk_end}s) データが空。")

    # --- 6. 位相別追跡実行 ---
    df_walk_processed = process_phase_switched_tracking( # ★ 関数名を変更
        df_long_walk, df_cycles_indexed, initial_pose_map,
        template_geometry_static, static_templates, swing_templates, cfg
    )
    if df_walk_processed.empty: exit("エラー: 歩行区間の追跡処理結果が空。")
    if df_walk_processed[['x','y','z']].isnull().values.any(): print("警告: 歩行区間の追跡結果にNaNあり。")

    # --- 7. 全区間の結合 ---
    print("全フレームデータの結合と静止/隙間区間補完を開始...")
    df_scaffold = create_full_scaffold(df_long, all_template_ids)
    # fill_static_zones は静止区間を埋め、歩行区間+隙間をNaNにする
    df_filled = fill_static_zones(df_scaffold, template_geometry_static, cfg)

    # --- ▼▼▼【修正箇所】データ結合ロジック変更 ▼▼▼ ---
    # merge を使って歩行区間データを結合 (Time列を保持するため df_filled をベースにする)
    df_merged = pd.merge(
        df_filled, # Frame, Time, id, x_static_or_nan, y_static_or_nan, z_static_or_nan
        df_walk_processed[['Frame', 'id', 'x', 'y', 'z']], # 歩行区間の結果
        on=['Frame', 'id'],
        how='left',
        suffixes=('_filled', '_walk') # suffix を変更
    )
    # 歩行データ(_walk)があれば優先、なければ静止/Gap(_filled)のデータを使う
    df_merged['x'] = df_merged['x_walk'].fillna(df_merged['x_filled'])
    df_merged['y'] = df_merged['y_walk'].fillna(df_merged['y_filled'])
    df_merged['z'] = df_merged['z_walk'].fillna(df_merged['z_filled'])
    # 最終的に必要な列を選択
    df_merged = df_merged[['Frame', 'Time', 'id', 'x', 'y', 'z']]
    # --- ▲▲▲ 修正箇所 ▲▲▲ ---

    # --- 8. 最終補間 ---
    df_final = final_interpolate_gaps(df_merged, all_template_ids)

    # --- 9. 保存 ---
    if not df_final.empty:
        try:
            output_dir = os.path.dirname(cfg['OUTPUT_CSV_PATH'])
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            df_final.to_csv(cfg['OUTPUT_CSV_PATH'], index=False, float_format='%.6f')
            print(f"\n処理結果を保存しました: {cfg['OUTPUT_CSV_PATH']} ({len(df_final)} 行)")
        except Exception as e: print(f"エラー: 結果の保存に失敗。 {e}")
    else: print("\nエラー: 最終結果が空。ファイルは保存されません。")

    print(f"\n--- {task_key} の処理完了 ---")