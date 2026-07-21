# 近傍点探索
# =============================================================================
# opti_edit_D_v2.py (v24 - Persistent ID Map)
#
# 概要:
#   opti_edit_D (v9.1) をベースに、IDの永続マッピング機能を追加。
#
# アプローチ: "キーフレーム" + "生ID/IDマップ優先追跡" + "近傍点探索(マップ登録)" + "セグメント補完"
#
# 処理フロー:
# 1. テンプレート作成 (静止立位)
# 2. キーフレーム特定 (手動マップから initial_pose_map と id_map を作成)
# 3. 歩行区間の追跡 (track_walking_from_keyframe_v2):
#    - (A) テンプレートID優先: 観測点の生IDがテンプレートIDセットにあれば採用。
#    - (B) IDマップ優先: 観測点の生IDが id_map にあれば採用。
#    - (C) 近傍点探索: (A)(B)以外の「ID不明観測点」と
#          「未発見テンプレートID」をマッチング。
#    - (D) IDマップ登録: (C)でマッチしたら、id_map[生ID] = テンプレートID を登録。
#    - (E) セグメント補完: (A)～(D)でも見つからないIDを補完。
#    - (F) 状態更新: 最終ポーズを次フレームの近傍探索用に保存。
# 4. 全区間の結合と補完
# ============================================================================
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# --- ▼▼▼ 【最重要】設定 ▼▼▼ ---
# --- ▼▼▼ 設定 ▼▼▼ ---
OPTITRACK_CSV_PATH = r"C:\FuttoAnalysis\opti\20251027\task3.csv"
OUTPUT_CSV_PATH    = r"C:\FuttoAnalysis\opti\20251027\task3_corrected_D.csv"


# 1. テンプレート作成用の静止立位区間 (秒)
#STATIC_START = 3.0
#STATIC_END   = 4.0
STATIC_START = 5.00,
STATIC_END = 10.00,


# 2. 処理対象の区間定義 (秒)
T1_STATIC_END = 10.0

#task2
#T1_WALK_START = 12.596 # 隙間なく歩行開始
#task3
T1_WALK_START = 13.498 # 隙間なく歩行開始

T2_WALK_END   = 50.0
T2_STATIC_START = 50.0 # 隙間なく静止開始

# 3. テンプレートIDとセグメント定義 (ユーザー指定)
#task2

SEG_HIP =  [8686, 8682, 8680, 8688]
SEG_THIGH = [8674]
SEG_KNEE = [8692, 8698, 8690, 8702]
SEG_SHANK = [8676] # ★ユーザーの指定に基づき 24752 -> 24788 に変更
SEG_FOOT = [8678, 8696, 8700, 8684, 8694]

SEGMENTS = {
    'Hip': SEG_HIP, 'Thigh': SEG_THIGH, 'Knee': SEG_KNEE,
    'Shank': SEG_SHANK, 'Foot': SEG_FOOT
}
TEMPLATE_IDS = SEG_HIP + SEG_THIGH + SEG_KNEE + SEG_SHANK + SEG_FOOT
# ★ 高速検索用にセットを作成
TEMPLATE_ID_SET = set(TEMPLATE_IDS)

PROCESSING_ORDER = ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']
CHAIN_HIERARCHY = {
    'Hip': None, 'Thigh': 'Hip', 'Knee': 'Thigh', 'Shank': 'Knee', 'Foot': 'Shank'
}

# 4. ★★★【手動設定】キーフレーム・マッピング ★★★
# visualize_frame.py を実行し、T1_WALK_START (10.0s) 付近の
#屈曲時のテンプレ
# 「生ID」と「テンプレートID」の対応を記述します。

KEYFRAME_MAP = {
    # --- Hip ---
    8686: 8686,
    8682: 8682,
    8680: 8680,
    8688: 8688,
    # --- Thigh ---
    8674: 8674,
    # --- Knee ---
    8692: 8692,
    8698: 8698,
    8690: 8690,
    8702: 8702,
    # --- Shank ---
    8676: 8676,
    # --- Foot ---
    8678: 8678,
    8696: 8696,
    8700: 8700,
    8684: 8684,
    8694: 8694,

}

'''
# 3. テンプレートIDとセグメント定義 (ユーザー指定)
#task2
SEG_HIP =  [7264, 7260, 7250, 7258]
SEG_THIGH = [7266]
SEG_KNEE = [7256, 7254, 7252, 7248]
SEG_SHANK = [7240] # ★ユーザーの指定に基づき 24752 -> 24788 に変更
SEG_FOOT = [7246, 7262, 7244, 7238, 7242]

SEGMENTS = {
    'Hip': SEG_HIP, 'Thigh': SEG_THIGH, 'Knee': SEG_KNEE,
    'Shank': SEG_SHANK, 'Foot': SEG_FOOT
}
TEMPLATE_IDS = SEG_HIP + SEG_THIGH + SEG_KNEE + SEG_SHANK + SEG_FOOT

PROCESSING_ORDER = ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']
CHAIN_HIERARCHY = {
    'Hip': None, 'Thigh': 'Hip', 'Knee': 'Thigh', 'Shank': 'Knee', 'Foot': 'Shank'
}

# 4. ★★★【手動設定】キーフレーム・マッピング ★★★
# visualize_frame.py を実行し、T1_WALK_START (10.0s) 付近の
#屈曲時のテンプレ
# 「生ID」と「テンプレートID」の対応を記述します。

#task3
KEYFRAME_MAP = {
    # --- ▼▼▼ (例) 実際の生ID -> テンプレートID ▼▼▼ ---
    # --- Hip ---
    7264: 7264,
    7260: 7260,
    7250: 7250,
    7258: 7258,
    # --- Thigh ---
    7266: 7266,
    # --- Knee ---
    7256: 7256,
    7254: 7254,
    7252: 7252,
    7248: 7248,
    # --- Shank ---
    7240: 7240,
    # --- Foot ---
    7270: 7246,
    7262: 7262,
    7244: 7244,
    7238: 7238,
    7242: 7242,
    # --- ▲▲▲ (例) ここまで ▲▲▲ ---
}
'''


# マッピングの基準とする時刻 (T1_WALK_START と一致させるのが推奨)
KEYFRAME_TIME = T1_WALK_START

# 5. ★【修正済み】物理的にありえる座標範囲
PLAUSIBLE_BOUNDS = {'x': (-300, 150), 'y': (0, 1100), 'z': (-1000, 100)}

# 6. その他
MATCHING_THRESHOLD_MM = 75.0 # 近傍追跡の許容範囲
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


# =============================================================================
# ヘルパー関数群
# =============================================================================

def load_opti_data_to_long_robust(file_path):
    """OptiTrack CSV を頑健に読み込み、mm単位に変換する。"""
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}"); return None
    print(f"'{os.path.basename(file_path)}' を読み込み中...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダー
            for line_num, line in enumerate(f, 44):
                parts = line.strip().split(',')
                try:
                    if len(parts) < 5: continue
                    frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4]); base_col = 5
                    if len(parts) >= base_col + n_markers * 4:
                        for i in range(n_markers):
                            x = float(parts[base_col + 4*i]) * 1000.0
                            y = float(parts[base_col + 4*i + 1]) * 1000.0
                            z = float(parts[base_col + 4*i + 2]) * 1000.0
                            mid = int(parts[base_col + 4*i + 3])
                            rows.append((frame, t, mid, x, y, z))
                except (ValueError, IndexError): continue
        if not rows: print("エラー: 有効なデータ行が読み込めませんでした。"); return None
        df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"]); print(f"読み込み成功: {len(df)} 行"); return df
    except Exception as e: print(f"ファイル読み込みエラー: {e}"); return None

def build_segment_templates(df_long, segments, hierarchy, all_template_ids):
    """静止立位区間から、セグメントベースの補完用テンプレートを作成する。"""
    print(f"静止立位区間 ({STATIC_START}s - {STATIC_END}s) からテンプレートを作成...")
    # STATIC_START と STATIC_END がタプルの場合 (例: (5.0,))、最初の要素を取得
    start_time = STATIC_START[0] if isinstance(STATIC_START, tuple) else STATIC_START
    end_time = STATIC_END[0] if isinstance(STATIC_END, tuple) else STATIC_END

    static_df = df_long[(df_long['Time'] >= start_time) & (df_long['Time'] <= end_time)]
    if static_df.empty:
        print(f"エラー: 静止区間 ({start_time}s - {end_time}s) にデータがありません。"); return None, None
    
    mean_pos_df = static_df[static_df['id'].isin(all_template_ids)].groupby('id')[['x','y','z']].mean()
    
    if len(mean_pos_df) < len(all_template_ids):
        print(f"警告: 静止区間で {len(mean_pos_df)}/{len(all_template_ids)} 個のIDしか見つかりません。")
        found_ids = set(mean_pos_df.index)
        missing = set(all_template_ids) - found_ids; print(f"  (不足: {missing})")
        if len(mean_pos_df) < 10: print("エラー: テンプレートマーカーが少なすぎます。"); return None, None
            
    static_mean_pos = {tid: pos.to_numpy() for tid, pos in mean_pos_df.iterrows()}
    for tid in all_template_ids: static_mean_pos.setdefault(tid, np.full(3, np.nan)) # 不足分はNaN

    segment_of_tid = {tid: seg for seg, tids in segments.items() for tid in tids}

    # セグメント重心、相対ベクトル、重心間ベクトルを計算
    static_centroids, marker_rel_vectors, centroid_rel_vectors = {}, {}, {}
    try:
        for seg_name, tids in segments.items():
            valid_pos = [p for tid in tids if (p := static_mean_pos.get(tid)) is not None and not np.isnan(p).any()]
            static_centroids[seg_name] = np.mean(valid_pos, axis=0) if valid_pos else np.zeros(3)
        for tid in all_template_ids:
            pos = static_mean_pos.get(tid); seg_name = segment_of_tid.get(tid)
            if pos is not None and not np.isnan(pos).any() and seg_name and seg_name in static_centroids:
                marker_rel_vectors[tid] = pos - static_centroids[seg_name]
        for seg_name, parent_name in hierarchy.items():
            if parent_name and seg_name in static_centroids and parent_name in static_centroids:
                centroid_rel_vectors[seg_name] = static_centroids[seg_name] - static_centroids[parent_name]
            elif seg_name in static_centroids:
                centroid_rel_vectors[seg_name] = np.zeros(3)
    except Exception as e:
        print(f"セグメントプロパティ計算エラー: {e}"); return None, None

    templates = (static_centroids, marker_rel_vectors, centroid_rel_vectors, segment_of_tid)
    print("テンプレート作成完了。")
    return static_mean_pos, templates

def filter_plausible_markers(frame_df, bounds):
    """物理的にありえる範囲内のマーカーのみを抽出する"""
    if frame_df.empty: return frame_df
    try:
        mask = (frame_df['x'].between(*bounds['x']) &
                frame_df['y'].between(*bounds['y']) &
                frame_df['z'].between(*bounds['z']))
        return frame_df[mask]
    except KeyError:
        print(f"エラー: filter_plausible - 列 x, y, z が見つかりません。"); return pd.DataFrame()


# --- ▼▼▼【関数修正】id_map を返すように変更 ▼▼▼ ---
def get_keyframe_data(df_long, keyframe_time, keyframe_map, template_ids):
    """
    手動マップに基づき、キーフレームの「正しい」位置と
    「初期IDマップ」({raw_id: tid}) を返す。
    """
    print(f"キーフレーム (t~{keyframe_time:.3f}s) のデータを取得しています...")
    
    try:
        key_frame_row = df_long.iloc[(df_long['Time'] - keyframe_time).abs().argmin()]
        key_frame_index = int(key_frame_row['Frame'])
        actual_time = key_frame_row['Time']
    except (IndexError, KeyError):
        print("エラー: キーフレーム時刻付近のデータが見つかりません。"); return None, None, None
    
    key_frame_df = df_long[df_long['Frame'] == key_frame_index]
    print(f"  -> {keyframe_time}s に最も近いフレーム {key_frame_index} (実時間 {actual_time:.3f}s) を使用します。")
    
    # ノイズ除去
    obs_df = filter_plausible_markers(key_frame_df, PLAUSIBLE_BOUNDS)
    if obs_df.empty:
         print("エラー: キーフレームにノイズ除去後のマーカーがありません。"); return None, None, None

    key_frame_pos_map = {int(row.id): (row.x, row.y, row.z) for row in obs_df.itertuples()}

    initial_pose_map = {} # {TID: pos}
    id_map = {}           # ★ {RawID: TID}
    found_count = 0
    missing_in_map = []
    missing_in_frame = []
    
    for template_id in template_ids:
        raw_id_found = next((r_id for r_id, t_id in keyframe_map.items() if t_id == template_id), None)
        
        if raw_id_found:
            if raw_id_found in key_frame_pos_map:
                initial_pose_map[template_id] = np.array(key_frame_pos_map[raw_id_found])
                # ★ IDマップも作成
                if raw_id_found != template_id:
                     id_map[raw_id_found] = template_id
                found_count += 1
            else:
                missing_in_frame.append(f"{raw_id_found} (->{template_id})")
        else:
            missing_in_map.append(f"{template_id}")

    if missing_in_map: print(f"  警告: KEYFRAME_MAP に未定義のTID: {missing_in_map}")
    if missing_in_frame: print(f"  警告: KEYFRAME_MAP の生IDが実データに不在: {missing_in_frame}")
    if found_count < len(template_ids): print(f"  {found_count}/{len(template_ids)} 個のマーカーで追跡開始。")
    else: print(f"  {found_count}/{len(template_ids)} 個全てのマーカーで追跡開始。")
    if found_count == 0: print("エラー: キーフレームでマーカーゼロ。"); return None, None, None
    
    print(f"  初期IDマップ: {id_map}")
    return key_frame_index, initial_pose_map, id_map
# --- ▲▲▲ 関数修正ここまで ▲▲▲ ---

def predict_missing_marker_pos(tid, current_centroids, templates, hierarchy, segments):
    """セグメント連鎖に基づき、単一の欠損マーカー(tid)の位置を予測する。"""
    (static_centroids, marker_rel_vectors, 
     centroid_rel_vectors, segment_of_tid) = templates
        
    if tid not in segment_of_tid or tid not in marker_rel_vectors:
        return np.full(3, np.nan) # テンプレートにない

    seg_name = segment_of_tid[tid]
    
    # 予測に必要な親セグメントが計算されているか確認
    curr_seg = seg_name
    while curr_seg is not None:
        if curr_seg not in current_centroids:
            return np.full(3, np.nan) # 親(または自身)が未計算
        curr_seg = hierarchy.get(curr_seg) # .get() で安全に
    
    # 補完
    centroid = current_centroids.get(seg_name)
    rel_vector = marker_rel_vectors.get(tid)
    
    if centroid is not None and rel_vector is not None and not np.isnan(centroid).any():
        return centroid + rel_vector
    else:
        return np.full(3, np.nan)


# --- ▼▼▼【関数修正】永続IDマップ(id_map)ロジック導入 ▼▼▼ ---
def track_walking_with_id_map(df_long_walk, keyframe_frame, initial_pose_map,
                              templates, all_template_ids, initial_id_map):
    """
    キーフレームから開始し、「生ID/IDマップ優先」 + 「近傍点探索(マップ登録)」 + 「セグメント補完」
    で歩行区間を処理する。
    """
    print("歩行区間の処理を開始 (永続IDマップ + 近傍点探索)...")
    
    (static_centroids, marker_rel_vectors, 
     centroid_rel_vectors, segment_of_tid) = templates
        
    # --- 状態変数初期化 ---
    last_known_pose_map = initial_pose_map.copy() 
    current_centroids = static_centroids.copy()
    id_map = initial_id_map.copy() # ★ 永続IDマップ

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
        if frame_idx % 500 == 0:
            print(f"  Processing frame {frame_idx}/{len(unique_frames)}...")

        frame_group = df_long_walk[df_long_walk['Frame'] == frame]
        if frame_group.empty: continue
        time_scalar = frame_group['Time'].iloc[0]
        obs_df = filter_plausible_markers(frame_group, PLAUSIBLE_BOUNDS)
        
        # このフレームの最終座標 {TID: pos}
        current_pose_map = {}
        # (A) (B) で使われなかった観測点 (raw_id, pos)
        unassigned_obs_rows = []
        # (A) (B) ですでに割り当てられたTID
        assigned_tids = set()

        # 1. テンプレートID / IDマップ優先マッチング
        if not obs_df.empty:
            for row in obs_df.itertuples():
                raw_id = int(row.id)
                pos = np.array([row.x, row.y, row.z])
                
                if raw_id in TEMPLATE_ID_SET:
                    # (A) 生IDがテンプレートID
                    current_pose_map[raw_id] = pos
                    assigned_tids.add(raw_id)
                elif raw_id in id_map:
                    # (B) 生IDが既知のマップにある
                    tid = id_map[raw_id]
                    if tid not in assigned_tids: # 複数の生IDが同じTIDにマップされるのを防止
                         current_pose_map[tid] = pos
                         assigned_tids.add(tid)
                else:
                    # (C) ID不明の観測点
                    unassigned_obs_rows.append((raw_id, pos))

        # 2. 近傍点探索 (ID不明マーカーの割り当て)
        # (A)(B)で見つからなかったテンプレートID
        missing_template_ids = [tid for tid in all_template_ids if tid not in assigned_tids]
        
        if missing_template_ids and unassigned_obs_rows:
            # 見つかっていないTIDの「前フレームの位置」
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
                
                if last_pose_array.ndim == 2 and last_pose_array.shape[0] > 0 and \
                   unknown_obs_coords.ndim == 2 and unknown_obs_coords.shape[0] > 0:
                    try:
                        cost_matrix = cdist(last_pose_array, unknown_obs_coords)
                        last_indices, obs_indices = linear_sum_assignment(cost_matrix)
                        used_obs_indices = set()
                        for last_idx, obs_idx in zip(last_indices, obs_indices):
                            if obs_idx not in used_obs_indices and \
                               obs_idx < unknown_obs_coords.shape[0] and \
                               cost_matrix[last_idx, obs_idx] < MATCHING_THRESHOLD_MM:
                                
                                tid = last_pose_tids_missing[last_idx]
                                raw_id = unknown_obs_ids[obs_idx]
                                pos = unknown_obs_coords[obs_idx]
                                
                                current_pose_map[tid] = pos # 最終ポーズに採用
                                id_map[raw_id] = tid      # ★ (D) IDマップに登録
                                assigned_tids.add(tid)    # このTIDは割り当て済み
                                used_obs_indices.add(obs_idx)
                                
                    except ValueError as e:
                        print(f"警告: 近傍点探索マッチングエラー @F{frame}: {e}")

        # 3. セグメント重心の「現在位置」を計算
        temp_centroids = {}
        for seg_name in PROCESSING_ORDER:
            # (A)(B)(C) で確定したマーカーを使って重心を計算
            found_markers_pos = [
                current_pose_map[tid] 
                for tid in SEGMENTS[seg_name]
                if tid in current_pose_map # current_pose_map には観測点のみ
            ]
            if found_markers_pos:
                temp_centroids[seg_name] = np.mean(found_markers_pos, axis=0)
            else: # 親から予測
                parent_name = CHAIN_HIERARCHY[seg_name]
                if parent_name and parent_name in temp_centroids:
                    parent_centroid = temp_centroids[parent_name]
                    rel_vec = centroid_rel_vectors.get(seg_name, np.zeros(3))
                    if parent_centroid is not None and not np.isnan(parent_centroid).any():
                        temp_centroids[seg_name] = parent_centroid + rel_vec
                    else:
                         temp_centroids[seg_name] = current_centroids.get(seg_name, np.zeros(3))
                else:
                    temp_centroids[seg_name] = current_centroids.get(seg_name, static_centroids.get(seg_name, np.zeros(3)))
        current_centroids = temp_centroids

        # 4. 欠損補完 と 格納
        for tid in all_template_ids:
            if tid in current_pose_map:
                # (A)(B)(C) で見つかった
                pos = current_pose_map[tid]
            else:
                # (E) セグメントロジックで補完
                pos = predict_missing_marker_pos(tid, current_centroids, templates, CHAIN_HIERARCHY, SEGMENTS)
                current_pose_map[tid] = pos # 補完位置もマップに追加
                
            corrected_rows.append((frame, time_scalar, int(tid), *pos))

        # 5. 状態更新
        last_known_pose_map = current_pose_map # 次フレームの近傍探索用
    
    print(f"歩行区間の追跡が完了しました。")
    print(f"  最終的なIDマップ ({len(id_map)}件): {id_map}") # ★ 最終マップを出力
    if not corrected_rows: return pd.DataFrame()
    return pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])
# --- ▲▲▲ 関数修正ここまで ▲▲▲ ---


def create_full_scaffold(df_long, template_ids):
    """全フレーム x 全テンプレートID の「抜け殻」DataFrameを作成する"""
    if df_long.empty: return pd.DataFrame()
    all_frames_df = df_long[['Frame', 'Time']].drop_duplicates().sort_values('Frame').reset_index(drop=True)
    all_ids_df = pd.DataFrame({'id': template_ids})
    df_scaffold = pd.merge(all_frames_df, all_ids_df, how='cross')
    return df_scaffold.sort_values(['Frame', 'id']).reset_index(drop=True)

def fill_static_zones(df_full_scaffold, static_mean_pos):
    """静止区間を、静止時平均座標で埋める"""
    print(f"静止区間 (t<{T1_STATIC_END}s, t>{T2_STATIC_START}s) をテンプレート座標で補完...")
    template_df = pd.DataFrame.from_dict(static_mean_pos, orient='index', columns=['x_template', 'y_template', 'z_template'])
    template_df['id'] = template_df.index.astype(int)
    template_df = template_df.dropna() # NaNテンプレートを除外

    df_full_scaffold['id'] = df_full_scaffold['id'].astype(int)
    df_filled = pd.merge(df_full_scaffold, template_df, on='id', how='left', suffixes=('', '_tpl'))
    
    # 歩行区間(T1_WALK_START <= t <= T2_WALK_END)の座標をNaNに戻す
    is_static = (df_filled['Time'] <= T1_STATIC_END) | (df_filled['Time'] >= T2_STATIC_START)
    df_filled['x'] = np.where(is_static, df_filled['x_template'], np.nan)
    df_filled['y'] = np.where(is_static, df_filled['y_template'], np.nan)
    df_filled['z'] = np.where(is_static, df_filled['z_template'], np.nan)

    df_filled = df_filled[['Frame', 'Time', 'id', 'x', 'y', 'z']]
    return df_filled

def final_interpolate_gaps(df_merged):
    """隙間区間を線形補間する"""
    print("隙間区間を線形補間します...")
    if df_merged.empty: return pd.DataFrame()
    df_interpolated_list = []
    df_merged = df_merged.sort_values('Time')
    all_ids = df_merged['id'].unique()
    
    for tid in all_ids:
        marker_df = df_merged[df_merged['id'] == tid].copy()
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
    
    if df_final.isnull().values.any(): print("警告: 最終補間後もNaNが残っています。")
    else: print("最終補間が完了しました。")
    
    return df_final


if __name__ == "__main__":
    df_long = load_opti_data_to_long_robust(OPTITRACK_CSV_PATH)
    
    if df_long is not None and not df_long.empty:
        
        # 1. 静止立位から「お手本」を作成
        static_mean_pos, templates = build_segment_templates(
            df_long, SEGMENTS, CHAIN_HIERARCHY, TEMPLATE_IDS
        )
        
        if static_mean_pos and templates:
            # 2. 歩行区間のデータのみを抽出
            df_long_walk = df_long[
                (df_long['Time'] >= T1_WALK_START) & (df_long['Time'] <= T2_WALK_END)
            ].copy()
            if df_long_walk.empty:
                print(f"エラー: 歩行区間 ({T1_WALK_START}s - {T2_WALK_END}s) のデータが0行です。")
                exit()

            # 3. キーフレーム(10.0s)の「正解データ」と「初期IDマップ」を取得
            key_frame_index, initial_pose_map, initial_id_map = get_keyframe_data(
                df_long, KEYFRAME_TIME, KEYFRAME_MAP, TEMPLATE_IDS
            )
            
            if initial_pose_map:
                # 4. 歩行区間を「キーフレーム」から追跡 (★ 修正後の関数を呼ぶ)
                df_walk_processed = track_walking_with_id_map(
                    df_long_walk, key_frame_index, initial_pose_map, 
                    templates, TEMPLATE_IDS, initial_id_map
                )

                # 5. 全フレーム x 全ID の「抜け殻」を作成
                df_full_scaffold = create_full_scaffold(df_long, TEMPLATE_IDS)

                # 6. 抜け殻に「静止区間」のデータを埋める
                df_with_static = fill_static_zones(df_full_scaffold, static_mean_pos)

                # 7. (6)に(4)の「歩行区間」データを上書き
                # ★ KeyError: 'Time' 修正ロジック (v20.1)
                df_merged = pd.merge(
                    df_with_static,
                    df_walk_processed[['Frame', 'id', 'x', 'y', 'z']],
                    on=['Frame', 'id'],
                    how='left',
                    suffixes=('_filled', '_walk')
                )
                df_merged['x'] = df_merged['x_walk'].fillna(df_merged['x_filled'])
                df_merged['y'] = df_merged['y_walk'].fillna(df_merged['y_filled'])
                df_merged['z'] = df_merged['z_walk'].fillna(df_merged['z_filled'])
                df_merged = df_merged[['Frame', 'Time', 'id', 'x', 'y', 'z']]

                # 8. 隙間区間を線形補間
                df_final = final_interpolate_gaps(df_merged)

                if df_final.empty:
                    print("エラー: 最終処理結果が空になりました。")
                else:
                    print(f"処理完了。補正後のデータ: {len(df_final)} 行")
                    out_dir = os.path.dirname(OUTPUT_CSV_PATH)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    
                    # 9. 保存
                    df_final.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
                    print(f"補正済みデータを保存しました: {OUTPUT_CSV_PATH}")
            else:
                print("エラー: キーフレームを処理できませんでした。KEYFRAME_MAP を確認してください。")
        else:
            print("エラー: テンプレートを作成できなかったため、処理を中断します。")
    else:
        print("データが読み込めなかったため、処理を終了します。")