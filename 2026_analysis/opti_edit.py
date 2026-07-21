# 近傍点探索 & 点群レジストレーション (Resync)
# =============================================================================
# opti_edit.py (v27 - Anatomical Resync for task03)
#
# 概要:
#   v26 (Smart Resync Engine) をベースに、task03(関節直接配置・5点)向けの
#   解剖学的座標ルールによる Resync モードを追加。
#
# Resync モード (CONFIG の RESYNC_MODE で切り替え):
#   'y_slice'    : Y座標スライス + ローカル形状マッチング  ← task01/02
#   'anatomical' : 解剖学的座標ルール                      ← task03
#       Y降順: Hip > Knee > Ankle > (Heel or Toe)
#       Z降順: Heel > Toe  (踵の方が進行方向後ろ = Z大)
#
# アプローチ:
#   1. 生ID/IDマップ優先追跡 & 近傍点探索 (通常時)
#   2. Resync発動時: モードに応じた全体再同期
#   3. 前フレーム最終位置フォールバック (静止テンプレート張り付き防止)
#
# 0-40s:offset / 40-340s:walk / 340-360s:offset
# ============================================================================
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# --- CONFIG.py のパス解決 ---
# このスクリプトは C:\FuttoAnalysis\2026_analysis\ に置かれている想定。
# CONFIG.py は C:\FuttoAnalysis\2026_analysis\futto_common\ に存在する。
_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futto_common')
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import CONFIG # 設定モジュールのインポート

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

def build_segment_templates(df_long, segments, hierarchy, all_template_ids, static_start, static_end):
    """静止立位区間から、セグメントベースの補完用テンプレートを作成する。"""
    print(f"静止立位区間 ({static_start}s - {static_end}s) からテンプレートを作成...")
    start_time = static_start[0] if isinstance(static_start, tuple) else static_start
    end_time = static_end[0] if isinstance(static_end, tuple) else static_end

    static_df = df_long[(df_long['Time'] >= start_time) & (df_long['Time'] <= end_time)]
    if static_df.empty:
        print(f"エラー: 静止区間 ({start_time}s - {end_time}s) にデータがありません。"); return None, None
    
    mean_pos_df = static_df[static_df['id'].isin(all_template_ids)].groupby('id')[['x','y','z']].mean()
    
    if len(mean_pos_df) < len(all_template_ids):
        print(f"警告: 静止区間で {len(mean_pos_df)}/{len(all_template_ids)} 個のIDしか見つかりません。")
        found_ids = set(mean_pos_df.index)
        missing = set(all_template_ids) - found_ids; print(f"  (不足: {missing})")
        # マーカーが少ないタスク(task03等)では全数必須、多いタスクでは半数以上
        min_required = len(all_template_ids) if len(all_template_ids) <= 5 else max(len(all_template_ids) // 2, 4)
        if len(mean_pos_df) < min_required:
             print("エラー: テンプレートマーカーが少なすぎます。"); return None, None
            
    static_mean_pos = {tid: pos.to_numpy() for tid, pos in mean_pos_df.iterrows()}
    for tid in all_template_ids: static_mean_pos.setdefault(tid, np.full(3, np.nan))

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
    """物理的にありえる範囲内のマーカーのみを抽出する (外れ値ノイズの除去)"""
    if frame_df.empty: return frame_df
    try:
        mask = (frame_df['x'].between(*bounds['x']) &
                frame_df['y'].between(*bounds['y']) &
                frame_df['z'].between(*bounds['z']))
        return frame_df[mask]
    except KeyError:
        return pd.DataFrame()

def get_keyframe_data(df_long, keyframe_time, keyframe_map, template_ids, bounds):
    """キーフレームの初期データを取得する"""
    print(f"キーフレーム (t~{keyframe_time:.3f}s) のデータを取得しています...")
    try:
        key_frame_row = df_long.iloc[(df_long['Time'] - keyframe_time).abs().argmin()]
        key_frame_index = int(key_frame_row['Frame'])
        actual_time = key_frame_row['Time']
    except (IndexError, KeyError):
        print("エラー: キーフレーム時刻付近のデータが見つかりません。"); return None, None, None
    
    key_frame_df = df_long[df_long['Frame'] == key_frame_index]
    obs_df = filter_plausible_markers(key_frame_df, bounds)
    if obs_df.empty: return None, None, None

    key_frame_pos_map = {int(row.id): (row.x, row.y, row.z) for row in obs_df.itertuples()}
    initial_pose_map, id_map = {}, {}
    found_count = 0
    
    for template_id in template_ids:
        raw_id_found = next((r_id for r_id, t_id in keyframe_map.items() if t_id == template_id), None)
        if raw_id_found and raw_id_found in key_frame_pos_map:
            initial_pose_map[template_id] = np.array(key_frame_pos_map[raw_id_found])
            if raw_id_found != template_id: id_map[raw_id_found] = template_id
            found_count += 1

    if found_count == 0: return None, None, None
    return key_frame_index, initial_pose_map, id_map

def predict_missing_marker_pos(tid, current_centroids, templates, hierarchy, segments):
    """セグメント連鎖に基づき、単一の欠損マーカーの位置を予測する"""
    (static_centroids, marker_rel_vectors, centroid_rel_vectors, segment_of_tid) = templates
    if tid not in segment_of_tid or tid not in marker_rel_vectors: return np.full(3, np.nan)

    seg_name = segment_of_tid[tid]
    curr_seg = seg_name
    while curr_seg is not None:
        if curr_seg not in current_centroids: return np.full(3, np.nan)
        curr_seg = hierarchy.get(curr_seg)
    
    centroid = current_centroids.get(seg_name)
    rel_vector = marker_rel_vectors.get(tid)
    if centroid is not None and rel_vector is not None and not np.isnan(centroid).any():
        return centroid + rel_vector
    return np.full(3, np.nan)


def resync_anatomical(obs_df, all_template_ids, anatomical_rules):
    """
    解剖学的座標ルールによる Resync (task03専用)。

    判別手順:
      1. Y降順でソートし、上位3点を Hip / Knee / Ankle に確定。
      2. 残り2点(Heel/Toe候補)を Z降順でソートし、
         Z大 → Heel(67634)、Z小 → Toe(67630) に確定。

    Parameters
    ----------
    obs_df          : ノイズ除去済みの観測フレーム DataFrame
    all_template_ids: 全テンプレートIDのリスト (順序は問わない)
    anatomical_rules: CONFIG から渡す判別ルール辞書
        {
          'y_order' : [tid_1st, tid_2nd, tid_3rd],  # Y降順で上位3点
          'foot_z'  : {'z_high': tid_toe, 'z_low': tid_heel}  # Z大=Toe, Z小=Heel
        }

    Returns
    -------
    (pose_map, id_map, ok)
      pose_map : {template_id: np.array([x,y,z])}
      id_map   : {raw_id: template_id}
      ok       : bool  マッチング成功かどうか
    """
    if len(obs_df) != len(all_template_ids):
        return {}, {}, False

    obs_sorted_y = obs_df.sort_values('y', ascending=False).reset_index(drop=True)
    obs_coords   = obs_sorted_y[['x', 'y', 'z']].values
    obs_raw_ids  = obs_sorted_y['id'].values

    y_order   = anatomical_rules['y_order']    # [Hip, Knee, Ankle]
    foot_z    = anatomical_rules['foot_z']     # {'z_high': Heel_tid, 'z_low': Toe_tid}

    n_top = len(y_order)  # 通常3
    n_foot = len(all_template_ids) - n_top  # 通常2

    if n_top + n_foot != len(all_template_ids):
        return {}, {}, False

    pose_map = {}
    id_map   = {}

    # --- 上位 n_top 点: Y降順で順番に確定 ---
    for i, tid in enumerate(y_order):
        pose_map[tid] = obs_coords[i]
        id_map[int(obs_raw_ids[i])] = tid

    # --- 残り n_foot 点: Z降順で Heel/Toe を判別 ---
    foot_obs_coords  = obs_coords[n_top:]
    foot_obs_raw_ids = obs_raw_ids[n_top:]

    if n_foot == 2:
        z_sort_idx = np.argsort(foot_obs_coords[:, 2])[::-1]  # Z降順
        pose_map[foot_z['z_high']] = foot_obs_coords[z_sort_idx[0]]
        pose_map[foot_z['z_low']]  = foot_obs_coords[z_sort_idx[1]]
        id_map[int(foot_obs_raw_ids[z_sort_idx[0]])] = foot_z['z_high']
        id_map[int(foot_obs_raw_ids[z_sort_idx[1]])] = foot_z['z_low']
    else:
        # 2点以外の足部マーカーは ICP で割り当て (将来の拡張用)
        return {}, {}, False

    ok = (len(pose_map) == len(all_template_ids))
    return pose_map, id_map, ok


# =============================================================================
# スマート・トラッキング・エンジン (Resync機能搭載)
# =============================================================================
def track_walking_with_id_map(df_long_walk, keyframe_frame, initial_pose_map,
                              templates, all_template_ids, initial_id_map,
                              bounds, template_id_set, matching_threshold,
                              processing_order, segments, hierarchy,
                              resync_mode='y_slice', anatomical_rules=None):
    """
    resync_mode:
      'y_slice'    : Y座標スライス + ローカル形状マッチング (task01/02)
      'anatomical' : 解剖学的座標ルールによる判別            (task03)
    """
    print(f"スマート・トラッキングを開始 (Resyncモード: {resync_mode})...")
    
    (static_centroids, marker_rel_vectors, centroid_rel_vectors, segment_of_tid) = templates
        
    last_known_pose_map = initial_pose_map.copy() 
    current_centroids = static_centroids.copy()
    id_map = initial_id_map.copy()
    corrected_rows = []
    
    # --- Resync用にテンプレートのローカル形状(重心からの相対座標)を事前計算 ---
    tpl_local_info = {}
    static_mean_pos = {}
    for tid in all_template_ids:
        # static_centroidsを計算した時と同じロジックで元の平均位置を復元
        seg_name = segment_of_tid[tid]
        static_mean_pos[tid] = static_centroids[seg_name] + marker_rel_vectors[tid]

    for seg_name in processing_order:
        tids = segments[seg_name]
        if not tids: continue
        coords = np.array([static_mean_pos[tid] for tid in tids])
        centroid = np.mean(coords, axis=0)
        local_coords = coords - centroid
        tpl_local_info[seg_name] = {'tids': tids, 'centroid': centroid, 'local': local_coords}

    unique_frames = sorted(df_long_walk['Frame'].unique())
    start_index = unique_frames.index(keyframe_frame) if keyframe_frame in unique_frames else 0

    resync_count = 0

    for frame_idx, frame in enumerate(unique_frames[start_index:], start=start_index):
        if frame_idx % 500 == 0:
            print(f"  Processing frame {frame_idx}/{len(unique_frames)}...")

        frame_group = df_long_walk[df_long_walk['Frame'] == frame]
        if frame_group.empty: continue
        time_scalar = frame_group['Time'].iloc[0]
        
        # 1. ノイズ除去 (物理的範囲外のマーカーを消去)
        obs_df = filter_plausible_markers(frame_group, bounds)
        
        current_pose_map = {}
        resync_successful = False

        # =====================================================================
        # ★ スマート Resync (再同期) ロジック
        # ノイズ除去後のマーカー数が期待数とピッタリ一致したフレームで発動。
        # モードに応じて異なる判別ロジックを使用。
        # =====================================================================
        if len(obs_df) == len(all_template_ids) and len(obs_df) > 0:

            if resync_mode == 'anatomical' and anatomical_rules is not None:
                # -----------------------------------------------------------------
                # 【anatomical モード】task03専用: 解剖学的座標ルールで判別
                # -----------------------------------------------------------------
                temp_pose_map, temp_id_map, match_ok = resync_anatomical(
                    obs_df, all_template_ids, anatomical_rules
                )

            else:
                # -----------------------------------------------------------------
                # 【y_slice モード】task01/02: Y座標スライス + ローカル形状マッチング
                # -----------------------------------------------------------------
                obs_sorted = obs_df.sort_values('y', ascending=False)
                obs_raw_ids = obs_sorted['id'].values
                obs_coords  = obs_sorted[['x', 'y', 'z']].values

                temp_pose_map = {}
                temp_id_map   = {}
                idx      = 0
                match_ok = True

                for seg_name in processing_order:
                    if seg_name not in tpl_local_info: continue
                    info  = tpl_local_info[seg_name]
                    tids  = info['tids']
                    count = len(tids)

                    if idx + count > len(obs_coords):
                        match_ok = False; break

                    seg_obs_ids    = obs_raw_ids[idx : idx+count]
                    seg_obs_coords = obs_coords[idx  : idx+count]
                    idx += count

                    if count == 1:
                        # 1点セグメント: 形状マッチング不要
                        temp_pose_map[tids[0]]        = seg_obs_coords[0]
                        temp_id_map[int(seg_obs_ids[0])] = tids[0]
                    else:
                        obs_centroid = np.mean(seg_obs_coords, axis=0)
                        obs_local    = seg_obs_coords - obs_centroid
                        cost_matrix  = cdist(info['local'], obs_local)
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)

                        for r, c in zip(row_ind, col_ind):
                            if cost_matrix[r, c] > matching_threshold * 1.5:
                                match_ok = False; break
                            temp_pose_map[tids[r]]           = seg_obs_coords[c]
                            temp_id_map[int(seg_obs_ids[c])] = tids[r]

                        if not match_ok: break

            # --- Resync 成功時: id_map を刷新して状態を更新 ---
            if match_ok and len(temp_pose_map) == len(all_template_ids):
                current_pose_map = temp_pose_map
                id_map.clear()
                id_map.update(temp_id_map)
                resync_successful = True
                resync_count += 1

                temp_centroids = {}
                for seg_name in processing_order:
                    if segments[seg_name]:
                        pts = [current_pose_map[t] for t in segments[seg_name]]
                        temp_centroids[seg_name] = np.mean(pts, axis=0)
                current_centroids = temp_centroids

        # =====================================================================
        # 2. 通常トラッキング (Resyncが発動しなかった、または不一致フレーム)
        #
        # 優先順位:
        #   (A) id_mapに登録済みの raw_id  → 閾値なし・無条件採用
        #   (B) template_id_set内の raw_id → そのまま採用
        #   (C) 未登録 raw_id              → 前フレームの位置で近傍探索し登録
        #                                    ※登録後は (A) として扱われる
        #   (D) 観測データなし(本当に映っていない) → 前フレーム位置で補完
        #
        # ★ 原則: 観測データが存在する限り補完より優先。
        #         一度登録した id_map は閾値で弾かない。
        # =====================================================================
        if not resync_successful:
            assigned_tids     = set()
            unassigned_obs    = []   # (raw_id, pos) まだTIDが決まっていない観測点

            # --- (A)(B) id_map / template_id_set で確定できる観測点を全て採用 ---
            if not obs_df.empty:
                for row in obs_df.itertuples():
                    raw_id = int(row.id)
                    pos    = np.array([row.x, row.y, row.z])

                    if raw_id in template_id_set:
                        # raw_id がそのままテンプレートID
                        current_pose_map[raw_id] = pos
                        assigned_tids.add(raw_id)

                    elif raw_id in id_map:
                        # ★ 登録済みIDは閾値なし・無条件で採用
                        tid = id_map[raw_id]
                        if tid not in assigned_tids:
                            current_pose_map[tid] = pos
                            assigned_tids.add(tid)
                        # 同TIDへの2重割り当て(稀)は先着優先 → elseは何もしない

                    else:
                        # まだ id_map に登録されていない未知の raw_id
                        unassigned_obs.append((raw_id, pos))

            # --- (C) 未知 raw_id を前フレーム位置で近傍探索して登録 ---
            missing_tids = [tid for tid in all_template_ids if tid not in assigned_tids]

            if missing_tids and unassigned_obs:
                # 前フレームの既知位置があるTIDのみ候補にする
                ref_tids  = []
                ref_coords = []
                for tid in missing_tids:
                    prev = last_known_pose_map.get(tid)
                    if prev is not None and not np.isnan(prev).any():
                        ref_tids.append(tid)
                        ref_coords.append(prev)

                if ref_tids:
                    ref_arr = np.array(ref_coords)
                    obs_raw_ids_arr, obs_coords_arr = zip(*unassigned_obs)
                    obs_arr = np.array(obs_coords_arr)

                    cost = cdist(ref_arr, obs_arr)
                    row_ind, col_ind = linear_sum_assignment(cost)

                    used_obs = set()
                    for ri, ci in zip(row_ind, col_ind):
                        if ci in used_obs:
                            continue
                        tid    = ref_tids[ri]
                        raw_id = int(obs_raw_ids_arr[ci])
                        pos    = obs_arr[ci]

                        # ★ matching_threshold は「id_map 登録の判定」にのみ使用。
                        #   閾値内なら登録して以後永続的に追跡。
                        #   閾値外でも、他に候補がなければ観測データとして採用する。
                        current_pose_map[tid] = pos
                        assigned_tids.add(tid)
                        used_obs.add(ci)

                        if cost[ri, ci] < matching_threshold:
                            # 十分近い → id_map に永続登録
                            if raw_id not in id_map:  # 新規登録のみ表示
                                print(f"    [ID登録] raw={raw_id} -> tid={tid}  dist={cost[ri,ci]:.1f}mm  @t={time_scalar:.2f}s")
                            id_map[raw_id] = tid

            # --- (D) それでも未取得のTIDは前フレーム位置で補完 ---
            #   ここに来るのは「本当に観測されていない」場合のみ
            temp_centroids = {}
            for seg_name in processing_order:
                pts = [current_pose_map[tid] for tid in segments[seg_name]
                       if tid in current_pose_map]
                if pts:
                    temp_centroids[seg_name] = np.mean(pts, axis=0)
                elif not segments[seg_name]:
                    parent_name = hierarchy.get(seg_name)
                    if parent_name and parent_name in temp_centroids:
                        rel_vec = centroid_rel_vectors.get(seg_name, np.zeros(3))
                        temp_centroids[seg_name] = temp_centroids[parent_name] + rel_vec
                    else:
                        temp_centroids[seg_name] = current_centroids.get(seg_name, np.zeros(3))
                else:
                    parent_name = hierarchy.get(seg_name)
                    if parent_name and parent_name in temp_centroids:
                        rel_vec = centroid_rel_vectors.get(seg_name, np.zeros(3))
                        temp_centroids[seg_name] = temp_centroids[parent_name] + rel_vec
                    else:
                        temp_centroids[seg_name] = current_centroids.get(seg_name, np.zeros(3))
            current_centroids = temp_centroids

            for tid in all_template_ids:
                if tid not in current_pose_map:
                    # 前フレーム位置を優先。なければセグメント連鎖予測
                    prev_pos = last_known_pose_map.get(tid)
                    if prev_pos is not None and not np.isnan(prev_pos).any():
                        pos = prev_pos
                    else:
                        pos = predict_missing_marker_pos(
                            tid, current_centroids, templates, hierarchy, segments
                        )
                    current_pose_map[tid] = pos

        # --- 結果の格納と状態更新 ---
        for tid in all_template_ids:
            pos = current_pose_map[tid]
            corrected_rows.append((frame, time_scalar, int(tid), *pos))
            
        last_known_pose_map = current_pose_map
    
    print(f"歩行区間の追跡完了 (全 {resync_count} 回の再同期が発動しました)")
    if not corrected_rows: return pd.DataFrame()
    return pd.DataFrame(corrected_rows, columns=["Frame", "Time", "id", "x", "y", "z"])

# =============================================================================
# ポスト処理関数
# =============================================================================
def create_full_scaffold(df_long, template_ids):
    """全フレーム x 全テンプレートID の「抜け殻」DataFrameを作成する"""
    if df_long.empty: return pd.DataFrame()
    all_frames_df = df_long[['Frame', 'Time']].drop_duplicates().sort_values('Frame').reset_index(drop=True)
    all_ids_df = pd.DataFrame({'id': template_ids})
    df_scaffold = pd.merge(all_frames_df, all_ids_df, how='cross')
    return df_scaffold.sort_values(['Frame', 'id']).reset_index(drop=True)

def fill_static_zones(df_full_scaffold, static_mean_pos, t1_static_end, t2_static_start):
    """静止区間を、静止時平均座標で埋める"""
    print(f"静止区間 (t<{t1_static_end}s, t>{t2_static_start}s) をテンプレート座標で補完...")
    template_df = pd.DataFrame.from_dict(static_mean_pos, orient='index', columns=['x_template', 'y_template', 'z_template'])
    template_df['id'] = template_df.index.astype(int)
    template_df = template_df.dropna()

    df_full_scaffold['id'] = df_full_scaffold['id'].astype(int)
    df_filled = pd.merge(df_full_scaffold, template_df, on='id', how='left', suffixes=('', '_tpl'))
    
    is_static = (df_filled['Time'] <= t1_static_end) | (df_filled['Time'] >= t2_static_start)
    df_filled['x'] = np.where(is_static, df_filled['x_template'], np.nan)
    df_filled['y'] = np.where(is_static, df_filled['y_template'], np.nan)
    df_filled['z'] = np.where(is_static, df_filled['z_template'], np.nan)

    return df_filled[['Frame', 'Time', 'id', 'x', 'y', 'z']]

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
    print("最終補間が完了しました。")
    return df_final


# =============================================================================
# task03 専用処理
# =============================================================================
def process_task03(cfg):
    """
    task03 (マーカー5点・Futto非着用) 専用の処理フロー。
    
    方針:
      - 補完なし・前フレーム近傍探索なし
      - 5点揃っているフレーム → anatomical判別でTIDを確定し id_map に蓄積
      - 5点未満のフレーム    → id_map に登録済みのraw_idだけTIDに変換して採用
      - 観測できたデータのみ出力（NaN行・補完行は一切含まない）

    anatomical判別ルール (CONFIGのSEGMENTSから自動生成):
      Y降順上位3点 → [Hip, Knee, Ankle] に順番に確定
      残り2点をZ降順 → Z大=Heel(foot_ids[0]), Z小=Toe(foot_ids[1])
    """
    print("\n--- task03 専用処理を開始します ---")

    opti_csv_path  = cfg['OPTI_CSV_PATH']
    output_csv_path = cfg['OUTPUT_CSV_PATH']
    plausible_bounds = cfg.get('PLAUSIBLE_BOUNDS',
                                {'x': (-1000, 1000), 'y': (0, 2000), 'z': (-2000, 2000)})

    # テンプレートIDリストを構築
    segments = cfg['SEGMENTS']
    template_ids = [tid for tids in segments.values() for tid in tids]
    n_expected = len(template_ids)   # 5

    # anatomical判別ルールをCONFIGのSEGMENTSから生成
    # y_order: Foot以外のセグメントIDをPROCESSING_ORDER順に並べる
    y_order = []
    for seg_name in CONFIG.PROCESSING_ORDER:
        if seg_name == 'Foot':
            continue
        ids_in_seg = segments.get(seg_name, [])
        if ids_in_seg:
            y_order.append(ids_in_seg[0])

    foot_ids = segments.get('Foot', [])
    if len(foot_ids) != 2:
        print("エラー: task03のFootセグメントに2個のIDが必要です。")
        return
    # CONFIG の Foot: [つま先ID(67634,Z大), 踵ID(67630,Z小)]
    # 進行方向がZの正方向 → つま先の方がZが大きい
    tid_toe  = foot_ids[0]   # 67634: Z大(進行方向前)
    tid_heel = foot_ids[1]   # 67630: Z小(進行方向後ろ)

    print(f"  期待マーカー数 : {n_expected}")
    print(f"  Y降順TID順序  : {y_order}")
    print(f"  Toe(Z大)TID   : {tid_toe}, Heel(Z小)TID: {tid_heel}")

    # データ読み込み
    df_long = load_opti_data_to_long_robust(opti_csv_path)
    if df_long is None or df_long.empty:
        print("エラー: データ読み込み失敗。"); return

    # 空間範囲フィルタ (外れ値ノイズ除去)
    df_long = df_long[
        df_long['x'].between(*plausible_bounds['x']) &
        df_long['y'].between(*plausible_bounds['y']) &
        df_long['z'].between(*plausible_bounds['z'])
    ].copy()
    print(f"  空間範囲フィルタ後: {len(df_long)} 行")

    # id_map: {raw_id → template_id}  一度登録したら永続
    id_map = {}

    # フレームごとに処理
    output_rows = []
    unique_frames = sorted(df_long['Frame'].unique())
    print(f"  処理フレーム数: {len(unique_frames)}")

    for frame_idx, frame in enumerate(unique_frames):
        if frame_idx % 1000 == 0:
            print(f"  frame {frame_idx}/{len(unique_frames)} ...")

        frame_df = df_long[df_long['Frame'] == frame]
        time_val = frame_df['Time'].iloc[0]
        obs = frame_df[['id', 'x', 'y', 'z']].copy()
        obs['id'] = obs['id'].astype(int)
        n_obs = len(obs)

        if n_obs == n_expected:
            # =====================================================
            # ★ 5点揃い: anatomical判別でTIDを確定
            #
            # 判別手順:
            #   Step1: Y降順で上位2点 → Hip(67628), Knee(67626) に確定
            #          (Hip/KneeがToe/Heel/Ankleを下回ることはない)
            #   Step2: 残り3点のうち最もYが大きい点 → Ankle(67632)
            #          (Ankleは稀に足部と逆転する可能性があるため単独で確認)
            #   Step3: 残り2点をZ降順 → Z大=Heel(67634), Z小=Toe(67630)
            # =====================================================
            obs_sorted = obs.sort_values('y', ascending=False).reset_index(drop=True)

            new_map = {}
            ok = True

            # Step1: Y上位2点 → Hip, Knee
            for i, tid in enumerate(y_order[:2]):   # [67628, 67626]
                new_map[int(obs_sorted.iloc[i]['id'])] = tid

            # Step2: 残り3点のうちY最大 → Ankle
            rest3 = obs_sorted.iloc[2:].reset_index(drop=True)
            if len(rest3) != 3:
                ok = False
            else:
                ankle_tid = y_order[2]   # 67632
                new_map[int(rest3.iloc[0]['id'])] = ankle_tid  # rest3はすでにY降順

                # Step3: 残り2点をZ降順 → Toe/Heel
                foot2 = rest3.iloc[1:].reset_index(drop=True)
                foot_sorted_z = foot2.sort_values('z', ascending=False).reset_index(drop=True)
                new_map[int(foot_sorted_z.iloc[0]['id'])] = tid_toe   # Z大 → Toe(進行方向前)
                new_map[int(foot_sorted_z.iloc[1]['id'])] = tid_heel  # Z小 → Heel(進行方向後ろ)

            if ok:
                # 新しいマッピングで id_map を更新（既存登録と矛盾があれば上書き）
                for raw_id, tid in new_map.items():
                    if raw_id in id_map and id_map[raw_id] != tid:
                        print(f"  [ID更新] raw={raw_id}: {id_map[raw_id]} → {tid}  @t={time_val:.2f}s")
                    id_map[raw_id] = tid

                # 出力行を追加
                for _, row in obs_sorted.iterrows():
                    raw_id = int(row['id'])
                    tid = new_map[raw_id]
                    output_rows.append((frame, time_val, tid, row['x'], row['y'], row['z']))
            else:
                # anatomical判別失敗 → id_map済みのraw_idだけ出力
                for _, row in obs.iterrows():
                    raw_id = int(row['id'])
                    if raw_id in id_map:
                        output_rows.append((frame, time_val, id_map[raw_id],
                                            row['x'], row['y'], row['z']))

        elif n_obs > 0:
            # =====================================================
            # ★ 5点未満: id_map登録済みのraw_idだけTID変換して採用
            # =====================================================
            for _, row in obs.iterrows():
                raw_id = int(row['id'])
                if raw_id in id_map:
                    output_rows.append((frame, time_val, id_map[raw_id],
                                        row['x'], row['y'], row['z']))
        # n_obs == 0 → 出力なし（補完しない）

    if not output_rows:
        print("エラー: 出力データが0行になりました。")
        return

    df_out = pd.DataFrame(output_rows, columns=['Frame', 'Time', 'id', 'x', 'y', 'z'])

    # 同一フレーム・同一TIDに複数行が入った場合は先着優先で重複除去
    df_out = df_out.drop_duplicates(subset=['Frame', 'id'], keep='first')
    df_out = df_out.sort_values(['Frame', 'id']).reset_index(drop=True)

    # 座標系統一: 生OptiTrack (X=左右, Y=鉛直上, Z=進行前) → 統一系 (X=前, Y=横, Z=上)
    x_raw = df_out['x'].copy()
    y_raw = df_out['y'].copy()
    z_raw = df_out['z'].copy()
    df_out['x'] = z_raw
    df_out['y'] = x_raw
    df_out['z'] = y_raw
    print("  [座標変換] X=横,Y=上,Z=前 → X=前,Y=横,Z=上 を適用しました。")

    print(f"\n処理完了。出力行数: {len(df_out)}")
    print(f"  id_map 最終登録数: {len(id_map)} 件")
    for raw_id, tid in sorted(id_map.items()):
        print(f"    raw={raw_id} → tid={tid}")

    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"保存完了: {output_csv_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n=== OptiTrack Data Cleanup (opti_edit_D with Smart Resync) ===")
    
    while True:
        try:
            task_key = input("クリーンアップ処理を行うタスク名 (task01, task02, task03): ").strip().lower()
            if task_key in CONFIG.TASK_CONFIGS:
                cfg = CONFIG.TASK_CONFIGS[task_key]
                print(f"[{task_key}] の設定を CONFIG.py から読み込みました。")
                break
            else:
                print("エラー: CONFIG.py に定義されていないタスク名です。")
        except KeyboardInterrupt:
            sys.exit()
            
    opti_csv_path = cfg.get('OPTI_CSV_PATH')
    output_csv_path = cfg.get('OUTPUT_CSV_PATH')

    # =========================================================
    # task03 は専用フローで処理して終了
    # =========================================================
    if task_key == 'task03':
        process_task03(cfg)
        return

    static_start = cfg.get('STATIC_START', 0.0)
    static_end = cfg.get('STATIC_END', 10.0)
    t1_static_end = cfg.get('T1_STATIC_END', 10.0)
    t1_walk_start = cfg.get('T1_WALK_START', 10.0)
    t2_walk_end = cfg.get('T2_WALK_END', 50.0)
    t2_static_start = cfg.get('T2_STATIC_START', 50.0)
    
    segments = cfg.get('SEGMENTS', {})
    keyframe_map = cfg.get('KEYFRAME_MAP', {})
    keyframe_time = cfg.get('KEYFRAME_TIME', t1_walk_start)
    plausible_bounds = cfg.get('PLAUSIBLE_BOUNDS', {'x': (-1000, 1000), 'y': (0, 2000), 'z': (-2000, 2000)})
    
    processing_order = CONFIG.PROCESSING_ORDER
    hierarchy = CONFIG.CHAIN_HIERARCHY
    matching_threshold = CONFIG.MATCHING_THRESHOLD_MM

    template_ids = []
    for seg_list in segments.values():
        template_ids.extend(seg_list)
    template_id_set = set(template_ids)

    # Resync モードと解剖学的ルールの構築
    resync_mode = cfg.get('RESYNC_MODE', 'y_slice')
    anatomical_rules = None
    if resync_mode == 'anatomical':
        # CONFIG の SEGMENTS から自動生成:
        #   y_order  = Foot以外のセグメントを処理順に並べた1点ずつのID
        #   foot_z   = Footセグメントの2点を {z_high: Heel, z_low: Toe} に対応
        #
        # task03の具体例:
        #   y_order  = [67628(Hip), 67626(Knee), 67632(Ankle)]
        #   foot_z   = {'z_high': 67634(Heel), 'z_low': 67630(Toe)}
        #
        # ★ foot_z の割り当ては CONFIG.TASK_CONFIGS[task]['FOOT_Z_RULE'] で
        #   明示指定することもできる。未定義の場合は Foot セグメントの
        #   最初のIDを z_high(Heel)、2番目を z_low(Toe) として扱う。
        y_order = []
        for seg_name in CONFIG.PROCESSING_ORDER:
            if seg_name == 'Foot': continue
            ids_in_seg = segments.get(seg_name, [])
            if len(ids_in_seg) == 1:
                y_order.append(ids_in_seg[0])
            elif len(ids_in_seg) > 1:
                print(f"警告: anatomicalモードでセグメント'{seg_name}'に複数マーカー。先頭IDのみ使用します。")
                y_order.append(ids_in_seg[0])

        foot_ids = segments.get('Foot', [])
        if cfg.get('FOOT_Z_RULE'):
            foot_z = cfg['FOOT_Z_RULE']
        elif len(foot_ids) == 2:
            foot_z = {'z_high': foot_ids[0], 'z_low': foot_ids[1]}
        else:
            print("エラー: anatomicalモードにはFootセグメントに2個のIDが必要です。")
            return

        anatomical_rules = {'y_order': y_order, 'foot_z': foot_z}
        print(f"  anatomical Resyncルール: y_order={y_order}, foot_z={foot_z}")

    df_long = load_opti_data_to_long_robust(opti_csv_path)
    
    if df_long is not None and not df_long.empty:
        static_mean_pos, templates = build_segment_templates(
            df_long, segments, hierarchy, template_ids, static_start, static_end
        )
        
        if static_mean_pos and templates:
            df_long_walk = df_long[
                (df_long['Time'] >= t1_walk_start) & (df_long['Time'] <= t2_walk_end)
            ].copy()
            if df_long_walk.empty:
                print(f"エラー: 歩行区間 ({t1_walk_start}s - {t2_walk_end}s) のデータが0行です。"); return

            key_frame_index, initial_pose_map, initial_id_map = get_keyframe_data(
                df_long, keyframe_time, keyframe_map, template_ids, plausible_bounds
            )
            
            if initial_pose_map:
                df_walk_processed = track_walking_with_id_map(
                    df_long_walk, key_frame_index, initial_pose_map,
                    templates, template_ids, initial_id_map,
                    plausible_bounds, template_id_set, matching_threshold,
                    processing_order, segments, hierarchy,
                    resync_mode=resync_mode, anatomical_rules=anatomical_rules
                )

                df_full_scaffold = create_full_scaffold(df_long, template_ids)
                df_with_static = fill_static_zones(df_full_scaffold, static_mean_pos, t1_static_end, t2_static_start)

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

                df_final = final_interpolate_gaps(df_merged)

                if df_final.empty:
                    print("エラー: 最終処理結果が空になりました。")
                else:
                    out_dir = os.path.dirname(output_csv_path)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                    df_final.to_csv(output_csv_path, index=False, float_format='%.6f')
                    print(f"補正済みデータを保存しました: {output_csv_path}")
            else:
                print("エラー: キーフレームを処理できませんでした。CONFIGの KEYFRAME_MAP を確認してください。")
        else:
            print("エラー: テンプレートを作成できなかったため、処理を中断します。")

if __name__ == "__main__":
    main()