# =============================================================================
# opti_edit_auto_gait.py (v16.1 - Ultra-Fast Y-Stratification with Hip Safety)
#
# 概要:
#   LabChartの歩行周期データを用い、全歩行周期を連続的にクリーンアップ・補完する。
#   修正:
#   - 計算速度を数秒に抑えるNumpy配列による一括処理（v10.0ベース）を使用。
#   - 毎フレームの強制スワップを行わず、0, 50, 60, 70% の4つのフレームでのみ絶対再同期。
#   - 70%で非常に上手く機能した「下位7マーカーの厳密なYスライス＋X/Z判別」のルールを、
#     60%のフェーズにも適用するように統合しました。
#   - ★追加: どの区間・どのフレームにおいても「16012のZ > 16014のZ」を強制する
#     セーフティスワップ処理を毎フレームのトラッキングの最後に導入しました。
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# --- CONFIG.py のパス解決 ---
# このスクリプトは C:\FuttoAnalysis\2026_analysis\ に置かれている想定。
# CONFIG.py は C:\FuttoAnalysis\2026_analysis\futto_common\ に存在する。
_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futto_common')
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import CONFIG

# =============================================================================
# ユーティリティ関数
# =============================================================================

def show_matching_result_plotly(raw_frame_df, id_map, actual_time):
    fig = go.Figure()
    matched_df = raw_frame_df[raw_frame_df['id'].isin(id_map.keys())].copy()
    unmatched_df = raw_frame_df[~raw_frame_df['id'].isin(id_map.keys())].copy()
    
    if not matched_df.empty:
        texts = [f"TID:{id_map[rid]}<br>(Raw:{int(rid)})" for rid in matched_df['id']]
        fig.add_trace(go.Scatter3d(
            x=matched_df['x'], y=matched_df['y'], z=matched_df['z'],
            mode='markers+text', marker=dict(size=6, color='green'),
            text=texts, textposition='middle right', textfont=dict(color='green', size=12),
            name="Matched"
        ))
        
    if not unmatched_df.empty:
        fig.add_trace(go.Scatter3d(
            x=unmatched_df['x'], y=unmatched_df['y'], z=unmatched_df['z'],
            mode='markers+text', marker=dict(size=4, color='gray', opacity=0.5),
            text=unmatched_df['id'].astype(int).astype(str),
            textposition='middle left', textfont=dict(color='gray', size=8),
            name="Unmatched (Noise)"
        ))

    fig.update_layout(
        title=f"Cycle 1 Initial Matching Result (Time: {actual_time:.3f}s)",
        scene=dict(
            xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
            aspectmode='data', xaxis=dict(range=[-500, 300]),
            yaxis=dict(range=[0, 1500]), zaxis=dict(range=[0, 1400])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def load_opti_data(file_path):
    print(f"'{os.path.basename(file_path)}' を読み込み中...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダー
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5: continue
                frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4]); base_col = 5
                if len(parts) >= base_col + n_markers * 4:
                    for i in range(n_markers):
                        x, y, z = float(parts[base_col + 4*i])*1000, float(parts[base_col + 4*i + 1])*1000, float(parts[base_col + 4*i + 2])*1000
                        mid = int(parts[base_col + 4*i + 3])
                        rows.append((frame, t, mid, x, y, z))
        return pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])
    except Exception as e: print(f"読み込みエラー: {e}"); return None

# =============================================================================
# 事前データクレンジング関数
# =============================================================================

def clean_global_noise(df, template_ids, first_hs_time):
    print(f"\n--- 事前データクレンジングを実行 (歩行開始 {first_hs_time:.2f}s 以降を対象) ---")
    original_len = len(df)
    bounds = {'x': (-500, 300), 'y': (0, 1500), 'z': (0, 1400)}
    protected_ids = set(template_ids)
    
    mask_bounds = (df['x'].between(*bounds['x']) & 
                   df['y'].between(*bounds['y']) & 
                   df['z'].between(*bounds['z']))
    df = df[mask_bounds].copy()
    print(f"  ✅ 空間範囲外ノイズを除去: {original_len} 行 -> {len(df)} 行")

    grouped = df.groupby('id')
    ranges = grouped[['x', 'y', 'z']].agg(lambda x: x.max() - x.min())
    counts = grouped.size()
    always_static_ids = set(ranges[(ranges['x'] < 10.0) & (ranges['y'] < 10.0) & (ranges['z'] < 10.0) & (counts >= 10)].index)
    always_static_ids = always_static_ids - protected_ids
    
    if always_static_ids:
        df = df[~df['id'].isin(always_static_ids)].copy() 
        print(f"  ✅ 完全に固定されたノイズIDを削除 ({len(always_static_ids)}個)")

    print("  ⏳ 途中でフリーズするゴーストマーカーの静止区間を探索中...")
    df_sorted = df.sort_values(['id', 'Frame']).copy()
    diffs = df_sorted.groupby('id')[['x', 'y', 'z']].diff().abs()
    
    is_frozen = (diffs['x'] < 0.5) & (diffs['y'] < 0.5) & (diffs['z'] < 0.5)
    changed = ~is_frozen
    df_sorted['group'] = changed.groupby(df_sorted['id']).cumsum()
    df_sorted['group_size'] = df_sorted.groupby(['id', 'group'])['Frame'].transform('count')
    
    to_drop_mask = (df_sorted['group_size'] >= 20) & (df_sorted['Time'] >= first_hs_time)
    
    num_dropped = to_drop_mask.sum()
    if num_dropped > 0:
        drop_indices = df_sorted[to_drop_mask].index
        df = df.drop(index=drop_indices).copy()
        affected_ids = df_sorted[to_drop_mask]['id'].nunique()
        print(f"  ✅ フリーズしたゴーストデータを部分削除 ({affected_ids} 個のIDで検出, 計 {num_dropped} 行削除)")

    print(f"  事前クレンジング完了: 最終データ数 {len(df)} 行\n")
    return df.reset_index(drop=True)

# =============================================================================
# 絶対再同期エンジン (高速Numpy版)
# =============================================================================

def absolute_resync_by_rules_fast(obs_coords, obs_ids, reference_pose, phase_type):
    """
    指定された歩行相 (0, 50, 60, 70) に応じて、生マーカーをテンプレートIDに強制割り当てする。
    トラッキング結果に依存せず、15個の生データを純粋にY座標でソートして階層化する。
    """
    # Yの降順（上から下）にソート
    sort_idx = np.argsort(obs_coords[:, 1])[::-1]
    sorted_coords = obs_coords[sort_idx]
    sorted_ids = obs_ids[sort_idx]
    
    new_id_map = {}
    
    # ---------------------------------------------
    # 全フェーズ共通の上位8個のマーカー処理
    # ---------------------------------------------
    # 1. Hip (上から4つ: 0,1,2,3) -> Z降順
    hip_coords = sorted_coords[0:4]
    hip_ids = sorted_ids[0:4]
    z_idx = np.argsort(hip_coords[:, 2])[::-1]
    hip_targets = [16000, 16012, 16014, 15960]
    for i, idx in enumerate(z_idx): new_id_map[hip_ids[idx]] = hip_targets[i]
        
    # 2. Thigh/Knee Upper (次の2つ: 4,5) -> Z降順
    th_coords = sorted_coords[4:6]
    th_ids = sorted_ids[4:6]
    z_idx = np.argsort(th_coords[:, 2])[::-1]
    th_targets = [15970, 15958]
    for i, idx in enumerate(z_idx): new_id_map[th_ids[idx]] = th_targets[i]
        
    # 3. Knee L/R (次の2つ: 6,7) -> 直前リファレンスと比較(ICP)
    kn_coords = sorted_coords[6:8]
    kn_ids = sorted_ids[6:8]
    kn_targets = [15956, 15968]
    ref_kn = np.array([reference_pose.get(t, kn_coords[0]) for t in kn_targets])
    cost = cdist(ref_kn, kn_coords)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind): new_id_map[kn_ids[c]] = kn_targets[r]

    # ---------------------------------------------
    # フェーズごとの下位7個のマーカー処理
    # ---------------------------------------------
    if phase_type in [0, 50]:
        # 4. Shank H (インデックス 8)
        new_id_map[sorted_ids[8]] = 15974
        
        # 5. Shank L / Toe (次の2つ: 9, 10) -> Z降順
        st_coords = sorted_coords[9:11]
        st_ids = sorted_ids[9:11]
        z_idx = np.argsort(st_coords[:, 2])[::-1]
        st_targets = [15950, 15972]
        for i, idx in enumerate(z_idx): new_id_map[st_ids[idx]] = st_targets[i]
            
        # 6. Foot 残り4つ (インデックス 11,12,13,14)
        ft_coords = sorted_coords[11:15]
        ft_ids = sorted_ids[11:15]
        z_idx = np.argsort(ft_coords[:, 2])[::-1]
        
        # 前(Z大)2つ と 後(Z小)2つ に分割
        front_idx = z_idx[0:2]
        back_idx = z_idx[2:4]
        
        # 前2つ: Xの降順 (15964 > 15918)
        front_coords = ft_coords[front_idx]
        front_ids = ft_ids[front_idx]
        x_idx = np.argsort(front_coords[:, 0])[::-1]
        new_id_map[front_ids[x_idx[0]]] = 15964
        new_id_map[front_ids[x_idx[1]]] = 15918
        
        # 後2つ: 直前リファレンスと比較(ICP)
        back_coords = ft_coords[back_idx]
        back_ids = ft_ids[back_idx]
        back_targets = [15966, 15948]
        ref_back = np.array([reference_pose.get(t, back_coords[0]) for t in back_targets])
        cost = cdist(ref_back, back_coords)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind): new_id_map[back_ids[c]] = back_targets[r]

    elif phase_type in [60, 70]:
        # ---------------------------------------------
        # ★ 60%, 70% 共通の条件 (下位7個のマーカー: インデックス 8〜14)
        # ---------------------------------------------
        # 4. Yが最も小さい2つ(ボトム2つ: 13, 14) -> 15964, 15918
        low_coords = sorted_coords[13:15]
        low_ids = sorted_ids[13:15]
        x_idx = np.argsort(low_coords[:, 0])[::-1] # Xの降順 (15964 > 15918)
        new_id_map[low_ids[x_idx[0]]] = 15964
        new_id_map[low_ids[x_idx[1]]] = 15918
        
        # 5. その次にYが小さい3つ(インデックス: 10, 11, 12) -> 15950, 15966, 15948
        mid3_coords = sorted_coords[10:13]
        mid3_ids = sorted_ids[10:13]
        z_idx = np.argsort(mid3_coords[:, 2])[::-1] # Zの降順
        idx_15950 = z_idx[0] # Z最大のものが15950
        new_id_map[mid3_ids[idx_15950]] = 15950
        
        # 15950を除外した残り2つ [15966, 15948]
        rem2_mask = np.ones(3, dtype=bool)
        rem2_mask[idx_15950] = False
        rem2_coords = mid3_coords[rem2_mask]
        rem2_ids = mid3_ids[rem2_mask]
        x_idx2 = np.argsort(rem2_coords[:, 0])[::-1] # Xの降順 (15948 > 15966)
        new_id_map[rem2_ids[x_idx2[0]]] = 15948
        new_id_map[rem2_ids[x_idx2[1]]] = 15966
        
        # 6. その次にYが小さい2つ(インデックス: 8, 9) -> 15974, 15972
        upper2_coords = sorted_coords[8:10]
        upper2_ids = sorted_ids[8:10]
        z_idx_u = np.argsort(upper2_coords[:, 2])[::-1] # Zの降順 (15974 > 15972)
        new_id_map[upper2_ids[z_idx_u[0]]] = 15974
        new_id_map[upper2_ids[z_idx_u[1]]] = 15972

    return new_id_map

def find_full_marker_frame_fast(target_time, unique_frames, times_of_frames, frame_to_idx, expected_count=15, window=0.1):
    """指定時刻周辺で、期待する数のマーカーが揃っているフレームのインデックスを高速に探す"""
    mask = (times_of_frames >= target_time - window) & (times_of_frames <= target_time + window)
    candidate_frames = unique_frames[mask]
    candidate_times = times_of_frames[mask]
    
    valid_frames = []
    valid_times = []
    for f, t in zip(candidate_frames, candidate_times):
        s, e = frame_to_idx[f]
        if (e - s) == expected_count:
            valid_frames.append(f)
            valid_times.append(t)
            
    if valid_frames:
        idx = np.abs(np.array(valid_times) - target_time).argmin()
        return valid_frames[idx]
    return None

# =============================================================================
# 毎フレームのキネマティック予測
# =============================================================================

def build_kinematic_templates(template_info, segments, hierarchy):
    static_centroids, marker_rel_vectors, segment_of_tid = {}, {}, {}
    for seg_name, tids in segments.items():
        valid_pos = [template_info[tid] for tid in tids if tid in template_info]
        static_centroids[seg_name] = np.mean(valid_pos, axis=0) if valid_pos else np.zeros(3)
        for tid in tids:
            segment_of_tid[tid] = seg_name
            if tid in template_info:
                marker_rel_vectors[tid] = template_info[tid] - static_centroids[seg_name]
    return static_centroids, marker_rel_vectors, segment_of_tid

def get_predicted_pos(tid, current_centroids, kinematic_templates, hierarchy, last_known_pos, template_info):
    static_centroids, marker_rel_vectors, segment_of_tid = kinematic_templates
    if tid not in segment_of_tid or tid not in marker_rel_vectors: 
        return last_known_pos.get(tid, template_info.get(tid, np.full(3, np.nan)))
    
    seg_name = segment_of_tid[tid]
    curr_seg = seg_name
    while curr_seg is not None:
        if curr_seg in current_centroids: break
        curr_seg = hierarchy.get(curr_seg)
        
    if curr_seg is None: 
        return last_known_pos.get(tid, template_info.get(tid, np.full(3, np.nan)))
    
    predicted_seg_centroid = current_centroids[curr_seg]
    if seg_name != curr_seg:
        offset = static_centroids[seg_name] - static_centroids[curr_seg]
        predicted_seg_centroid = predicted_seg_centroid + offset
        
    return predicted_seg_centroid + marker_rel_vectors[tid]

# =============================================================================
# メイン処理
# =============================================================================

def main():
    task_key = 'task01'
    cfg = CONFIG.TASK_CONFIGS[task_key]
    
    template_ids = []
    for seg_list in cfg['SEGMENTS'].values(): template_ids.extend(seg_list)
    
    # 1. データロード
    df_raw = load_opti_data(cfg['OPTI_CSV_PATH'])
    cycles_df = pd.read_csv(cfg['LABCHART_CYCLES_PATH'])
    if df_raw is None: return
    
    valid_cycles = cycles_df[cycles_df['hs_time'] >= 40.0].reset_index(drop=True)
    if valid_cycles.empty:
        print("エラー: 40秒以降の歩行周期が見つかりません。")
        return
    first_hs_time = valid_cycles.iloc[0]['hs_time']
    
    # 2. 事前データクレンジング
    df_all = clean_global_noise(df_raw, template_ids, first_hs_time)

    # 3. 静止テンプレート作成
    print("立位テンプレートを作成中...")
    static_df = df_all[(df_all['Time'] >= cfg['STATIC_START']) & (df_all['Time'] <= cfg['STATIC_END'])]
    template_info_raw = static_df[static_df['id'].isin(template_ids)].groupby('id')[['x','y','z']].mean().to_dict('index')
    
    missing_tids = [tid for tid in template_ids if tid not in template_info_raw]
    if missing_tids:
        print(f"\n❌ エラー: 静止区間で以下のテンプレートIDが見つかりませんでした:\n  {missing_tids}")
        return

    template_info = {k: np.array([v['x'], v['y'], v['z']]) for k, v in template_info_raw.items()}
    kinematic_templates = build_kinematic_templates(template_info, cfg['SEGMENTS'], CONFIG.CHAIN_HIERARCHY)

    # --- 高速化のための Numpy 配列化 ---
    print("\n--- 高速トラッキング処理を開始します (数秒で完了します) ---")
    # 歩行区間のみに絞る
    walk_start = valid_cycles.iloc[0]['hs_time']
    walk_end = valid_cycles.iloc[-1]['next_hs_time']
    df_walk = df_all[(df_all['Time'] >= walk_start) & (df_all['Time'] <= walk_end)]
    
    frames = df_walk['Frame'].values
    times = df_walk['Time'].values
    ids = df_walk['id'].values
    coords = df_walk[['x', 'y', 'z']].values
    
    unique_frames, start_indices = np.unique(frames, return_index=True)
    end_indices = np.append(start_indices[1:], len(frames))
    frame_to_idx = {f: (s, e) for f, s, e in zip(unique_frames, start_indices, end_indices)}
    times_of_frames = np.array([times[s] for s in start_indices])
    
    # 絶対再同期を行うフレームを事前に計算
    resync_target_frames = {}
    for i, row in valid_cycles.iterrows():
        t_start = row['hs_time']
        t_end = row['next_hs_time']
        
        t_0 = t_start
        t_50 = t_start + 0.5 * (t_end - t_start)
        t_60 = t_start + 0.6 * (t_end - t_start)
        t_70 = t_start + 0.7 * (t_end - t_start)
        
        f_0 = find_full_marker_frame_fast(t_0, unique_frames, times_of_frames, frame_to_idx)
        f_50 = find_full_marker_frame_fast(t_50, unique_frames, times_of_frames, frame_to_idx)
        f_60 = find_full_marker_frame_fast(t_60, unique_frames, times_of_frames, frame_to_idx)
        f_70 = find_full_marker_frame_fast(t_70, unique_frames, times_of_frames, frame_to_idx)
        
        if f_0 is not None: resync_target_frames[f_0] = 0
        if f_50 is not None: resync_target_frames[f_50] = 50
        if f_60 is not None: resync_target_frames[f_60] = 60
        if f_70 is not None: resync_target_frames[f_70] = 70

    # リファレンスポーズの初期化
    last_resync_poses = {
        0: template_info.copy(),
        50: template_info.copy(),
        60: template_info.copy(),
        70: template_info.copy()
    }
    
    full_corrected_rows = []
    debug_mapping_rows = []
    id_map = {}
    last_known_pos = {}
    
    # 初回表示フラグ
    first_resync_done = False
    
    # 全フレームの高速イテレーション
    for f, t_val in zip(unique_frames, times_of_frames):
        s, e = frame_to_idx[f]
        obs_ids = ids[s:e]
        obs_coords = coords[s:e]
        
        # === A. 絶対再同期の適用 ===
        phase_type = resync_target_frames.get(f)
        if phase_type is not None:
            resync_map = absolute_resync_by_rules_fast(obs_coords, obs_ids, last_resync_poses[phase_type], phase_type)
            if len(resync_map) == 15:
                id_map.update(resync_map) # マッピングを強制更新し、未来に引き継ぐ
                
                # 新しい姿勢を保存して次回の基準にする
                temp_pose = {}
                for raw_id, tid in resync_map.items():
                    idx = np.where(obs_ids == raw_id)[0]
                    if len(idx) > 0: temp_pose[tid] = obs_coords[idx[0]]
                last_resync_poses[phase_type] = temp_pose
                
                # 初回のみプロット表示
                if not first_resync_done and phase_type == 0:
                    print("\n=== 🎯 Cycle 1 (0%) マッチング結果 ===")
                    for r_id, t_id in resync_map.items(): print(f"  Raw ID: {int(r_id)}  -->  Template ID: {t_id}")
                    print("=================================\n")
                    df_first = pd.DataFrame({'id': obs_ids, 'x': obs_coords[:,0], 'y': obs_coords[:,1], 'z': obs_coords[:,2]})
                    show_matching_result_plotly(df_first, resync_map, t_val)
                    first_resync_done = True
        
        # === B. 通常のトラッキング (毎フレーム) ===
        frame_pose = {}
        assigned_raw_ids = set()
        used_raw_map = {}
        
        # 1. 前フレームから引き継いだ id_map で生データを最優先割り当て
        for raw_id, pos in zip(obs_ids, obs_coords):
            if raw_id in id_map:
                tid = id_map[raw_id]
                frame_pose[tid] = pos
                assigned_raw_ids.add(raw_id)
                used_raw_map[tid] = raw_id
                
        # 現在のセグメント重心を計算
        current_centroids = {}
        for seg_name in CONFIG.PROCESSING_ORDER:
            pts = [frame_pose[tid] for tid in cfg['SEGMENTS'][seg_name] if tid in frame_pose]
            if pts: current_centroids[seg_name] = np.mean(pts, axis=0)

        # 2. 未割り当てTIDを近傍探索(ICP)で埋める (実在する生データを優先利用)
        unassigned_tids = [tid for tid in template_ids if tid not in frame_pose]
        available_indices = [i for i, r_id in enumerate(obs_ids) if r_id not in assigned_raw_ids]
        
        if unassigned_tids and available_indices:
            raw_coords_avail = obs_coords[available_indices]
            raw_ids_avail = obs_ids[available_indices]
            
            prev_coords = []
            for tid in unassigned_tids:
                pred_pos = get_predicted_pos(tid, current_centroids, kinematic_templates, CONFIG.CHAIN_HIERARCHY, last_known_pos, template_info)
                prev_coords.append(pred_pos)
            prev_coords = np.array(prev_coords)
            
            cost = cdist(prev_coords, raw_coords_avail)
            r_idx, c_idx = linear_sum_assignment(cost)
            
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] < 150: # 15cm以内なら新しい生データを採用
                    tid = unassigned_tids[ri]
                    pos = raw_coords_avail[ci]
                    raw_id = raw_ids_avail[ci]
                    
                    frame_pose[tid] = pos
                    id_map[raw_id] = tid # 新しいマーカー対応を学習(上書き)
                    used_raw_map[tid] = raw_id
                    
                    seg = kinematic_templates[2].get(tid)
                    if seg:
                        pts = [frame_pose[t] for t in cfg['SEGMENTS'][seg] if t in frame_pose]
                        current_centroids[seg] = np.mean(pts, axis=0)

        # === ★ C. 特別なセーフティ制約 (Hipマーカーの入れ替わり防止) ===
        # どの区間においても、16012 は 16014 よりも Z が大きい (前にある)
        if 16012 in frame_pose and 16014 in frame_pose:
            if frame_pose[16012][2] <= frame_pose[16014][2]:
                # 座標をスワップ
                temp_pos = frame_pose[16012]
                frame_pose[16012] = frame_pose[16014]
                frame_pose[16014] = temp_pos
                
                # id_map を更新して未来のフレームにも引き継ぐ
                raw_16012 = used_raw_map.get(16012)
                raw_16014 = used_raw_map.get(16014)
                
                if raw_16012 is not None:
                    id_map[raw_16012] = 16014
                    used_raw_map[16014] = raw_16012
                if raw_16014 is not None:
                    id_map[raw_16014] = 16012
                    used_raw_map[16012] = raw_16014

        # 3. 補完と記録
        for tid in template_ids:
            if tid not in frame_pose:
                pred_pos = get_predicted_pos(tid, current_centroids, kinematic_templates, CONFIG.CHAIN_HIERARCHY, last_known_pos, template_info)
                full_corrected_rows.append([t_val, f, tid, pred_pos[0], pred_pos[1], pred_pos[2]])
                last_known_pos[tid] = pred_pos
                used_raw_map[tid] = "Predicted"
            else:
                pos = frame_pose[tid]
                full_corrected_rows.append([t_val, f, tid, pos[0], pos[1], pos[2]])
                last_known_pos[tid] = pos
            
            debug_mapping_rows.append([f, t_val, tid, used_raw_map.get(tid, "Predicted")])

    # 4. メイン補正データの保存
    df_out = pd.DataFrame(full_corrected_rows, columns=["Time", "Frame", "id", "x", "y", "z"])
    output_path = cfg['OUTPUT_CSV_PATH'].replace('.csv', '_auto_sequential.csv')
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ 補正済みデータを保存しました: {output_path}")

    # 5. デバッグ用マッピングデータの保存 (PermissionError 保護付き)
    df_debug = pd.DataFrame(debug_mapping_rows, columns=["Frame", "Time", "Template_ID", "Raw_ID"])
    debug_output_path = cfg['OUTPUT_CSV_PATH'].replace('.csv', '_debug_mapping.csv')
    try:
        df_debug.to_csv(debug_output_path, index=False)
        print(f"✅ デバッグ用マッピング履歴を保存しました: {debug_output_path}")
    except PermissionError:
        print(f"\n❌ 警告: '{os.path.basename(debug_output_path)}' がExcel等で開かれているため保存できませんでした。")
        print("   ファイルを閉じてから再度実行してください。(メインの補正データは保存済みです)")

if __name__ == "__main__":
    main()