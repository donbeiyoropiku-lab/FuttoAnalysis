# =============================================================================
# opti_edit_for_joint.py
#
# 役割:
#   新実験(task01〜03すべてで左右の脚関節に直接マーカーを貼付する方式)の
#   OptiTrack生データを処理し、テンプレートID(CONFIG.JOINT_MARKER_TEMPLATE_IDS)
#   へのマッピング・座標変換・オクルージョン(遮蔽)検出を行う。
#
# opti_edit.py との関係:
#   opti_edit.py は旧実験(Futtoゴムマーカー / 旧task03の単脚5点)専用として
#   一切変更せず凍結する。このファイルはそこから独立した新規プログラムであり、
#   process_task03() の解剖学的判別ロジック(Y降順→Hip/Knee/Ankle、
#   Z降順→Toe/Heel)と、左右分割による両脚10点への拡張(opti_edit.py の
#   process_joint_markers() で使われているのと同じ考え方)を、この新実験の
#   データパス向けに独立コピー・適応したもの。
#
# 前実験との違い:
#   前実験は task03(器具非着用)のみ関節マーカー方式だった(他タスクは
#   Futtoゴムに貼付したマーカーを処理し、仮想関節は後で抽出していた)。
#   今回の実験は task01〜03 すべてで関節マーカー方式を用いる。
#   device_worn(Futto着用の有無)は task01/task02=True, task03=False。
#
# データパス:
#   入力: C:\FuttoAnalysis\opti\joint\{subject}\{day}\{task}.csv
#         (CONFIG.get_joint_opti_path で生成。既存の get_raw_path('opti', ...)
#          が指す C:\FuttoAnalysis\opti\{subject}\{day}\{task}.csv とは別物)
#   出力: 入力と同じフォルダに {task}_corrected_joints.csv
#         オクルージョンログ: {task}_occlusion_log.csv
#
# 出力データの下流互換性 (joint_analysis との関係):
#   joint_analysis/calc_joint_centers_main.py が直接読むのは、この
#   スクリプトの出力(Frame, Time, id, x, y, z の生フレームデータ)ではなく、
#   create_anime_grad.py が歩行周期平均化して生成する
#   "{task}_mean_cycle_PhaseN_{speed}ms.csv" (gait_cycle_%, id, x, y, z)。
#   したがって本スクリプトの出力をそのまま joint_analysis に渡すことはできず、
#   以下の追加設定・追加ステップが必要:
#     (1) 新実験のLABCHART側 歩行周期ファイル ({task}_gait_cycles.csv、
#         gaitcycle_force_labchart.py が生成) を新被験者・新パス向けに用意する。
#     (2) create_anime_grad.py の DataProcessor.calculate_mean_gait_cycle() を
#         REFERENCE_MARKER_ID=CONFIG.JOINT_REFERENCE_MARKER_ID (=L_Hip) で実行し、
#         mean_cycle CSV を生成する。
#     (3) CONFIG.JOINT_CENTER_DEFS_MULTI (本ファイル追加時に futto_common/CONFIG.py
#         へ追加済み。テンプレートID 1〜10 を 'single' タイプで参照する定義) を
#         joint_analysis の JOINT_CENTER_DEFS として使う。
#   (1)(2) は本スクリプトの範囲外(別途の歩行周期抽出・平均化処理)。
#   出力の列構成 (Frame, Time, id, x, y, z) 自体は create_anime_grad.py の
#   load_and_prepare_data() が読める形式と一致することを確認済み。
# =============================================================================

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- CONFIG.py のパス解決 ---
_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futto_common')
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import CONFIG  # 設定モジュールのインポート


# =============================================================================
# ヘルパー関数群 (opti_edit.py から独立コピー・適応)
# =============================================================================

def load_opti_data_to_long_robust(file_path):
    """OptiTrack CSV を頑健に読み込み、mm単位に変換する。(opti_edit.py と同一仕様)"""
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}"); return None
    print(f"'{os.path.basename(file_path)}' を読み込み中...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f)  # ヘッダー
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


def assign_heel_toe_by_z(candidate_df, tid_toe, tid_heel):
    """
    残り2点(踵・つま先候補)を、進行方向(Z軸、大きい方が前=つま先)に基づいて
    テンプレートIDへ割り当てる (process_task03() の判別ロジックと同一)。

    Parameters
    ----------
    candidate_df : pd.DataFrame  'id', 'z' 列を持つ、ちょうど2行のDataFrame
    tid_toe, tid_heel : int  割り当て先テンプレートID (Z大→toe, Z小→heel)

    Returns
    -------
    dict {raw_id(int): template_id(int)}  2点に一致しない場合は空dict
    """
    if len(candidate_df) != 2:
        return {}
    foot_sorted_z = candidate_df.sort_values('z', ascending=False).reset_index(drop=True)
    return {
        int(foot_sorted_z.iloc[0]['id']): tid_toe,   # Z大 → つま先(進行方向前)
        int(foot_sorted_z.iloc[1]['id']): tid_heel,  # Z小 → 踵(進行方向後ろ)
    }


def split_left_right_by_x(obs_df):
    """
    観測点群をX座標(左右方向)に基づき2グループに分割する。

    脚は左右に離れて配置されるため、X値でソートし、最大ギャップ(隙間)の
    直後で2分割する(k-meansの代わりに、外れ値に強い単純な手法を採用)。

    Returns
    -------
    (group_a, group_b) : (pd.DataFrame, pd.DataFrame) いずれかの順序不定
                          (左右どちらがgroup_aかは呼び出し側でX平均から判定する)
    """
    if len(obs_df) < 2:
        return None, None
    sorted_df = obs_df.sort_values('x').reset_index(drop=True)
    x_vals = sorted_df['x'].to_numpy()
    gaps = np.diff(x_vals)
    if len(gaps) == 0:
        return None, None
    split_idx = int(np.argmax(gaps)) + 1
    group_a = sorted_df.iloc[:split_idx].reset_index(drop=True)
    group_b = sorted_df.iloc[split_idx:].reset_index(drop=True)
    return group_a, group_b


def resync_bilateral_anatomical(obs_df, template_ids_map, left_is_negative_x=True):
    """
    左右10点(各5点: Hip/Knee/Ankle/Heel/Toe)の解剖学的座標ルールによる Resync。

    判別手順 (各脚ごとに process_task03() の anatomical 判別と同じロジックを適用):
      1. X座標で左右2グループに分割 (split_left_right_by_x、最大ギャップで2分割)。
      2. 各グループ内でY降順上位3点 → Hip / Knee / Ankle に確定。
      3. 残り2点をZ降順 → Toe(Z大)/Heel(Z小) に確定 (assign_heel_toe_by_z)。

    Parameters
    ----------
    obs_df : pd.DataFrame  'id','x','y','z' 列を持つ、ちょうど10行のDataFrame
    template_ids_map : dict  CONFIG.JOINT_MARKER_TEMPLATE_IDS
        ({'L_Hip': 1, 'L_Knee': 2, ..., 'R_Toe': 10} 形式)
    left_is_negative_x : bool
        True の場合、X平均が小さい(負側)グループを左脚とみなす。
        ★ OptiTrackワールド座標のX軸符号はキャリブレーション設定に依存するため、
          実データ取得後に必ず検証すること (要検証事項)。

    Returns
    -------
    (id_map, ok) : ({raw_id(int): template_id(int)}, bool)
    """
    if len(obs_df) != 10:
        return {}, False

    group_a, group_b = split_left_right_by_x(obs_df)
    if group_a is None or len(group_a) != 5 or len(group_b) != 5:
        return {}, False

    mean_x_a = group_a['x'].mean()
    mean_x_b = group_b['x'].mean()
    a_is_left = (mean_x_a < mean_x_b) if left_is_negative_x else (mean_x_a > mean_x_b)
    left_grp, right_grp = (group_a, group_b) if a_is_left else (group_b, group_a)

    id_map = {}
    for leg_prefix, leg_df in (('L', left_grp), ('R', right_grp)):
        leg_sorted_y = leg_df.sort_values('y', ascending=False).reset_index(drop=True)
        if len(leg_sorted_y) != 5:
            return {}, False

        # Y上位3点 → Hip, Knee, Ankle (近位→遠位の順)
        for i, joint in enumerate(['Hip', 'Knee', 'Ankle']):
            tid = template_ids_map[f'{leg_prefix}_{joint}']
            id_map[int(leg_sorted_y.iloc[i]['id'])] = tid

        # 残り2点 → Toe/Heel (Z降順、共通関数を使用)
        foot2 = leg_sorted_y.iloc[3:].reset_index(drop=True)
        tid_toe = template_ids_map[f'{leg_prefix}_Toe']
        tid_heel = template_ids_map[f'{leg_prefix}_Heel']
        id_map.update(assign_heel_toe_by_z(foot2, tid_toe, tid_heel))

    ok = (len(id_map) == 10)
    return id_map, ok


# =============================================================================
# メイン処理: 新実験(task01〜03すべて関節マーカー方式)専用
# =============================================================================

def process_joint_markers_new_experiment(subject, day, task, device_worn,
                                          opti_csv_path=None, output_csv_path=None,
                                          occlusion_log_path=None,
                                          left_is_negative_x=True):
    """
    新実験(task01〜03すべて左右10点の関節マーカー方式)の
    OptiTrack生データを処理する。

    方針 (process_task03() / process_joint_markers() と同じ):
      - 補完なし・前フレーム近傍探索なし
      - 10点揃っているフレーム → anatomical判別でTIDを確定し id_map に蓄積
      - 10点未満のフレーム    → id_map に登録済みのraw_idだけTIDに変換して採用
      - 観測できたデータのみ出力(NaN行・補完行は一切含まない)
      - 一度マッピングされた生マーカーID → テンプレートIDの対応は、以後のフレームでも固定する

    device_worn=True の場合(task01/task02):
        Futtoのゴムによるマーカー遮蔽(オクルージョン)が発生しうるため、
        連続 CONFIG.OCCLUSION_MIN_GAP_FRAMES フレーム以上の欠損区間を検出し、
        ログ (occlusion_log_path) に出力する。

    Parameters
    ----------
    subject, day, task : str  被験者・計測日('First'/'Second'/'Third')・タスク
    device_worn : bool  Futto着用の有無 (task01/task02=True, task03=False)
    opti_csv_path, output_csv_path : str または Path または None
        None の場合、CONFIG.get_joint_opti_path(subject, day, task) から自動生成する。
    occlusion_log_path : str または Path または None
        None の場合、output_csv_path と同じディレクトリに
        '{task}_occlusion_log.csv' として保存する。
    left_is_negative_x : bool  左右脚判別のX軸符号仮定 (要実データ検証)。

    Returns
    -------
    pd.DataFrame または None
    """
    print(f"\n--- process_joint_markers_new_experiment: {subject}/{day}/{task} "
          f"(device_worn={device_worn}) ---")

    if opti_csv_path is None:
        opti_csv_path = CONFIG.get_joint_opti_path(subject, day, task)
    opti_csv_path = Path(opti_csv_path)

    if output_csv_path is None:
        output_csv_path = opti_csv_path.parent / f"{opti_csv_path.stem}_corrected_joints.csv"
    output_csv_path = Path(output_csv_path)

    if occlusion_log_path is None:
        occlusion_log_path = output_csv_path.parent / f"{task}_occlusion_log.csv"
    occlusion_log_path = Path(occlusion_log_path)

    template_ids_map = CONFIG.JOINT_MARKER_TEMPLATE_IDS
    template_ids = list(template_ids_map.values())
    n_expected = len(template_ids)  # 10
    min_gap = CONFIG.OCCLUSION_MIN_GAP_FRAMES

    plausible_bounds = {'x': (-1500, 1500), 'y': (0, 2000), 'z': (-3000, 3000)}

    df_long = load_opti_data_to_long_robust(str(opti_csv_path))
    if df_long is None or df_long.empty:
        print("エラー: データ読み込み失敗。"); return None

    df_long = df_long[
        df_long['x'].between(*plausible_bounds['x']) &
        df_long['y'].between(*plausible_bounds['y']) &
        df_long['z'].between(*plausible_bounds['z'])
    ].copy()
    print(f"  空間範囲フィルタ後: {len(df_long)} 行")

    id_map = {}              # raw_id -> template_id (一度登録したら永続)
    registered_tids = set()  # 一度でも登録されたtemplate_id (遮蔽検出の起点)
    output_rows = []

    # オクルージョン検出用: template_idごとの現在の連続欠損長・開始位置
    missing_run = {tid: {'len': 0, 'start_time': None, 'start_frame': None}
                   for tid in template_ids}
    occlusion_events = []

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

        frame_tid_positions = {}  # template_id -> (x, y, z) このフレームで確定した座標

        if n_obs == n_expected:
            # ★ 10点揃い: 解剖学的判別でTIDを確定
            new_map, ok = resync_bilateral_anatomical(obs, template_ids_map, left_is_negative_x)
            if ok:
                for raw_id, tid in new_map.items():
                    if raw_id in id_map and id_map[raw_id] != tid:
                        print(f"  [ID更新] raw={raw_id}: {id_map[raw_id]} -> {tid}  @t={time_val:.3f}s")
                    id_map[raw_id] = tid
                registered_tids.update(new_map.values())
                for _, row in obs.iterrows():
                    raw_id = int(row['id'])
                    tid = new_map[raw_id]
                    frame_tid_positions[tid] = (row['x'], row['y'], row['z'])
            else:
                # 判別失敗 → id_map登録済みのraw_idだけ採用
                for _, row in obs.iterrows():
                    raw_id = int(row['id'])
                    if raw_id in id_map:
                        frame_tid_positions[id_map[raw_id]] = (row['x'], row['y'], row['z'])
        elif n_obs > 0:
            # ★ 10点未満: id_map登録済みのraw_idだけTID変換して採用 (補完しない)
            for _, row in obs.iterrows():
                raw_id = int(row['id'])
                if raw_id in id_map:
                    frame_tid_positions[id_map[raw_id]] = (row['x'], row['y'], row['z'])
        # n_obs == 0 → 何も採用しない

        # --- 出力行の確定 & オクルージョン(欠損)検出 ---
        for tid in template_ids:
            if tid in frame_tid_positions:
                x, y, z = frame_tid_positions[tid]
                output_rows.append((frame, time_val, tid, x, y, z))

                run = missing_run[tid]
                if run['len'] >= min_gap:
                    occlusion_events.append({
                        'template_id': tid,
                        'start_frame': run['start_frame'], 'end_frame': frame,
                        'start_time': run['start_time'], 'end_time': time_val,
                        'n_frames': run['len'],
                    })
                run['len'] = 0
                run['start_time'] = None
                run['start_frame'] = None
            else:
                # 一度も識別されていないTIDは「立ち上げ待ち」であり遮蔽とはみなさない
                if tid in registered_tids:
                    run = missing_run[tid]
                    if run['len'] == 0:
                        run['start_time'] = time_val
                        run['start_frame'] = frame
                    run['len'] += 1
                # 欠損として出力しない(補間しない)

    # ループ終了時点でまだ続いている欠損runも記録する
    last_frame = unique_frames[-1] if unique_frames else None
    for tid, run in missing_run.items():
        if run['len'] >= min_gap:
            occlusion_events.append({
                'template_id': tid,
                'start_frame': run['start_frame'], 'end_frame': last_frame,
                'start_time': run['start_time'], 'end_time': None,
                'n_frames': run['len'],
            })

    label_by_tid = {v: k for k, v in template_ids_map.items()}
    if occlusion_events:
        print(f"\n  [遮蔽検出] {len(occlusion_events)} 件 "
              f"(device_worn={device_worn}, しきい値={min_gap}フレーム)")
        for ev in occlusion_events:
            label = label_by_tid.get(ev['template_id'], ev['template_id'])
            print(f"    {label}: frame {ev['start_frame']}〜{ev['end_frame']}"
                  f"  ({ev['n_frames']}フレーム, t={ev['start_time']:.3f}s〜)")
        df_occlusion = pd.DataFrame(occlusion_events)
        df_occlusion['marker_label'] = df_occlusion['template_id'].map(label_by_tid)
        occlusion_log_path.parent.mkdir(parents=True, exist_ok=True)
        df_occlusion.to_csv(occlusion_log_path, index=False)
        print(f"  -> 遮蔽ログを保存しました: {occlusion_log_path}")
    else:
        print("  [遮蔽検出] 該当区間なし。")

    if not output_rows:
        print("エラー: 出力データが0行になりました。")
        return None

    df_out = pd.DataFrame(output_rows, columns=['Frame', 'Time', 'id', 'x', 'y', 'z'])
    df_out = df_out.drop_duplicates(subset=['Frame', 'id'], keep='first')
    df_out = df_out.sort_values(['Frame', 'id']).reset_index(drop=True)

    # 座標系統一: 生OptiTrack (X=左右, Y=鉛直上, Z=進行前) → 統一系 (X=前, Y=横, Z=上)
    x_raw, y_raw, z_raw = df_out['x'].copy(), df_out['y'].copy(), df_out['z'].copy()
    df_out['x'] = z_raw
    df_out['y'] = x_raw
    df_out['z'] = y_raw
    print("  [座標変換] X=横,Y=上,Z=前 → X=前,Y=横,Z=上 を適用しました。")

    print(f"\n処理完了。出力行数: {len(df_out)}")
    print(f"  id_map 最終登録数: {len(id_map)} 件")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"保存完了: {output_csv_path}")

    return df_out


# =============================================================================
# Main (対話式CLI)
# =============================================================================

def main():
    print("\n=== OptiTrack Data Cleanup for Joint Markers (opti_edit_for_joint) ===")
    print("新実験(task01〜03すべて左右10点の関節マーカー方式)専用。")

    subjects = CONFIG.SUBJECTS_MULTI
    days = CONFIG.DAYS
    tasks = CONFIG.TASKS
    # task01/02 は Futto着用(遮蔽検出あり)、task03 は非着用
    device_worn_map = {'task01': True, 'task02': True, 'task03': False}

    print(f"\n被験者: {', '.join(subjects)}")
    subject = input("被験者IDを入力してください: ").strip()
    if subject not in subjects:
        print(f"警告: '{subject}' は CONFIG.SUBJECTS_MULTI に登録されていません。処理は続行します。")

    print(f"\n計測日: {', '.join(days)}")
    day = input("計測日を入力してください: ").strip()
    if day not in days:
        print(f"エラー: day は {days} のいずれかを指定してください。")
        return

    print(f"\nタスク: {', '.join(tasks)}")
    task = input("タスク名を入力してください: ").strip().lower()
    if task not in tasks:
        print(f"エラー: task は {tasks} のいずれかを指定してください。")
        return

    device_worn = device_worn_map.get(task, False)
    process_joint_markers_new_experiment(subject, day, task, device_worn)


if __name__ == "__main__":
    main()
