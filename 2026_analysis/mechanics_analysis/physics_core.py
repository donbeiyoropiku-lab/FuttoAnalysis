# =============================================================================
# mechanics_analysis/physics_core.py
#
# 役割:
#   全解析モジュールが共通して使う物理量計算を集約する。
#   描画・入出力には一切依存しない純粋な計算関数群。
#
# 収録関数:
#   unify_coordinate_system()   座標系統一 (昨年度そのまま移植)
#   calc_joint_center()         関節中心座標算出
#   calc_all_torques()          全関節トルク計算 (昨年度移植・今年度対応)
#   calc_work_data()            ゴム仕事量計算 (昨年度移植)
#   calc_net_shank_force()      下腿合力ベクトル計算 (昨年度移植)
#   calc_joint_angles()         関節角度計算 (今年度新規追加)
#   calc_frame_physics()        フレームごと一括計算 (3D可視化用)
# =============================================================================

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# 張力データの形式正規化
# ---------------------------------------------------------------------------

def normalize_tension_df(df_tension: pd.DataFrame) -> pd.DataFrame:
    """
    張力CSVをロング形式に統一する。

    strength_visualize が出力するワイド形式:
        gait_cycle_%, Front_Upper_In, Back_Upper_In, ...
    各関数が期待するロング形式:
        gait_cycle_%, segment, tension_N

    どちらの形式で渡されても正しく動作する。
    """
    # すでにロング形式の場合はそのまま返す
    if 'segment' in df_tension.columns and 'tension_N' in df_tension.columns:
        return df_tension

    # ワイド形式 → ロング形式に変換
    seg_cols = [c for c in df_tension.columns if c != 'gait_cycle_%']
    df_long = df_tension.melt(
        id_vars='gait_cycle_%',
        value_vars=seg_cols,
        var_name='segment',
        value_name='tension_N',
    )
    return df_long



# ---------------------------------------------------------------------------
# 座標系統一 (昨年度 calculate_rubber_torque.py より移植・改名のみ)
# ---------------------------------------------------------------------------

def unify_coordinate_system(df_mean_cycle: pd.DataFrame,
                             task_key: str) -> pd.DataFrame | None:
    """
    OptiTrack 出力座標を統一座標系に揃える。

    2026年度統一座標系:
        X : 進行方向 (前が正)
        Y : 左右方向 (右→左 が正)
        Z : 床垂直上方向 (上が正)

    task01/02/03 いずれも前処理スクリプト (opti_edit*.py) で
    統一座標系に変換済みの corrected CSV が生成されるため、ここでの変換は不要。

    Returns
    -------
    変換済み DataFrame または失敗時 None。
    """
    if 'x' not in df_mean_cycle.columns:
        print("エラー: 入力データに 'x' 列がありません。")
        return None

    print(f"  -> '{task_key}': 座標系は統一済み (X=進行正, Y=右->左, Z=上方)。")
    return df_mean_cycle.copy()



def calc_joint_center(joint_def: dict,
                      positions: dict) -> np.ndarray:
    """
    1フレーム分の関節中心座標を返す。

    昨年度との差分:
        'ratio_1_3_between_mids' / 'mid_of_ratio_2_1' / 'plane_projection'
        の各タイプを追加 (joint_calc.py と同ロジック)。

    Returns
    -------
    np.ndarray([x, y, z]) または np.full(3, np.nan)
    """
    j_type = joint_def.get('type')
    m_ids  = joint_def.get('markers', [])

    pts = [np.array(positions[m], dtype=float) for m in m_ids if m in positions]
    if len(pts) != len(m_ids):
        return np.full(3, np.nan)

    if j_type == 'single':
        return pts[0]
    elif j_type in ('midpoint', 'centroid'):
        return np.mean(pts, axis=0)
    elif j_type == 'midpoint_offset_x' and len(pts) >= 2:
        mid = (pts[0] + pts[1]) / 2.0
        mid[0] += float(joint_def.get('offset_x', 0.0))
        return mid
    elif j_type == 'midpoint_offset_xz' and len(pts) >= 2:
        mid = (pts[0] + pts[1]) / 2.0
        mid[0] += float(joint_def.get('offset_x', 0.0))
        mid[2] += float(joint_def.get('offset_z', 0.0))
        return mid
    elif j_type == 'weighted_midpoint':
        w = joint_def.get('weight', 0.5)
        return pts[0] * (1 - w) + pts[1] * w
    elif j_type == 'ratio_1_3_between_mids' and len(pts) >= 4:
        mid1 = (pts[0] + pts[1]) / 2.0
        mid2 = (pts[2] + pts[3]) / 2.0
        return mid1 * 0.25 + mid2 * 0.75
    elif j_type == 'ratio_1_3_between_mids_offset_z' and len(pts) >= 4:
        mid1 = (pts[0] + pts[1]) / 2.0
        mid2 = (pts[2] + pts[3]) / 2.0
        pt = mid1 * 0.25 + mid2 * 0.75
        pt[2] += float(joint_def.get('offset_z', 0.0))
        return pt
    elif j_type == 'mid_of_ratio_2_1' and len(pts) >= 4:
        r1 = pts[0] * (2/3) + pts[1] * (1/3)
        r2 = pts[2] * (2/3) + pts[3] * (1/3)
        return (r1 + r2) / 2.0
    elif j_type == 'plane_projection' and len(pts) >= 5:
        return _plane_projection(pts)

    return np.full(3, np.nan)


def _plane_projection(pts: list) -> np.ndarray:
    """Task02 Hip の幾何学的投影 (joint_calc.py と同ロジック)。"""
    pa, pb, pc, ka, kb = pts[0], pts[1], pts[2], pts[3], pts[4]
    n_x = np.cross(pa - pc, pb - pc)
    if np.linalg.norm(n_x) < 1e-9:
        return (ka + kb) / 2.0
    n_x /= np.linalg.norm(n_x)
    v = pb - pa
    if np.linalg.norm(v) < 1e-9:
        return (ka + kb) / 2.0
    v /= np.linalg.norm(v)
    n_y = np.cross(n_x, v)
    if np.linalg.norm(n_y) < 1e-9:
        return (ka + kb) / 2.0
    n_y /= np.linalg.norm(n_y)
    knee_pt = (ka + kb) / 2.0
    t = np.dot(pa - knee_pt, n_y)
    return knee_pt + t * n_y


# ---------------------------------------------------------------------------
# トルク計算 (昨年度 calculate_all_torques より移植・今年度 CONFIG に対応)
# ---------------------------------------------------------------------------

def calc_all_torques(df_mean_cycle: pd.DataFrame,
                     df_tension: pd.DataFrame,
                     cfg: dict) -> pd.DataFrame | None:
    """
    各関節 (Hip/Knee/Ankle) にかかるゴムのトルクを計算する。

    今年度変更点:
        昨年度は RUBBER_TORQUE_MAP を別途 CONFIG に定義していたが、
        今年度は LINES_TO_DRAW と SEGMENTS から自動的に関節-セグメント対応を導出する。
        (LINES_TO_DRAW が空の task03 は空の DataFrame を返す)

    Returns
    -------
    pd.DataFrame with columns:
        gait_cycle_%, segment, joint, torque_x_Nm, torque_y_Nm, torque_z_Nm
    """
    df_tension        = normalize_tension_df(df_tension)
    lines_def        = cfg.get('LINES_TO_DRAW', {})
    joint_center_defs = cfg.get('JOINT_CENTER_DEFS', {})
    segments         = cfg.get('SEGMENTS', {})

    if not lines_def:
        print("  -> LINES_TO_DRAW が空のため、トルク計算をスキップします (task03等)。")
        return pd.DataFrame()

    # --- マーカーID → 帰属関節 の辞書を構築 ---
    torque_line_joints   = cfg.get('TORQUE_LINE_JOINTS', {})
    torque_marker_joints: dict[int, str] = cfg.get('TORQUE_MARKER_JOINTS', {})

    if not torque_line_joints and not torque_marker_joints:
        # フォールバック: SEGMENTS から自動生成
        marker_to_segment: dict[int, str] = {}
        for seg_name, ids in segments.items():
            for mid in ids:
                marker_to_segment[mid] = seg_name

        def _proximal_joint(seg_name: str) -> str | None:
            mapping = {'Foot': 'Ankle', 'Shank': 'Knee',
                       'Thigh': 'Hip', 'Hip': 'Hip'}
            return mapping.get(seg_name)

        torque_marker_joints = {
            mid: _proximal_joint(seg)
            for mid, seg in marker_to_segment.items()
            if _proximal_joint(seg) is not None
        }

    try:
        tension_pivot = df_tension.pivot(
            index='gait_cycle_%', columns='segment', values='tension_N'
        )
    except Exception as e:
        print(f"張力データのピボット失敗: {e}")
        return None

    torque_records = []
    grouped = df_mean_cycle.groupby(
        df_mean_cycle['gait_cycle_%'].apply(lambda x: round(x, 5))
    )

    for cycle_pct in sorted(df_mean_cycle['gait_cycle_%'].unique()):
        try:
            frame_df = grouped.get_group(round(cycle_pct, 5))
        except KeyError:
            continue

        positions = {
            int(row['id']): np.array([row['x'], row['y'], row['z']], dtype=float)
            for _, row in frame_df.iterrows()
        }
        joint_centers = {
            jname: calc_joint_center(jdef, positions)
            for jname, jdef in joint_center_defs.items()
        }

        for seg_name, (p1_id, p2_id) in lines_def.items():
            # このゴムに張力があるか
            if seg_name not in tension_pivot.columns:
                continue
            try:
                F_mag = tension_pivot.loc[cycle_pct, seg_name]
            except KeyError:
                continue
            if pd.isna(F_mag) or F_mag <= 0:
                continue

            p1 = positions.get(p1_id)
            p2 = positions.get(p2_id)
            if p1 is None or p2 is None:
                continue

            # このゴムがどの関節に寄与するかを判定 (Line定義優先、なければMarker定義)
            assigned_joint = torque_line_joints.get(seg_name)

            if assigned_joint:
                jc = joint_centers.get(assigned_joint)
                if jc is None or np.isnan(jc).any():
                    continue
                
                p1_pos = positions.get(p1_id)
                p2_pos = positions.get(p2_id)
                if p1_pos is None or p2_pos is None:
                    continue
                
                # ★ ゼロキャンセル防止 ★
                # ゴムの両端の力を合算すると内力として相殺されギザギザの波形になります。
                # 関節トルクは「遠位(Z座標が低い下側)のセグメント」に働く力のみで評価します。
                if p1_pos[2] < p2_pos[2]:
                    p_attach, p_origin = p1_pos, p2_pos
                else:
                    p_attach, p_origin = p2_pos, p1_pos
                
                vec = p_origin - p_attach
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    continue
                
                F_vec   = (vec / norm) * F_mag
                r_vec   = p_attach - jc
                tau_Nmm = np.cross(r_vec, F_vec)
                tau_Nm  = tau_Nmm / 1000.0

                torque_records.append({
                    'gait_cycle_%': cycle_pct,
                    'segment':      seg_name,
                    'joint':        assigned_joint,
                    'torque_y_Nm':  tau_Nm[1],
                    'torque_x_Nm':  tau_Nm[0],
                    'torque_z_Nm':  tau_Nm[2],
                })
            else:
                # フォールバック (旧: マーカー単位での帰属)
                for attach_id, origin_id in [(p1_id, p2_id), (p2_id, p1_id)]:
                    joint_name = torque_marker_joints.get(attach_id)
                    if joint_name not in joint_centers:
                        continue
                    jc = joint_centers[joint_name]
                    if np.isnan(jc).any():
                        continue

                    p_attach = positions[attach_id]
                    p_origin = positions[origin_id]
                    vec = p_origin - p_attach
                    norm = np.linalg.norm(vec)
                    if norm < 1e-6:
                        continue

                    F_vec    = (vec / norm) * F_mag
                    r_vec    = p_attach - jc
                    tau_Nmm  = np.cross(r_vec, F_vec)
                    tau_Nm   = tau_Nmm / 1000.0

                    torque_records.append({
                        'gait_cycle_%': cycle_pct,
                        'segment':      seg_name,
                        'joint':        joint_name,
                        'torque_y_Nm':  tau_Nm[1],
                        'torque_x_Nm':  tau_Nm[0],
                        'torque_z_Nm':  tau_Nm[2],
                    })

    if not torque_records:
        print("警告: トルク計算結果が0件です。CONFIG定義を確認してください。")
        return pd.DataFrame()

    print(f"  -> {len(torque_records)} レコードのトルクデータを算出しました。")
    return pd.DataFrame(torque_records)


# ---------------------------------------------------------------------------
# 仕事量計算 (昨年度 calculate_work_data より移植・変更なし)
# ---------------------------------------------------------------------------

def calc_work_data(df_mean_cycle: pd.DataFrame,
                   df_tension: pd.DataFrame,
                   cfg: dict,
                   speed: str = '',
                   smooth_window: int = 9,
                   smooth_poly: int = 3) -> tuple:
    """
    各ゴムセグメントの仕事量 (総量・時系列) を計算する。

    変更点:
        ① Savitzky-Golay フィルタで長さデータを平滑化してから dL を計算。
           → Instantaneous Work のギザギザを低減する。
        ② 弾性エネルギー df_energy を追加出力。
           → 力場マップのエネルギー等値面に使用。
           u_i ≈ T_i × ε_i / 2  (ε_i = 自然長からのひずみ[m])
           NATURAL_LENGTHS が CONFIG に未定義の場合は T × L / 2 で近似。

    Parameters
    ----------
    speed : str
        歩行速度文字列 ('0.7', '0.9', ...) T_cycle 計算用
    smooth_window : int
        Savitzky-Golay フィルタのウィンドウ幅 (奇数, デフォルト 9)。
        データ点数が少ない場合は自動で縮小する。
    smooth_poly : int
        Savitzky-Golay フィルタの多項式次数 (デフォルト 3)。

    Returns
    -------
    (df_summary, df_instant, df_cumulative, df_energy)
    または (None, None, None, None, None)
    """
    from scipy.signal import savgol_filter

    df_tension = normalize_tension_df(df_tension)
    lines_def  = cfg.get('LINES_TO_DRAW', {})
    nat_lengths = cfg.get('NATURAL_LENGTHS', {})   # mm単位

    if not lines_def:
        print("  -> LINES_TO_DRAW が空のため仕事量計算をスキップします。")
        return None, None, None, None, None

    try:
        tension_pivot = df_tension.pivot(
            index='gait_cycle_%', columns='segment', values='tension_N'
        )
    except Exception as e:
        print(f"張力ピボット失敗: {e}")
        return None, None, None, None, None

    marker_pivot = df_mean_cycle.pivot(
        index='gait_cycle_%', columns='id', values=['x', 'y', 'z']
    )

    steps = sorted(marker_pivot.index.unique())
    if len(steps) < 2:
        print("エラー: データのステップが少なすぎます。")
        return None, None, None, None, None

    # T_cycle の推定 (calc_joint_power と同じロジック)
    try:
        v = float(speed)
        # stride_length = 1.3 * v
        # T_cycle = stride_length / v
        # 上記は T_cycle = 1.3s で速度によらず一定になるため、より単純な経験則を採用
        # 健常者の快適歩行速度(1.2-1.4m/s)で歩行周期が約1.0sになるように調整
        T_cycle = 1.2 / v if v > 0 else 1.0
    except (ValueError, TypeError):
        T_cycle = 1.0   # 不明時は 1.0s（相対比較用）
        if speed:
            print(f"  -> T_cycle の推定に失敗したため 1.0s と仮定 (speed='{speed}')")

    # ウィンドウ幅をデータ点数に合わせて調整 (奇数・poly+2以上)
    wlen = min(smooth_window, len(steps))
    if wlen % 2 == 0:
        wlen -= 1
    wlen = max(wlen, smooth_poly + 2 if (smooth_poly + 2) % 2 == 1 else smooth_poly + 3)

    t_idx = steps
    df_instant    = pd.DataFrame(index=t_idx)
    df_cumulative = pd.DataFrame(index=t_idx)
    df_energy     = pd.DataFrame(index=steps)   # 全フレーム (力場マップ用)
    summary = []

    for seg_name, (p1, p2) in lines_def.items():
        try:
            p1_c = np.vstack([marker_pivot[('x', p1)].values,
                               marker_pivot[('y', p1)].values,
                               marker_pivot[('z', p1)].values]).T
            p2_c = np.vstack([marker_pivot[('x', p2)].values,
                               marker_pivot[('y', p2)].values,
                               marker_pivot[('z', p2)].values]).T
            lengths_mm = np.linalg.norm(p1_c - p2_c, axis=1)   # [mm]

            # ① Savitzky-Golay 平滑化 → dL のギザギザ低減
            try:
                lengths_smooth = savgol_filter(lengths_mm, wlen, smooth_poly)
            except Exception:
                lengths_smooth = lengths_mm   # フォールバック

            lengths_m = lengths_smooth / 1000.0   # [m]
            tension   = tension_pivot[seg_name].values
            # dL/dt [m/s] を計算
            d_pct = np.gradient(steps) # %/frame
            dt_d_pct = T_cycle / 100.0 # s/%
            dt = d_pct * dt_d_pct # s/frame
            dL_dt = np.gradient(lengths_m) / dt
            
            # P [W] = F * v = tension * dL/dt
            power_W = tension * dL_dt
            dW = -power_W * dt  # dW = -F・dL
            
            # dL < 0 (ゴムが縮む) -> dW > 0 (正の仕事)
            # dL > 0 (ゴムが伸びる) -> dW < 0 (負の仕事)
            pos_work = np.sum(dW[dW > 0])
            neg_work = np.sum(dW[dW < 0])
            summary.append({
                'segment':          seg_name,
                'positive_work_J':  pos_work,
                'negative_work_J':  neg_work, # neg_work is already negative
                'net_work_J':       pos_work + neg_work,
            })
            df_instant[seg_name]    = power_W
            df_cumulative[seg_name] = np.cumsum(-power_W * dt)

            # ② 弾性エネルギー u_i ≈ T × ε / 2
            #    ε [m] = (length - natural_length) / 1000
            #    NATURAL_LENGTHS が未定義の場合は ε ≈ length として近似
            nat_mm = nat_lengths.get(seg_name, 0.0)   # 0 なら絶対長で近似
            strain_m = np.maximum(0.0, (lengths_mm - nat_mm) / 1000.0)
            elastic_energy = tension * strain_m / 2.0   # [J]
            df_energy[seg_name] = elastic_energy

        except KeyError:
            print(f"警告: '{seg_name}' をスキップ (マーカーID {p1} or {p2} が存在しない)。")
        except Exception as e:
            print(f"警告: '{seg_name}' の計算エラー: {e}")

    if not summary:
        print("エラー: 仕事量計算結果が0件です。")
        return None, None, None, None, None

    return pd.DataFrame(summary), df_instant, df_cumulative, df_energy, T_cycle


# ---------------------------------------------------------------------------
# 下腿合力ベクトル (昨年度 calculate_net_force より移植・変更なし)
# ---------------------------------------------------------------------------

def calc_net_shank_force(df_mean_cycle: pd.DataFrame,
                          df_tension: pd.DataFrame,
                          cfg: dict) -> tuple:
    """
    Shank に関連するゴムの合力ベクトルを計算する。

    Returns
    -------
    (net_force: np.ndarray shape(N,3), cycles: np.ndarray) または (None, None)
    """
    df_tension = normalize_tension_df(df_tension)
    lines_def = cfg.get('LINES_TO_DRAW', {})
    segments  = cfg.get('SEGMENTS', {})
    shank_ids = set(segments.get('Shank', []))

    if not lines_def or not shank_ids:
        print("  -> LINES_TO_DRAW または Shank セグメントが空です。")
        return None, None

    rubbers = []
    for name, (p1, p2) in lines_def.items():
        p1_in = p1 in shank_ids
        p2_in = p2 in shank_ids
        if p1_in ^ p2_in:
            origin = p1 if p1_in else p2
            target = p2 if p1_in else p1
            rubbers.append({'name': name, 'oid': origin, 'tid': target})

    if not rubbers:
        print("Shank に関連するゴムが見つかりません。")
        return None, None

    try:
        coord_pivot = df_mean_cycle.pivot(
            index='gait_cycle_%', columns='id', values=['x', 'y', 'z']
        )
        coord_pivot.columns = [f"{col[1]}_{col[0]}" for col in coord_pivot.columns]
        tension_pivot = df_tension.pivot(
            index='gait_cycle_%', columns='segment', values='tension_N'
        )
    except Exception as e:
        print(f"ピボット失敗: {e}")
        return None, None

    cycles    = coord_pivot.index.values
    n         = len(cycles)
    net_force = np.zeros((n, 3))

    for item in rubbers:
        name, oid, tid = item['name'], item['oid'], item['tid']
        if name not in tension_pivot.columns:
            continue
        try:
            o_pos = coord_pivot[[f"{oid}_x", f"{oid}_y", f"{oid}_z"]].values
            t_pos = coord_pivot[[f"{tid}_x", f"{tid}_y", f"{tid}_z"]].values
        except KeyError:
            continue

        # Shank 上の origin マーカーに作用する力の向き:
        # ゴムが伸びているとき Shank を target(Knee/Foot 側) に向かって引く
        # = target - origin = t_pos - o_pos
        vec  = t_pos - o_pos
        dist = np.linalg.norm(vec, axis=1)[:, np.newaxis]
        dist[dist < 1e-6] = 1.0
        unit_vec = vec / dist
        ten = tension_pivot[name].values
        if len(ten) != n:
            ten = np.interp(np.linspace(0, 100, n),
                            np.linspace(0, 100, len(ten)), ten)
        net_force += unit_vec * ten[:, np.newaxis]

    return net_force, cycles


# ---------------------------------------------------------------------------
# 関節角度計算 (今年度新規追加)
# ---------------------------------------------------------------------------

def calc_joint_angles(df_mean_cycle: pd.DataFrame,
                      cfg: dict) -> pd.DataFrame | None:
    """
    Hip・Knee・Ankle の矢状面内屈曲伸展角度 (°) を計算する。

    算出方法:
        Hip  角度: Trunk-Hip-Knee の XZ 平面射影角度
        Knee 角度: Hip-Knee-Ankle の XZ 平面射影角度
        Ankle角度: Knee-Ankle-Toe の XZ 平面射影角度

    Returns
    -------
    pd.DataFrame with columns:
        gait_cycle_%, hip_angle_deg, knee_angle_deg, ankle_angle_deg
    """
    joint_defs = cfg.get('JOINT_CENTER_DEFS', {})
    required   = ['Hip', 'Knee', 'Ankle']
    if not all(j in joint_defs for j in required):
        print("警告: Hip/Knee/Ankle の全定義が必要です。関節角度計算をスキップします。")
        return None

    records = []
    grouped = df_mean_cycle.groupby(
        df_mean_cycle['gait_cycle_%'].apply(lambda x: round(x, 5))
    )

    for cycle_pct in sorted(df_mean_cycle['gait_cycle_%'].unique()):
        try:
            frame_df = grouped.get_group(round(cycle_pct, 5))
        except KeyError:
            continue

        positions = {
            int(row['id']): np.array([row['x'], row['y'], row['z']], dtype=float)
            for _, row in frame_df.iterrows()
        }
        jc = {
            jname: calc_joint_center(jdef, positions)
            for jname, jdef in joint_defs.items()
        }

        hip_pt   = jc.get('Hip')
        knee_pt  = jc.get('Knee')
        ankle_pt = jc.get('Ankle')

        # Toe: JOINT_CENTER_DEFS に 'Toe' があれば使う、なければ Foot セグメントの重心
        if 'Toe' in jc and not np.isnan(jc['Toe']).any():
            toe_pt = jc['Toe']
        else:
            foot_ids = cfg.get('SEGMENTS', {}).get('Foot', [])
            foot_pts = [positions[m] for m in foot_ids if m in positions]
            toe_pt   = np.mean(foot_pts, axis=0) if foot_pts else None

        def _signed_angle_xz(p_proximal, vertex, p_distal):
            """
            XZ平面 (矢状面) 内での符号付き屈曲伸展角度 (°) を返す。

            2026年度座標系 (X=進行正, Z=上方, 右手系) での定義:
                屈曲 (Flexion)  : 遠位セグメントが後方 (-X) に傾く → 正値 (+)
                伸展 (Extension): 遠位セグメントが前方 (+X) に傾く → 負値 (-)
                立位 (直立)     : 0°

            算出方法:
                v1 = p_proximal → vertex  (近位セグメント方向、例: Hip→Knee)
                v2 = vertex → p_distal    (遠位セグメント方向、例: Knee→Ankle)
                XZ 2次元外積を ZX 順 (v1z*v2x - v1x*v2z) で定義することで
                X=進行正・Z=上方の座標系において屈曲が正値となる。
            """
            if any(pt is None or np.isnan(pt).any() for pt in [p_proximal, vertex, p_distal]):
                return np.nan
            # 近位セグメントベクトル: p_proximal → vertex
            v1 = np.array([vertex[0]   - p_proximal[0], vertex[2]   - p_proximal[2]])
            # 遠位セグメントベクトル: vertex → p_distal
            v2 = np.array([p_distal[0] - vertex[0],     p_distal[2] - vertex[2]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                return np.nan
            v1n, v2n = v1 / n1, v2 / n2
            # ZX外積 (v1z*v2x - v1x*v2z): 屈曲方向が正になる符号定義
            cross2d = v1n[1] * v2n[0] - v1n[0] * v2n[1]
            dot2d   = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
            return np.degrees(np.arctan2(cross2d, dot2d))

        # 仮想体幹方向: Hip の真上 (+Z) → X=進行正座標系で正しい
        trunk_pt = hip_pt + np.array([0, 0, 200]) if hip_pt is not None else None

        records.append({
            'gait_cycle_%':      cycle_pct,
            # Hip: 体幹(上方)を近位, Knee を遠位として屈曲伸展を定義
            'hip_angle_deg':     _signed_angle_xz(trunk_pt, hip_pt,   knee_pt),
            # Knee: Hip を近位, Ankle を遠位として屈曲伸展を定義
            'knee_angle_deg':    _signed_angle_xz(hip_pt,   knee_pt,  ankle_pt),
            # Ankle: Knee を近位, Toe を遠位として背屈底屈を定義
            'ankle_angle_deg':   _signed_angle_xz(knee_pt,  ankle_pt, toe_pt),
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    print(f"  -> 関節角度を {len(df)} フレーム分算出しました。")
    return df



# ---------------------------------------------------------------------------
# 関節パワー計算 (P = τ_y × ω_y)
# ---------------------------------------------------------------------------

def calc_joint_power(df_torque: pd.DataFrame,
                     df_angles: pd.DataFrame,
                     cfg: dict,
                     speed: str = '',
                     smooth_window: int = 9,
                     smooth_poly: int = 3) -> pd.DataFrame | None:
    """
    各関節の矢状面内パワー P_y = τ_y × ω_y を計算する。

    算出方法:
        τ_y [Nm]: calc_all_torques() の torque_y_Nm を関節ごとに合算
        ω_y [rad/s]: 関節角度 θ [deg] を正規化時間で微分して実時間に換算
            ω_y = (dθ/d%) × (1/T_cycle) × (π/180)
            T_cycle = 歩行周期の実時間 [s]
                    = CONFIG.PHASES[speed]['duration'] / 歩数
                    speed が不明の場合は 1.0s と仮定（相対比較用）

    符号の定義 (右手系 X=進行正, Y=右→左, Z=上方):
        P_y > 0: τ_y > 0 かつ ω_y > 0 (伸展方向のトルクで伸展方向に動く)
        P_y < 0: 符号が逆 (トルクと運動方向が逆 = エネルギー吸収)

    Parameters
    ----------
    df_torque  : calc_all_torques() の出力
    df_angles  : calc_joint_angles() の出力
    cfg        : TASK_CONFIGS[task_key]
    speed      : 歩行速度文字列 ('0.7', '0.9', ...) T_cycle 計算用
    smooth_window : Savitzky-Golay フィルタのウィンドウ幅
    smooth_poly   : Savitzky-Golay フィルタの多項式次数

    Returns
    -------
    pd.DataFrame with columns:
        gait_cycle_%, hip_power_W, knee_power_W, ankle_power_W
    または None
    """
    from scipy.signal import savgol_filter

    if df_torque is None or df_torque.empty:
        print("警告: トルクデータがありません。パワー計算をスキップします。")
        return None
    if df_angles is None or df_angles.empty:
        print("警告: 関節角度データがありません。パワー計算をスキップします。")
        return None

    # --- τ_y: 関節ごとの torque_y_Nm を歩行周期%ごとに合算 ---
    tau_grouped = (
        df_torque
        .groupby(['gait_cycle_%', 'joint'])['torque_y_Nm']
        .sum()
        .unstack(level='joint')
    )

    # --- 共通の gait_cycle_% 軸を作成 ---
    cycles = sorted(df_angles['gait_cycle_%'].unique())
    n      = len(cycles)
    if n < 2:
        print("警告: データのフレーム数が少なすぎます。")
        return None

    # --- T_cycle の推定 ---
    # speed が与えられていれば PHASES の duration から概算
    # 例: 0.7m/s フェーズ = 60s 区間 / 区間内の推定歩数
    # 実際の歩数データがないため、歩行速度から歩幅を仮定して概算
    # v [m/s] = stride_length [m] × cadence [steps/s]
    # stride_length ≈ 1.3 × speed (経験式) → T_cycle ≈ stride_length / speed
    try:
        v = float(speed)
        stride_length = 1.3 * v   # 歩幅の経験式 [m]
        T_cycle = stride_length / v   # = 1.3 s (速度によらず一定)
        # より精度を上げるなら: T_cycle = 60s / 区間内の実歩数
    except (ValueError, TypeError):
        T_cycle = 1.0   # 不明時は 1.0s（相対比較用）
        print("  -> T_cycle が不明なため 1.0s と仮定（速度間の相対比較用）")

    print(f"  -> T_cycle = {T_cycle:.3f} s (speed={speed}m/s)")
    print(f"  -> ω = dθ/d(%) × {1/T_cycle:.3f} [rad/s per deg/percent]")

    # --- ウィンドウ幅調整 ---
    wlen = min(smooth_window, n)
    if wlen % 2 == 0:
        wlen -= 1
    wlen = max(wlen, smooth_poly + 2 if (smooth_poly + 2) % 2 == 1 else smooth_poly + 3)

    # --- 関節ごとにパワーを計算 ---
    angle_cols = {
        'Hip':   'hip_angle_deg',
        'Knee':  'knee_angle_deg',
        'Ankle': 'ankle_angle_deg',
    }
    power_cols = {
        'Hip':   'hip_power_W',
        'Knee':  'knee_power_W',
        'Ankle': 'ankle_power_W',
    }

    df_power = pd.DataFrame({'gait_cycle_%': cycles})

    for joint, angle_col in angle_cols.items():
        power_col = power_cols[joint]

        if angle_col not in df_angles.columns:
            df_power[power_col] = np.nan
            continue
        if joint not in tau_grouped.columns:
            df_power[power_col] = np.nan
            continue

        # 角度 θ [deg] を正規化時間% で微分 → [deg/%]
        theta_deg = df_angles.set_index('gait_cycle_%')[angle_col].reindex(cycles).values

        # Savitzky-Golay で平滑化してから微分
        try:
            theta_smooth = savgol_filter(theta_deg, wlen, smooth_poly)
        except Exception:
            theta_smooth = theta_deg

        # dθ/d(%) [deg/%]
        dtheta_dpct = np.gradient(theta_smooth, cycles)

        # ω_y [rad/s] = dθ/d(%) × (1/T_cycle) × (π/180) / 100
        # ※ gait_cycle_% は 0〜100 なので 1/100 が必要
        omega_y = dtheta_dpct * (np.pi / 180.0) / (T_cycle * 100.0)

        # τ_y [Nm] を同じ gait_cycle_% に合わせてリサンプル
        tau_y = tau_grouped[joint].reindex(cycles).interpolate(
            method='linear', limit_direction='both'
        ).values

        # P_y = τ_y × ω_y [W]
        power = tau_y * omega_y
        df_power[power_col] = power

    print(f"  -> パワーを {len(df_power)} フレーム分算出しました。")
    return df_power

# ---------------------------------------------------------------------------
# フレームごと一括計算 (3D可視化用・昨年度 calculate_physics_per_step より移植)
# ---------------------------------------------------------------------------

def calc_frame_physics(df_mean_cycle: pd.DataFrame,
                       df_tension: pd.DataFrame,
                       cfg: dict,
                       smoothing_sigma: float = 2.0) -> list:
    """
    3D アニメーション用に歩行周期の全フレームの物理量を一括計算する。

    Returns
    -------
    list of dict, 各要素:
        step, pos_map, joint_centers, joint_torques, rubber_states
    """
    df_tension = normalize_tension_df(df_tension)
    joint_defs = cfg.get('JOINT_CENTER_DEFS', {})
    segments   = cfg.get('SEGMENTS', {})
    lines_def  = cfg.get('LINES_TO_DRAW', {})

    foot_ids  = segments.get('Foot', [])
    shank_ids = segments.get('Shank', [])
    thigh_ids = segments.get('Thigh', [])

    try:
        marker_pivot = df_mean_cycle.pivot(
            index='gait_cycle_%', columns='id', values=['x', 'y', 'z']
        )
        marker_pivot.columns = [f"{col[1]}_{col[0]}" for col in marker_pivot.columns]
        tension_pivot = df_tension.pivot(
            index='gait_cycle_%', columns='segment', values='tension_N'
        )
    except Exception as e:
        print(f"ピボット失敗: {e}")
        return []

    steps = sorted(marker_pivot.index.unique())
    frames_data = []
    prev_lengths: dict = {}

    for i, step in enumerate(steps):
        row_coord   = marker_pivot.loc[step]
        row_tension = (tension_pivot.loc[step]
                       if step in tension_pivot.index else pd.Series(dtype=float))

        # pos_map 構築
        pos_map: dict[int, np.ndarray] = {}
        for col in marker_pivot.columns:
            mid_str, axis = col.rsplit('_', 1)
            mid = int(mid_str)
            if mid not in pos_map:
                pos_map[mid] = np.zeros(3)
            pos_map[mid][['x', 'y', 'z'].index(axis)] = row_coord[col]

        # 関節中心
        joint_centers = {
            jname: calc_joint_center(jdef, pos_map)
            for jname, jdef in joint_defs.items()
        }

        # マーカーごとの力ベクトルを積算
        marker_forces: dict[int, np.ndarray] = {
            mid: np.zeros(3) for mid in pos_map
        }
        for seg_name, (p1, p2) in lines_def.items():
            if seg_name not in row_tension:
                continue
            tension = row_tension[seg_name]
            if pd.isna(tension) or tension <= 0:
                continue
            if p1 not in pos_map or p2 not in pos_map:
                continue
            vec  = pos_map[p2] - pos_map[p1]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue
            marker_forces[p1] += (vec / norm) * tension
            marker_forces[p2] += (-vec / norm) * tension

        # 関節トルク計算
        # TORQUE_MARKER_JOINTS が定義されている場合はそれを優先し、
        # 各マーカーの力を帰属関節に対するモーメントとして積算する。
        # 未定義の場合は SEGMENTS 階層ループにフォールバック。
        joint_torques: dict[str, np.ndarray] = {
            j: np.zeros(3) for j in joint_centers
        }
        torque_line_joints_frame   = cfg.get('TORQUE_LINE_JOINTS', {})
        torque_marker_joints_frame = cfg.get('TORQUE_MARKER_JOINTS', {})

        if torque_line_joints_frame:
            # ゴム(ライン)ごとにトルクを計算して各関節に足し込む（正確なロジック）
            for seg_name, (p1, p2) in lines_def.items():
                if seg_name not in row_tension:
                    continue
                tension = float(row_tension[seg_name])
                if pd.isna(tension) or tension <= 0:
                    continue
                if p1 not in pos_map or p2 not in pos_map:
                    continue
                
                jname = torque_line_joints_frame.get(seg_name)
                if jname not in joint_centers:
                    continue
                jc = joint_centers[jname]
                if jc is None or (hasattr(jc, '__len__') and np.any(np.isnan(jc))):
                    continue
                
                p1_pos = pos_map[p1]
                p2_pos = pos_map[p2]
                
                # 3Dアニメ用も同様に遠位の力のみを加算する
                if p1_pos[2] < p2_pos[2]:
                    p_attach, p_origin = p1_pos, p2_pos
                else:
                    p_attach, p_origin = p2_pos, p1_pos
                
                vec = p_origin - p_attach
                norm = np.linalg.norm(vec)
                if norm < 1e-6:
                    continue
                
                F_vec = (vec / norm) * tension
                r_vec = p_attach - jc
                joint_torques[jname] += np.cross(r_vec, F_vec) / 1000.0

        elif torque_marker_joints_frame:
            # TORQUE_MARKER_JOINTS ベース (task01/02 で正確な帰属)
            for mid, jname in torque_marker_joints_frame.items():
                if mid not in pos_map:
                    continue
                if jname not in joint_centers:
                    continue
                jc = joint_centers[jname]
                # NaN チェック: None または NaN 配列の両方を確実に除外
                if jc is None or (hasattr(jc, '__len__') and np.any(np.isnan(jc))):
                    continue
                r = pos_map[mid] - jc
                joint_torques[jname] += np.cross(r, marker_forces[mid]) / 1000.0
        else:
            # フォールバック: SEGMENTS 階層ループ (TORQUE_MARKER_JOINTS 未定義時)
            for mid in foot_ids:
                if mid not in pos_map:
                    continue
                for jname in ['Ankle', 'Knee', 'Hip']:
                    if jname in joint_centers and not np.isnan(joint_centers[jname]).any():
                        r = pos_map[mid] - joint_centers[jname]
                        joint_torques[jname] += np.cross(r, marker_forces[mid]) / 1000.0
            for mid in shank_ids:
                if mid not in pos_map:
                    continue
                for jname in ['Knee', 'Hip']:
                    if jname in joint_centers and not np.isnan(joint_centers[jname]).any():
                        r = pos_map[mid] - joint_centers[jname]
                        joint_torques[jname] += np.cross(r, marker_forces[mid]) / 1000.0
            for mid in thigh_ids:
                if mid not in pos_map:
                    continue
                if 'Hip' in joint_centers and not np.isnan(joint_centers['Hip']).any():
                    r = pos_map[mid] - joint_centers['Hip']
                    joint_torques['Hip'] += np.cross(r, marker_forces[mid]) / 1000.0

        # ゴムの仕事量
        rubber_states: dict = {}
        for seg_name, (p1, p2) in lines_def.items():
            if p1 in pos_map and p2 in pos_map:
                curr_len = np.linalg.norm(pos_map[p1] - pos_map[p2]) / 1000.0
                tension  = float(row_tension.get(seg_name, 0))
                dL = curr_len - prev_lengths.get(seg_name, curr_len) if i > 0 else 0.0
                prev_lengths[seg_name] = curr_len
                rubber_states[seg_name] = {
                    'len': curr_len, 'ten': tension, 'dL': dL,
                    'raw_work': tension * dL,
                    'p1': pos_map[p1], 'p2': pos_map[p2],
                }

        frames_data.append({
            'step':          step,
            'pos_map':       pos_map,
            'joint_centers': joint_centers,
            'joint_torques': joint_torques,
            'rubber_states': rubber_states,
        })

    # 仕事量の平滑化
    for name in lines_def:
        vals, idxs = [], []
        for idx, frame in enumerate(frames_data):
            if name in frame['rubber_states']:
                vals.append(frame['rubber_states'][name]['raw_work'])
                idxs.append(idx)
        if not vals:
            continue
        smoothed = gaussian_filter1d(vals, sigma=smoothing_sigma, mode='wrap')
        for i_f, val in zip(idxs, smoothed):
            frames_data[i_f]['rubber_states'][name]['smoothed_work'] = val

    return frames_data