# =============================================================================
# strength_visualize/joint_calc.py
#
# 役割:
#   マーカー座標データから仮想関節中心座標を算出し、
#   歩行周期平均データとして出力する。
#
# 対応する関節定義タイプ (CONFIG の JOINT_CENTER_DEFS で指定):
#
#   'midpoint'
#       markers = [a, b]
#       → (a + b) / 2
#
#   'ratio_1_3_between_mids'  ← Task01 Hip
#       markers = [m0, m1, m2, m3]
#       mid1 = (m0 + m1) / 2
#       mid2 = (m2 + m3) / 2
#       → mid1 * (1/4) + mid2 * (3/4)   (mid1 側から 1:3 に内分)
#
#   'mid_of_ratio_2_1'        ← Task01 Ankle
#       markers = [front_a, front_b, back_a, back_b]
#       r_front = front_a * (2/3) + front_b * (1/3)
#       r_back  = back_a  * (2/3) + back_b  * (1/3)
#       → (r_front + r_back) / 2
#
#   'plane_projection'        ← Task02 Hip
#       markers = [plane_a, plane_b, plane_c, knee_a, knee_b]
#       平面X   = plane_a, plane_b, plane_c の3点が定義する平面
#       平面Y   = plane_a→plane_b の方向ベクトルを含み、平面X に垂直な平面
#       knee_pt = (knee_a + knee_b) / 2
#       → knee_pt から平面Y へ下ろした垂線の足 (交点)
#
# 出力フォーマット:
#   gait_cycle_%  Hip_x  Hip_y  Hip_z  Knee_x  Knee_y  Knee_z  ...
#   各行が歩行周期 0〜100% の1フレームに対応する。
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# 関節タイプ別の座標算出関数
# ---------------------------------------------------------------------------

def _midpoint(positions):
    """markers = [a, b] → 中点"""
    return (positions[0] + positions[1]) / 2.0


def _ratio_1_3_between_mids(positions):
    """
    markers = [m0, m1, m2, m3]
    mid1 = (m0+m1)/2,  mid2 = (m2+m3)/2
    → mid1 を起点として 1:3 に内分する点
      = mid1*(1/4) + mid2*(3/4)
    """
    mid1 = (positions[0] + positions[1]) / 2.0
    mid2 = (positions[2] + positions[3]) / 2.0
    return mid1 * 0.25 + mid2 * 0.75


def _ratio_1_3_between_mids_offset_z(positions, offset_z=0.0):
    """
    _ratio_1_3_between_mids で求めた点のZ座標を offset_z ずらす。
    """
    pt = _ratio_1_3_between_mids(positions)
    pt[2] += offset_z
    return pt


def _mid_of_ratio_2_1(positions):
    """
    markers = [front_a, front_b, back_a, back_b]
    r_front = front_a*(2/3) + front_b*(1/3)
    r_back  = back_a*(2/3)  + back_b*(1/3)
    → (r_front + r_back) / 2
    """
    r_front = positions[0] * (2.0 / 3.0) + positions[1] * (1.0 / 3.0)
    r_back  = positions[2] * (2.0 / 3.0) + positions[3] * (1.0 / 3.0)
    return (r_front + r_back) / 2.0


def _single(positions):
    """
    markers = [a]
    → マーカー座標をそのまま返す。
    task03 のように関節上に直接マーカーが置かれている場合に使用。
    """
    return positions[0]


def _plane_projection(positions):
    """
    Task02 Hip の幾何学的投影。
    markers = [plane_a, plane_b, plane_c, knee_a, knee_b]

    手順:
      1. 平面X の法線 n_x = (plane_a - plane_c) × (plane_b - plane_c)
      2. 方向ベクトル  v   = plane_b - plane_a  (正規化)
      3. 平面Y の法線 n_y = n_x × v            (正規化)
         ※ 平面Y は plane_a, plane_b を含み、平面X に垂直
      4. knee_pt = (knee_a + knee_b) / 2
      5. knee_pt から平面Y への垂線の足:
           平面Y は plane_a を通り n_y を法線とする平面
           t = dot(plane_a - knee_pt, n_y) / dot(n_y, n_y)
           交点 Z = knee_pt + t * n_y
    """
    pa, pb, pc = positions[0], positions[1], positions[2]
    ka, kb     = positions[3], positions[4]

    # 1. 平面X の法線
    v1   = pa - pc
    v2   = pb - pc
    n_x  = np.cross(v1, v2)
    norm_nx = np.linalg.norm(n_x)
    if norm_nx < 1e-9:
        # 3点が縮退している場合は knee_pt をそのまま返す（フォールバック）
        return (ka + kb) / 2.0
    n_x = n_x / norm_nx

    # 2. 方向ベクトル v (plane_a → plane_b)
    v    = pb - pa
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        return (ka + kb) / 2.0
    v = v / norm_v

    # 3. 平面Y の法線
    n_y  = np.cross(n_x, v)
    norm_ny = np.linalg.norm(n_y)
    if norm_ny < 1e-9:
        return (ka + kb) / 2.0
    n_y = n_y / norm_ny

    # 4. 膝中点
    knee_pt = (ka + kb) / 2.0

    # 5. 平面Y への垂線の足
    #    平面Y : dot(P - pa, n_y) = 0
    denom = np.dot(n_y, n_y)   # = 1.0 (正規化済み)
    t     = np.dot(pa - knee_pt, n_y) / denom
    return knee_pt + t * n_y


def _midpoint_offset_x(positions, offset_x=0.0):
    """
    markers = [a, b]
    中点 M = (a + b) / 2 を算出し、X軸方向に offset_x (mm) だけずらした点を返す。

    CONFIG での指定例:
        {'type': 'midpoint_offset_x', 'markers': [57960, 57958], 'offset_x': -30.0}

    offset_x の符号:
        正値 (+) : 進行方向前方にずらす
        負値 (-) : 進行方向後方にずらす
    """
    mid = (positions[0] + positions[1]) / 2.0
    mid[0] += offset_x
    return mid

def _midpoint_offset_xz(positions, offset_x=0.0, offset_z=0.0):
    """
    markers = [a, b]
    中点 M = (a + b) / 2 を算出し、X, Z軸方向にオフセットした点を返す。
    """
    mid = (positions[0] + positions[1]) / 2.0
    mid[0] += offset_x
    mid[2] += offset_z
    return mid


# ---------------------------------------------------------------------------
# タイプ → 関数のディスパッチテーブル
# ---------------------------------------------------------------------------
_CALC_FUNCS = {
    'single':                 _single,
    'midpoint':               _midpoint,
    'midpoint_offset_x':      _midpoint_offset_x,
    'midpoint_offset_xz':     _midpoint_offset_xz,
    'ratio_1_3_between_mids': _ratio_1_3_between_mids,
    'ratio_1_3_between_mids_offset_z': _ratio_1_3_between_mids_offset_z,
    'mid_of_ratio_2_1':       _mid_of_ratio_2_1,
    'plane_projection':       _plane_projection,
}


# ---------------------------------------------------------------------------
# フレーム単位の関節座標算出
# ---------------------------------------------------------------------------

def calc_joint_center_at_frame(joint_def, current_positions):
    """
    1フレーム分の仮想関節座標を算出する。

    Parameters
    ----------
    joint_def : dict
        CONFIG の JOINT_CENTER_DEFS の1エントリ。
        {'type': str, 'markers': [id, ...], ...}
        type が 'midpoint_offset_x' の場合は 'offset_x' キーも参照する。
    current_positions : dict
        marker_id (int) -> np.array([x, y, z])

    Returns
    -------
    np.ndarray or None
        [x, y, z] の座標。マーカーが揃わない場合は None。
    """
    j_type  = joint_def['type']
    markers = joint_def['markers']

    if j_type not in _CALC_FUNCS:
        raise ValueError(f"未定義の関節タイプ: '{j_type}'")

    positions = [current_positions.get(m) for m in markers]
    if any(p is None for p in positions):
        return None

    pts = [np.asarray(p, dtype=float) for p in positions]

    # offset_x など追加パラメータが必要なタイプはここで kwargs を組み立てる
    if j_type == 'midpoint_offset_x':
        return _CALC_FUNCS[j_type](pts, offset_x=float(joint_def.get('offset_x', 0.0)))
    if j_type == 'midpoint_offset_xz':
        return _CALC_FUNCS[j_type](pts, offset_x=float(joint_def.get('offset_x', 0.0)), offset_z=float(joint_def.get('offset_z', 0.0)))
    if j_type == 'ratio_1_3_between_mids_offset_z':
        return _CALC_FUNCS[j_type](pts, offset_z=float(joint_def.get('offset_z', 0.0)))

    return _CALC_FUNCS[j_type](pts)


# ---------------------------------------------------------------------------
# 歩行周期平均データへの適用
# ---------------------------------------------------------------------------

def calc_joint_centers(df_mean_cycle, joint_center_defs):
    """
    歩行周期平均データ全フレームにわたって仮想関節座標を算出する。

    Parameters
    ----------
    df_mean_cycle : pd.DataFrame
        OptiTrack 平均化データ。'gait_cycle_%', 'id', 'x', 'y', 'z' 列を含む。
    joint_center_defs : dict
        CONFIG の JOINT_CENTER_DEFS。
        例: {'Hip': {'type': 'midpoint', 'markers': [...]}, ...}

    Returns
    -------
    pd.DataFrame
        列: gait_cycle_%,
            Hip_x, Hip_y, Hip_z,
            Knee_x, Knee_y, Knee_z,
            Ankle_x, Ankle_y, Ankle_z  (定義されている関節のみ)
        行数 = gait_cycle_% のユニーク数
    """
    frames = sorted(df_mean_cycle['gait_cycle_%'].unique())
    records = []

    for gc_pct in frames:
        frame_df = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == gc_pct]
        current_positions = {
            int(row.id): np.array([row.x, row.y, row.z])
            for _, row in frame_df.iterrows()
        }

        row_data = {'gait_cycle_%': gc_pct}
        for joint_name, joint_def in joint_center_defs.items():
            pos = calc_joint_center_at_frame(joint_def, current_positions)
            if pos is not None:
                row_data[f'{joint_name}_x'] = pos[0]
                row_data[f'{joint_name}_y'] = pos[1]
                row_data[f'{joint_name}_z'] = pos[2]
            else:
                row_data[f'{joint_name}_x'] = np.nan
                row_data[f'{joint_name}_y'] = np.nan
                row_data[f'{joint_name}_z'] = np.nan

        records.append(row_data)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# NaN フレームの線形補間
# ---------------------------------------------------------------------------

def interpolate_missing_frames(df_joints):
    """
    NaN フレームを線形補間する。
    両端が NaN の場合は外挿せず NaN のまま残す。

    Parameters
    ----------
    df_joints : pd.DataFrame
        calc_joint_centers() の戻り値。

    Returns
    -------
    pd.DataFrame
        補間済みの DataFrame。
    """
    coord_cols = [c for c in df_joints.columns if c != 'gait_cycle_%']
    df_out = df_joints.copy()
    nan_before = df_out[coord_cols].isna().sum().sum()
    if nan_before > 0:
        df_out[coord_cols] = df_out[coord_cols].interpolate(method='linear', limit_direction='forward')
        nan_after = df_out[coord_cols].isna().sum().sum()
        if nan_before > nan_after:
            print(f"  -> {nan_before - nan_after} 個の NaN フレームを線形補間しました。")
        if nan_after > 0:
            print(f"  -> 補間後も {nan_after} 個の NaN が残っています（端点部分）。")
    return df_out


# ---------------------------------------------------------------------------
# CSV保存
# ---------------------------------------------------------------------------

def save_joint_csv(df_joints, output_path):
    """
    仮想関節座標を CSV に保存する。

    Parameters
    ----------
    df_joints : pd.DataFrame
    output_path : str or Path
    """
    import os
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    df_joints.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  -> 仮想関節データを保存しました: {output_path}")