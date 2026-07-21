# =============================================================================
# strength_visualize/tension_calc.py
#
# 役割:
#   ゴム張力の計算と、筋肉マーカーの空間位置算出を担う。
#   ・計算ロジックのみを持つ純粋な関数群
#   ・描画や入出力には一切依存しない
#
# ★ 計算式を変更したい場合はここだけ修正すればよい。
# =============================================================================

import numpy as np
import pandas as pd


def calculate_all_tensions(df_mean_cycle, natural_lengths, lines_to_draw_def,
                            strain_to_force_interp, force_multiplier=1.0):
    """
    全ゴムセグメントの張力を計算する。

    Parameters
    ----------
    df_mean_cycle : pd.DataFrame
        OptiTrack平均化データ。'gait_cycle_%', 'id', 'x', 'y', 'z' 列を含む。
    natural_lengths : dict
        セグメント名 -> 自然長(mm) の辞書。
    lines_to_draw_def : dict
        セグメント名 -> (marker_id1, marker_id2) の辞書。
    strain_to_force_interp : scipy.interpolate.interp1d
        ひずみ(mm) -> 力(N) の補間関数。
    force_multiplier : float
        力の倍率（タスクごとのゴム本数補正など）。

    Returns
    -------
    tuple[dict, pd.DataFrame]
        tension_data        : セグメント名 -> numpy配列(N値の時系列)
        tension_df_for_csv  : gait_cycle_% + 各セグメント列のDataFrame
    """
    tension_data = {}
    cycle_percs = sorted(df_mean_cycle['gait_cycle_%'].unique())
    tension_df_for_csv = pd.DataFrame({'gait_cycle_%': cycle_percs})

    for line_name, (id1, id2) in lines_to_draw_def.items():
        if line_name not in natural_lengths:
            continue
        nl = natural_lengths[line_name]

        p1_df = df_mean_cycle[df_mean_cycle['id'] == id1].sort_values('gait_cycle_%')
        p2_df = df_mean_cycle[df_mean_cycle['id'] == id2].sort_values('gait_cycle_%')

        if p1_df.empty or p2_df.empty:
            continue

        p1_coords = p1_df[['x', 'y', 'z']].values
        p2_coords = p2_df[['x', 'y', 'z']].values

        distances = np.linalg.norm(p1_coords - p2_coords, axis=1)
        strains   = np.maximum(0, distances - nl)

        tensions = strain_to_force_interp(strains) * force_multiplier

        tension_data[line_name]          = tensions
        tension_df_for_csv[line_name]    = tensions

    return tension_data, tension_df_for_csv


def calculate_indicator_position(muscle_info, current_positions):
    """
    筋肉マーカーの表示位置（3D座標）を算出する。

    Parameters
    ----------
    muscle_info : dict
        CONFIG の MUSCLE_INDICATORS の各エントリ。
        'type', 'markers', 'weight', 'ref_marker' などのキーを持つ。
    current_positions : dict
        marker_id(int) -> np.array([x, y, z]) の辞書。

    Returns
    -------
    np.ndarray or None
        算出できない場合は None を返す。
    """
    m_type  = muscle_info.get('type')
    markers = muscle_info.get('markers') or muscle_info.get('ids_key', [])

    coords = [current_positions[m] for m in markers if m in current_positions]
    if not coords:
        return None

    if m_type == 'single' and len(coords) >= 1:
        return coords[0]

    elif m_type == 'midpoint' and len(coords) == 2:
        return (coords[0] + coords[1]) / 2.0

    elif m_type == 'centroid' and len(coords) >= 1:
        return np.mean(coords, axis=0)

    elif m_type == 'weighted_midpoint' and len(coords) == 2:
        w = muscle_info.get('weight', 0.5)
        return coords[0] * (1 - w) + coords[1] * w

    elif m_type == 'double_midpoint_interpolation' and len(coords) >= 4:
        x_pt = (coords[0] + coords[1]) / 2.0
        y_pt = (coords[2] + coords[3]) / 2.0
        w    = muscle_info.get('weight', 0.5)
        return x_pt * (1.0 - w) + y_pt * w

    elif m_type == 'offset' and len(coords) >= 1:
        ref_markers = muscle_info.get('ref_marker') or muscle_info.get('ref_ids_key', [])
        ref_coords  = [current_positions[m] for m in ref_markers if m in current_positions]
        centroid    = np.mean(coords, axis=0)
        if len(ref_coords) >= 2:
            vec  = ref_coords[1] - ref_coords[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec_norm   = vec / norm
                offset_dist = muscle_info.get('weight', 50.0)
                return centroid + vec_norm * offset_dist
        return centroid

    # フォールバック: 重心
    return np.mean(coords, axis=0)


def compute_segment_tension_bounds(tension_data):
    """
    各セグメントの張力の最小値・最大値を算出する（カラーバー正規化用）。

    Parameters
    ----------
    tension_data : dict
        セグメント名 -> numpy配列 の辞書。

    Returns
    -------
    dict
        セグメント名 -> {'min': float, 'max': float} の辞書。
    """
    bounds = {}
    for name, tension_array in tension_data.items():
        if len(tension_array) > 0:
            bounds[name] = {
                'min': float(np.min(tension_array)),
                'max': float(np.max(tension_array)),
            }
    return bounds