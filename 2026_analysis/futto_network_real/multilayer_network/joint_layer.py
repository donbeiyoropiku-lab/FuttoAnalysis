"""
multilayer_network/joint_layer.py
==================================
実CONFIGの JOINT_CENTER_DEFS に基づいて仮想関節位置を計算し、
関節層ネットワークを構築する。

実データパス:
  C:\\FuttoAnalysis\\result\\2026\\{task}\\{speed}\\
    {task}_Phase{phase}_{speed}ms_joint_centers.csv

JOINT_CENTER_DEFS のタイプ:
  task01:
    Hip   : 'ratio_1_3_between_mids_offset_z'  markers=[16012,16014,16000,15960]
    Knee  : 'midpoint'                markers=[15956,15968]
    Ankle : 'mid_of_ratio_2_1'        markers=[15964,15948,15918,15966]

  task02:
    Hip   : 'midpoint_offset_xz'      markers=[57960, 57958]
    Knee  : 'midpoint'                markers=[57948,57952]
    Ankle : 'midpoint'                markers=[57950,57954]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG


JOINT_NAMES = ['Hip', 'Knee', 'Ankle']   # 解析対象の3関節
T_GAIT = 101                              # 歩行周期点数（0〜100%）


# =============================================================================
# 関節座標計算（JOINT_CENTER_DEFS の各タイプ）
# =============================================================================

def _calc_joint_center(
    joint_def: dict,
    marker_positions: dict[int, np.ndarray],   # id → (T, 3) or (3,)
    t: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    1フレーム（または全時系列）の関節中心座標を計算する。

    marker_positions : {marker_id: ndarray shape (T,3) or (3,)}
    t                : None ならバッチ処理（全時刻）、int なら単時刻

    Returns
    -------
    ndarray shape (3,) 単時刻 or (T,3) バッチ
    """
    m_type  = joint_def['type']
    ids     = joint_def['markers']

    def _get(mid):
        pos = marker_positions.get(mid)
        if pos is None:
            return None
        if t is not None:
            return pos[t] if pos.ndim == 2 else pos
        return pos   # shape (T,3)

    coords = [_get(mid) for mid in ids]
    if any(c is None for c in coords):
        return None

    # ── midpoint ──────────────────────────────────────────────
    if m_type == 'midpoint':
        return (coords[0] + coords[1]) / 2.0

    # ── midpoint_offset_x ─────────────────────────────────────
    elif m_type == 'midpoint_offset_x':
        mid = (coords[0] + coords[1]) / 2.0
        offset_x = float(joint_def.get('offset_x', 0.0))
        if mid.ndim == 2:
            mid[:, 0] += offset_x
        else:
            mid[0] += offset_x
        return mid

    # ── midpoint_offset_xz ────────────────────────────────────
    elif m_type == 'midpoint_offset_xz':
        mid = (coords[0] + coords[1]) / 2.0
        offset_x = float(joint_def.get('offset_x', 0.0))
        offset_z = float(joint_def.get('offset_z', 0.0))
        if mid.ndim == 2:
            mid[:, 0] += offset_x
            mid[:, 2] += offset_z
        else:
            mid[0] += offset_x
            mid[2] += offset_z
        return mid

    # ── ratio_1_3_between_mids ────────────────────────────────
    # Mid(0,1) と Mid(2,3) を 1:3 に内分
    # → 結果 = Mid01 * (3/4) + Mid23 * (1/4)
    elif m_type == 'ratio_1_3_between_mids':
        mid01 = (coords[0] + coords[1]) / 2.0
        mid23 = (coords[2] + coords[3]) / 2.0
        return mid01 * 0.75 + mid23 * 0.25
        
    # ── ratio_1_3_between_mids_offset_z ───────────────────────
    elif m_type == 'ratio_1_3_between_mids_offset_z':
        mid01 = (coords[0] + coords[1]) / 2.0
        mid23 = (coords[2] + coords[3]) / 2.0
        pt = mid01 * 0.75 + mid23 * 0.25
        offset_z = float(joint_def.get('offset_z', 0.0))
        if pt.ndim == 2:
            pt[:, 2] += offset_z
        else:
            pt[2] += offset_z
        return pt

    # ── mid_of_ratio_2_1 ──────────────────────────────────────
    # 2点ペア × 2で 2:1 内分点を求め、その中点
    # markers = [A1, B1, A2, B2]
    # pt1 = A1 * (2/3) + B1 * (1/3)
    # pt2 = A2 * (2/3) + B2 * (1/3)
    # result = (pt1 + pt2) / 2
    elif m_type == 'mid_of_ratio_2_1':
        pt1 = coords[0] * (2 / 3) + coords[1] * (1 / 3)
        pt2 = coords[2] * (2 / 3) + coords[3] * (1 / 3)
        return (pt1 + pt2) / 2.0

    # ── plane_projection ─────────────────────────────────────
    # markers = [plane_a, plane_b, plane_c, knee_a, knee_b]
    # 平面 X : plane_a, plane_b, plane_c の3点が作る平面
    # 平面 Y : (plane_a→plane_b) 方向ベクトルを含み平面Xに垂直な平面
    # Hip = 膝中点(knee_a, knee_b) から平面Yへ下ろした垂線の交点
    elif m_type == 'plane_projection':
        a, b, c, ka, kb = coords  # それぞれ shape(T,3) or (3,)
        knee_mid = (ka + kb) / 2.0

        # 平面X の法線
        v1 = b - a
        v2 = c - a
        if v1.ndim == 2:
            n_x = np.cross(v1, v2)
            n_x_norm = np.linalg.norm(n_x, axis=1, keepdims=True)
            n_x = np.where(n_x_norm > 0, n_x / n_x_norm, n_x)
        else:
            n_x = np.cross(v1, v2)
            nx_norm = np.linalg.norm(n_x)
            n_x = n_x / nx_norm if nx_norm > 0 else n_x

        # 平面Y の法線 = n_x × v1 (正規化済み v1)
        if v1.ndim == 2:
            v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
            v1_unit = np.where(v1_norm > 0, v1 / v1_norm, v1)
            n_y = np.cross(n_x, v1_unit)
            n_y_norm = np.linalg.norm(n_y, axis=1, keepdims=True)
            n_y = np.where(n_y_norm > 0, n_y / n_y_norm, n_y)
        else:
            v1_norm = np.linalg.norm(v1)
            v1_unit = v1 / v1_norm if v1_norm > 0 else v1
            n_y = np.cross(n_x, v1_unit)
            ny_norm = np.linalg.norm(n_y)
            n_y = n_y / ny_norm if ny_norm > 0 else n_y

        # knee_mid から平面Y（点 a を通る）への射影
        diff = knee_mid - a
        if diff.ndim == 2:
            dist = (diff * n_y).sum(axis=1, keepdims=True)
        else:
            dist = np.dot(diff, n_y)
        return knee_mid - dist * n_y

    # フォールバック: 重心
    return np.mean(coords, axis=0)


# =============================================================================
# マーカー CSVの読み込み
# =============================================================================

def load_marker_csv(
    csv_path: str | Path,
) -> Optional[dict[int, np.ndarray]]:
    """
    マーカーCSV（gait_cycle_%, id, x, y, z 形式）を読み込み、
    {marker_id: ndarray shape (101, 3)} の辞書を返す。
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {'gait_cycle_%', 'id', 'x', 'y', 'z'}
    if not required.issubset(df.columns):
        print(f"[警告] マーカーCSV の列が不正です: {path}")
        return None

    t_new = np.arange(0, 101, dtype=float)
    marker_positions: dict[int, np.ndarray] = {}

    for mid, grp in df.groupby('id'):
        grp = grp.sort_values('gait_cycle_%')
        gc  = grp['gait_cycle_%'].values
        xyz = grp[['x', 'y', 'z']].values.astype(float)

        if len(gc) != 101:
            from scipy.interpolate import interp1d
            arr = np.zeros((101, 3))
            for dim in range(3):
                f = interp1d(gc, xyz[:, dim], kind='linear',
                             fill_value=(xyz[0, dim], xyz[-1, dim]),
                             bounds_error=False)
                arr[:, dim] = f(t_new)
            xyz = arr

        marker_positions[int(mid)] = xyz

    return marker_positions


def load_joint_centers_csv(csv_path: str | Path) -> Optional[dict[str, np.ndarray]]:
    """
    仮想関節データCSVを読み込む。

    想定フォーマット:
      gait_cycle_%, joint, x, y, z
    または
      gait_cycle_%, Hip_x, Hip_y, Hip_z, Knee_x, ...

    Returns
    -------
    dict: joint_name → ndarray shape (101, 3)
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    df = pd.read_csv(path)
    t_new = np.arange(0, 101, dtype=float)

    # フォーマットA: joint 列あり
    if 'joint' in df.columns and 'x' in df.columns:
        result = {}
        for jname in JOINT_NAMES:
            sub = df[df['joint'] == jname].sort_values('gait_cycle_%')
            if sub.empty:
                continue
            gc  = sub['gait_cycle_%'].values
            xyz = sub[['x', 'y', 'z']].values.astype(float)
            if len(gc) != 101:
                from scipy.interpolate import interp1d
                arr = np.zeros((101, 3))
                for dim in range(3):
                    f = interp1d(gc, xyz[:, dim], kind='linear',
                                 fill_value=(xyz[0, dim], xyz[-1, dim]),
                                 bounds_error=False)
                    arr[:, dim] = f(t_new)
                xyz = arr
            result[jname] = xyz
        return result if result else None

    # フォーマットB: Hip_x, Hip_y, Hip_z, Knee_x, ... 列
    result = {}
    for jname in JOINT_NAMES:
        cols = [f'{jname}_x', f'{jname}_y', f'{jname}_z']
        if all(c in df.columns for c in cols):
            gc  = df.get('gait_cycle_%', pd.Series(np.linspace(0, 100, len(df)))).values
            xyz = df[cols].values.astype(float)
            if len(gc) != 101:
                from scipy.interpolate import interp1d
                arr = np.zeros((101, 3))
                for dim in range(3):
                    f = interp1d(gc, xyz[:, dim], kind='linear',
                                 fill_value=(xyz[0, dim], xyz[-1, dim]),
                                 bounds_error=False)
                    arr[:, dim] = f(t_new)
                xyz = arr
            result[jname] = xyz
    return result if result else None


# =============================================================================
# 仮想関節位置の計算
# =============================================================================

def compute_joint_centers_from_markers(
    marker_positions: dict[int, np.ndarray],
    task_key: str,
) -> dict[str, np.ndarray]:
    """
    JOINT_CENTER_DEFS に従ってマーカー座標から仮想関節座標を計算する。

    Parameters
    ----------
    marker_positions : {marker_id: ndarray shape (T, 3)}
    task_key         : 'task01' or 'task02'

    Returns
    -------
    dict: joint_name → ndarray shape (T, 3)
    """
    cfg  = CFG.TASK_CONFIGS.get(task_key, {})
    defs = cfg.get('JOINT_CENTER_DEFS', {})
    if not defs:
        return {}

    joint_centers = {}
    for jname, jdef in defs.items():
        pos = _calc_joint_center(jdef, marker_positions)
        if pos is not None:
            joint_centers[jname] = pos
    return joint_centers


# =============================================================================
# 関節角度計算
# =============================================================================

def compute_joint_angles(
    joint_centers: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Hip-Knee-Ankle の3点から関節角度を計算する。

    Returns
    -------
    dict: 'Hip', 'Knee', 'Ankle' → ndarray shape (T,) [deg]
    """
    angles = {}

    def _angle_3pt(prox, mid, dist):
        """近位(prox)→関節(mid)→遠位(dist) の角度を計算する。"""
        v1 = prox - mid    # shape (T, 3)
        v2 = dist - mid
        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True)
        v1u = v1 / (n1 + 1e-12)
        v2u = v2 / (n2 + 1e-12)
        cos_ = np.clip((v1u * v2u).sum(axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(cos_))

    hip   = joint_centers.get('Hip')
    knee  = joint_centers.get('Knee')
    ankle = joint_centers.get('Ankle')

    if hip is not None and knee is not None and ankle is not None:
        angles['Knee']  = _angle_3pt(hip, knee, ankle)

    # Hip 角（垂直方向を参照ベクトルとして近似）
    if hip is not None and knee is not None:
        T = len(hip)
        vertical = np.tile([0, 0, -1], (T, 1)).astype(float)  # Z軸負方向 = 鉛直下向き
        thigh    = knee - hip
        n        = np.linalg.norm(thigh, axis=1, keepdims=True)
        thigh_u  = thigh / (n + 1e-12)
        cos_     = np.clip((thigh_u * vertical).sum(axis=1), -1.0, 1.0)
        angles['Hip'] = np.degrees(np.arccos(cos_))

    # Ankle 角（シャンク方向を参照）
    if knee is not None and ankle is not None:
        T = len(knee)
        vertical = np.tile([0, 0, -1], (T, 1)).astype(float)  # Z軸負方向 = 鉛直下向き
        shank    = ankle - knee
        n        = np.linalg.norm(shank, axis=1, keepdims=True)
        shank_u  = shank / (n + 1e-12)
        cos_     = np.clip((shank_u * vertical).sum(axis=1), -1.0, 1.0)
        angles['Ankle'] = np.degrees(np.arccos(cos_))

    return angles


# =============================================================================
# 関節層ネットワーク
# =============================================================================

@dataclass
class JointNetworkResult:
    task_key         : str
    phase            : int
    speed            : str
    joint_angles     : dict[str, np.ndarray]    # shape (T,) per joint
    coupling_matrix  : np.ndarray               # shape (3, 3) 角度相関
    coupling_sliding : np.ndarray               # shape (T, 3, 3)
    interlimb_proxy  : float                    # Hip–Knee 結合強度の代理指標
    coord_index      : np.ndarray               # shape (3,) 各関節の平均結合強度


def compute_joint_network(
    joint_angles: dict[str, np.ndarray],
    task_key: str,
    phase: int,
    speed: str,
    window: int = 20,
    threshold: float = 0.3,
) -> JointNetworkResult:
    """
    3関節角度の相互相関から関節層ネットワークを構築する。
    """
    jnames = [j for j in ['Hip', 'Knee', 'Ankle'] if j in joint_angles]
    J = len(jnames)
    T = 101

    if J == 0:
        return JointNetworkResult(
            task_key=task_key, phase=phase, speed=speed,
            joint_angles={}, coupling_matrix=np.zeros((3, 3)),
            coupling_sliding=np.zeros((T, 3, 3)),
            interlimb_proxy=0.0, coord_index=np.zeros(3),
        )

    angle_mat = np.column_stack([joint_angles[j] for j in jnames])  # (T, J)

    # 全周期相関
    corr = np.corrcoef(angle_mat.T)
    np.fill_diagonal(corr, 0)
    adj  = np.where(np.abs(corr) >= threshold, corr, 0.0)
    np.fill_diagonal(adj, 0)

    # 時変相関
    coup_slid = np.zeros((T, J, J))
    for t in range(T):
        ts, te = max(0, t - window // 2), min(T, t + window // 2 + 1)
        seg = angle_mat[ts:te, :]
        if seg.shape[0] < 3 or np.any(seg.std(axis=0) < 1e-10):
            coup_slid[t] = np.eye(J)
        else:
            c = np.corrcoef(seg.T)
            np.fill_diagonal(c, 0)
            coup_slid[t] = np.where(np.abs(c) >= threshold, c, 0.0)

    coord_index = np.abs(adj).sum(axis=1) / max(J - 1, 1)

    # Hip–Knee 結合 (interlimb proxy)
    il_proxy = 0.0
    if 'Hip' in jnames and 'Knee' in jnames:
        hi = jnames.index('Hip')
        ki = jnames.index('Knee')
        il_proxy = float(abs(adj[hi, ki]))

    # 3×3 に揃える（J < 3 の場合はゼロパディング）
    full_adj  = np.zeros((3, 3))
    full_slid = np.zeros((T, 3, 3))
    full_coord= np.zeros(3)
    for ii, jn in enumerate(jnames):
        fi = ['Hip', 'Knee', 'Ankle'].index(jn)
        full_adj[fi, :len(jnames)] = adj[ii]
        full_adj[:len(jnames), fi] = adj[:, ii]
        full_slid[:, fi, :len(jnames)] = coup_slid[:, ii, :]
        full_slid[:, :len(jnames), fi] = coup_slid[:, :, ii]
        full_coord[fi] = coord_index[ii]

    return JointNetworkResult(
        task_key        = task_key,
        phase           = phase,
        speed           = speed,
        joint_angles    = joint_angles,
        coupling_matrix = full_adj,
        coupling_sliding= full_slid,
        interlimb_proxy = il_proxy,
        coord_index     = full_coord,
    )


def build_joint_centers_csv_path(
    task_key: str,
    phase: int,
    speed: str,
    result_dir: str = r"C:\FuttoAnalysis\result",
    year: str = "2026",
) -> Path:
    """
    仮想関節データCSVのパスを生成する。
    例: C:\\FuttoAnalysis\\result\\2026\\task01\\0.7\\
          task01_Phase1_0.7ms_joint_centers.csv
    """
    return (
        Path(result_dir) / year / task_key / speed
        / f"{task_key}_Phase{phase}_{speed}ms_joint_centers.csv"
    )


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("=== Joint Layer テスト (シミュレーション) ===\n")
    for tk in ['task01', 'task02']:
        T = 101
        t = np.linspace(0, 2 * np.pi, T)
        dummy_joint_angles = {
            'Hip'  : 20 * np.sin(t) + 5 * np.random.randn(T),
            'Knee' : 40 * np.abs(np.sin(t)) + 5 * np.random.randn(T),
            'Ankle': 15 * np.sin(t + 0.5) + 3 * np.random.randn(T),
        }
        jn = compute_joint_network(dummy_joint_angles, tk, phase=3, speed='1.1')
        print(f"{tk}:")
        print(f"  Coord Index     : {jn.coord_index.round(3)}")
        print(f"  Interlimb Proxy : {jn.interlimb_proxy:.4f}")
        print(f"  Coupling Matrix :\n{jn.coupling_matrix.round(3)}")
        print()
