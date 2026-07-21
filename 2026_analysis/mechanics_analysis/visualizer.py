# =============================================================================
# mechanics_analysis/visualizer.py
#
# 役割:
#   全解析結果の描画を集約する。昨年度プログラムのグラフ形式・色使いを維持しつつ
#   今年度タスク (task01/02/03) に対応する。
#
# 収録関数:
#   plot_joint_torques()      関節トルクグラフ + スティック図
#   plot_work_time_series()   仕事量時系列グラフ
#   plot_work_loops()         ワークループグラフ
#   plot_polar_force()        ポーラーチャート
#   plot_joint_angles()       関節角度時系列グラフ (新規追加)
#   plot_task_comparison()    タスク間比較グラフ (新規追加)
#   animate_force_field_3d()  3D力場アニメーション
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .physics_core import calc_joint_center, normalize_tension_df

# ---- 共通定数 (昨年度プログラムの設定値をそのまま引き継ぐ) ----
TORQUE_SCALE       = 150.0
WORK_SENSITIVITY   = 0.05
ANIMATION_INTERVAL = 100  # ms


# ===========================================================================
# 関節トルクグラフ + スティック図
# (昨年度 calculate_rubber_torque.py: plot_joint_torques / plot_stick_figure より移植)
# ===========================================================================

def plot_joint_torques(torque_df: pd.DataFrame, cfg: dict,
                       task_key: str, df_mean_cycle_unified: pd.DataFrame,
                       save_dir: str | None = None) -> None:
    """関節ごと・軸ごとにゴムトルクをグラフ化し、スティック図を追加する。"""
    if torque_df is None or torque_df.empty:
        print("トルクデータがないため、グラフを描画できません。")
        return

    total_torque_df = (
        torque_df
        .groupby(['gait_cycle_%', 'joint'])[['torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm']]
        .sum().reset_index()
    )

    defined_joints  = list(cfg.get('JOINT_CENTER_DEFS', {}).keys())
    joints_in_data  = total_torque_df['joint'].unique()
    joints          = [j for j in defined_joints if j in joints_in_data]
    if not joints:
        print("エラー: プロット対象の関節データがありません。")
        return

    num_joints    = len(joints)
    height_ratios = [3] * num_joints + [2]
    fig, axes = plt.subplots(
        num_joints + 1, 1, figsize=(12, 4 * num_joints + 3),
        sharex=False,
        gridspec_kw={'height_ratios': height_ratios}
    )
    axes_list = np.atleast_1d(axes)
    ax_torque = axes_list[:-1]
    ax_stick  = axes_list[-1]

    fig.suptitle(
        f"Total Rubber Torque at Joints (Task: {task_key})\n"
        "Coords: X=Progression(+fwd), Y=Lateral(R→L+), Z=Vertical(+up)  |  torque_y: Extension(+) / Flexion(-)",
        fontsize=16
    )

    if len(ax_torque) > 1:
        for i in range(len(ax_torque) - 1):
            ax_torque[i].sharex(ax_torque[i + 1])

    highlight_axis  = 'torque_y_Nm'
    highlight_label = 'Extension(+)/Flexion(-) Torque  [Y-axis moment]'
    other_axes = {
        'torque_x_Nm': 'X-axis moment (ref)',
        'torque_z_Nm': 'Z-axis moment (ref)',
    }

    for i, (ax, joint_name) in enumerate(zip(ax_torque, joints)):
        jdata = total_torque_df[total_torque_df['joint'] == joint_name]
        ax.plot(jdata['gait_cycle_%'], jdata[highlight_axis],
                label=highlight_label, linewidth=3.0, color='red')
        for col, label in other_axes.items():
            if col in jdata.columns:
                ax.plot(jdata['gait_cycle_%'], jdata[col],
                        label=label, linewidth=1.5, alpha=0.7, linestyle='--')
        ax.set_title(f"Joint: {joint_name}")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlim(0, 100)
        if i < len(ax_torque) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax_torque[-1].set_xlabel("Gait Cycle (%)")
    _plot_stick_figure(ax_stick, df_mean_cycle_unified, cfg)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    _save_or_show(fig, save_dir, f"{task_key}_torque_graph.png")


def _plot_stick_figure(ax: plt.Axes, df_mean_cycle_unified: pd.DataFrame,
                       cfg: dict) -> None:
    """
    XZ平面 (矢状面) スティック図を10%刻みで描画する。

    2026年度修正点:
        - X 反転を廃止 (X=進行正なのでそのまま左→右に歩行が表示される)
        - base_pos の値を np.array に統一 (calc_joint_center への正しい入力形式)
        - 各フレームを等間隔に横並びで表示するオフセット処理を整理
    """
    joint_defs      = cfg.get('JOINT_CENTER_DEFS', {})
    foot_marker_ids = cfg.get('SEGMENTS', {}).get('Foot', [])
    bones = [('Hip', 'Knee'), ('Knee', 'Ankle')]
    if foot_marker_ids:
        bones.append(('Ankle', 'Toe'))

    target_percs = np.linspace(0, 100, 11)

    # フレームごとのマーカー座標を事前取得
    grouped = df_mean_cycle_unified.groupby(
        df_mean_cycle_unified['gait_cycle_%'].apply(lambda v: round(v, 5))
    )

    # 0%フレームのHip X座標を基準点にする
    base_hip_x = 0.0
    try:
        base_data = grouped.get_group(round(0.0, 5))
        base_pos  = {int(r['id']): np.array([r['x'], r['y'], r['z']])
                     for _, r in base_data.iterrows()}
        hip_0 = calc_joint_center(joint_defs.get('Hip', {}), base_pos)
        if hip_0 is not None and not np.isnan(hip_0).any():
            base_hip_x = hip_0[0]
    except KeyError:
        pass

    # 表示幅: 全マーカーのX範囲から算出
    x_range = df_mean_cycle_unified['x'].max() - df_mean_cycle_unified['x'].min()
    frame_spacing = x_range * 1.3   # フレーム間の横間隔

    all_x, all_z = [], []

    for i, pct in enumerate(target_percs):
        try:
            frame = grouped.get_group(round(pct, 5))
        except KeyError:
            continue

        positions = {int(r['id']): np.array([r['x'], r['y'], r['z']])
                     for _, r in frame.iterrows()}
        jc = {n: calc_joint_center(d, positions) for n, d in joint_defs.items()}

        # 足部マーカーの重心を Toe として扱う
        if foot_marker_ids:
            pts = [positions[m] for m in foot_marker_ids if m in positions]
            jc['Toe'] = np.mean(pts, axis=0) if pts else np.full(3, np.nan)

        # このフレームのHip X座標とのずれを補正し、等間隔に並べる
        curr_hip = jc.get('Hip', np.zeros(3))
        curr_hip_x = curr_hip[0] if curr_hip is not None and not np.isnan(curr_hip).any() else base_hip_x
        offset_x = i * frame_spacing - (curr_hip_x - base_hip_x)
        offset_vec = np.array([offset_x, 0, 0])

        for (p1n, p2n) in bones:
            p1, p2 = jc.get(p1n), jc.get(p2n)
            if (p1 is not None and p2 is not None
                    and not np.isnan(p1).any() and not np.isnan(p2).any()):
                # XZ平面に投影 (Y成分は無視)
                p1s = p1 + offset_vec
                p2s = p2 + offset_vec
                ax.plot([p1s[0], p2s[0]], [p1s[2], p2s[2]],
                        color='black', marker='o', markersize=3, linewidth=1.5)
                all_x.extend([p1s[0], p2s[0]])
                all_z.extend([p1s[2], p2s[2]])

        # パーセント表示 (Hip の上方)
        hip_pt = jc.get('Hip')
        if hip_pt is not None and not np.isnan(hip_pt).any():
            label_x = hip_pt[0] + offset_vec[0]
            label_z = hip_pt[2]
            ax.text(label_x, label_z + 50, f"{int(pct)}%", ha='center', fontsize=9)

    if all_z:
        ax.set_ylim(min(min(all_z) - 50, 0), max(all_z) * 1.1 + 80)
    if all_x:
        ax.set_xlim(min(all_x) - 50, max(all_x) + 50)

    ax.set_title('Sagittal Plane Stick Figure — XZ projection (10% intervals)')
    ax.set_xlabel('X: Progression direction (mm)')
    ax.set_ylabel('Z: Vertical (mm)')
    ax.set_aspect('equal')


# ===========================================================================
# 仕事量時系列グラフ
# (昨年度 calculate_rubber_work.py: plot_work_time_series より移植・変更なし)
# ===========================================================================

def plot_work_time_series(df_instant: pd.DataFrame, df_cumulative: pd.DataFrame,
                           task_key: str, seg_groups: dict, T_cycle: float,
                           save_dir: str | None = None,
                           ylim_power: tuple | None = (-50, 50),
                           ylim_work: tuple | None = None) -> None:
    """
    瞬時仕事率(Power)と累積仕事量(Cumulative Work)の時系列グラフを描画する。

    上段: 瞬時仕事率 (Power [W])
    下段: 累積仕事量 (Cumulative Work [J])
    """
    for group_name, seg_list in seg_groups.items():
        inst_segs = [s for s in seg_list if s in df_instant.columns]
        cum_segs  = [s for s in seg_list if s in df_cumulative.columns]
        if not inst_segs and not cum_segs:
            continue

        fig, (ax_power, ax_work) = plt.subplots(
            2, 1, figsize=(12, 10),
            gridspec_kw={'height_ratios': [6, 4]}
        )
        fig.suptitle(f'Rubber Power & Work Analysis (Time Series)\n'
                     f'Group: {group_name} | Task: {task_key} | T_cycle={T_cycle:.3f}s', fontsize=16, y=0.98)

        if inst_segs:
            df_i = df_instant[inst_segs]
            df_i.plot(ax=ax_power, linewidth=1.5)
            x = df_i.index
            for seg in inst_segs:
                y = df_i[seg]
                ax_power.fill_between(x, y, 0, where=(y > 0), facecolor='red',   alpha=0.2, interpolate=True)
                ax_power.fill_between(x, y, 0, where=(y < 0), facecolor='green', alpha=0.2, interpolate=True)

        ax_power.axhline(0, color='black', linewidth=0.5)
        ax_power.set_title('Instantaneous Power (P = F * v)')
        ax_power.set_ylabel('Power (W)')
        ax_power.grid(True, linestyle='--', alpha=0.7)
        ax_power.legend(loc='upper right')
        if ylim_power:
            ax_power.set_ylim(ylim_power)
        ax_power.set_xlim(0, 100)
        ax_power.set_xlabel('Gait Cycle (%)')

        if cum_segs:
            df_cumulative[cum_segs].plot(ax=ax_work, linewidth=2.0)

        ax_work.set_title('Cumulative Work (W = ∫P dt)')
        ax_work.set_ylabel('Cumulative Work (J)\n(Positive = Net Release)')
        ax_work.grid(True, linestyle='--', alpha=0.7)
        ax_work.legend(loc='upper right')
        if ylim_work:
            ax_work.set_ylim(ylim_work)
        ax_work.set_xlim(0, 100)
        ax_work.set_xlabel('Gait Cycle (%)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        _save_or_show(fig, save_dir, f"{task_key}_{group_name}_work_timeseries.png")


# ===========================================================================
# ワークループグラフ
# (昨年度 analyze_futto_polar.py: run_work_loop_mode より移植)
# ===========================================================================

def plot_work_loops(df_mean_cycle: pd.DataFrame, df_tension: pd.DataFrame,
                    cfg: dict, task_key: str,
                    targets: list | None = None,
                    save_dir: str | None = None) -> None:
    """ゴムの伸び vs 張力ワークループをプロットする。"""
    df_tension = normalize_tension_df(df_tension)   # ワイド形式→ロング形式に統一
    lines_def = cfg.get('LINES_TO_DRAW', {})
    if not lines_def:
        print("LINES_TO_DRAW が空のためワークループをスキップします。")
        return

    if targets is None:
        targets = list(lines_def.keys())

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
        return

    cycles  = coord_pivot.index.values
    results = {}
    for name in targets:
        if name not in lines_def or name not in tension_pivot.columns:
            continue
        p1_id, p2_id = lines_def[name]
        try:
            p1c = coord_pivot[[f"{p1_id}_x", f"{p1_id}_y", f"{p1_id}_z"]].values
            p2c = coord_pivot[[f"{p2_id}_x", f"{p2_id}_y", f"{p2_id}_z"]].values
            lengths  = np.linalg.norm(p1c - p2c, axis=1)
            tensions = tension_pivot[name].values
            results[name] = {'len': lengths, 'ten': tensions}
        except KeyError:
            continue

    if not results:
        print("ワークループ描画データがありません。")
        return

    cols = 3
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        L, T = data['len'], data['ten']
        sc = ax.scatter(L, T, c=cycles, cmap='hsv', s=15, alpha=0.8)
        ax.plot(L, T, 'k-', alpha=0.3, linewidth=1)
        ax.plot(L[0], T[0], 'ko', markersize=8, markerfacecolor='white', label='Start')
        if np.max(L) - np.min(L) > 5:
            z = np.polyfit(L, T, 1)
            ax.text(0.05, 0.9, f'k ≈ {z[0]:.2f} N/mm',
                    transform=ax.transAxes, fontsize=9, color='blue')
        ax.set_title(name)
        ax.set_xlabel('Length (mm)')
        ax.set_ylabel('Tension (N)')
        ax.grid(True, linestyle='--')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Gait Cycle (%)')
    plt.suptitle(f'Work Loops (Stiffness & Hysteresis) - {task_key}', fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.4)
    _save_or_show(fig, save_dir, f"{task_key}_work_loops.png")


# ===========================================================================
# ポーラーチャート
# (昨年度 analyze_futto_polar.py: run_polar_force_mode より移植)
# ===========================================================================

def plot_polar_force(net_force: np.ndarray, cycles: np.ndarray,
                     task_key: str, save_dir: str | None = None) -> None:
    """
    下腿合力ベクトルの方向・強さをポーラーチャートで表示する。

    2026年度座標系 (全タスク共通):
        X = 進行方向 (前が正)
        Z = 鉛直上方 (上が正)
    矢状面内の合力を XZ 平面に投影し、
        P_vec = Fx (進行方向成分)
        V_vec = Fz (垂直方向成分)
    として極座標に変換する。
    0° = 前方 (Fx正), 90° = 上方 (Fz正)
    """
    Fx, Fz = net_force[:, 0], net_force[:, 2]

    # 2026年度: 全タスク共通座標系なのでタスク分岐不要
    P_vec = Fx   # 進行方向成分
    V_vec = Fz   # 垂直上方成分

    r     = np.sqrt(V_vec ** 2 + P_vec ** 2)
    theta = np.arctan2(V_vec, P_vec)

    # 平均方向が前上象限 (0°-90°) に来るように補正
    mean_theta = np.mean(theta)
    if mean_theta < -np.pi / 2:
        theta += np.pi

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='polar')
    sc  = ax.scatter(theta, r, c=cycles, cmap='hsv', s=30, alpha=0.8)
    ax.plot(theta, r, 'k-', alpha=0.4, linewidth=1)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(
        f"Shank Assist Vector Polar Chart - {task_key}\n"
        "0°=Anterior(+X), 90°=Superior(+Z)  [XZ sagittal projection]",
        va='bottom', fontsize=14
    )
    plt.colorbar(sc, label='Gait Cycle (%)', fraction=0.03, pad=0.04)
    _save_or_show(fig, save_dir, f"{task_key}_polar_force.png")


# ===========================================================================
# 関節角度時系列グラフ (今年度新規追加)
# ===========================================================================

def plot_joint_angles(df_angles: pd.DataFrame, task_key: str,
                      save_dir: str | None = None) -> None:
    """
    Hip / Knee / Ankle の矢状面内屈曲伸展角度を時系列でプロットする。

    グラフ形式:
        - 3軸 (Hip/Knee/Ankle) を縦に並べて配置
        - 塗りつぶし: 屈曲方向 (角度増加側) を青で強調
        - 背景色: 立脚期 (0-60%) と遊脚期 (60-100%) を色分け
    """
    if df_angles is None or df_angles.empty:
        print("関節角度データがないためグラフをスキップします。")
        return

    angle_cols = {
        'hip_angle_deg':   'Hip Angle',
        'knee_angle_deg':  'Knee Angle',
        'ankle_angle_deg': 'Ankle Angle',
    }
    available = [c for c in angle_cols if c in df_angles.columns]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 4 * len(available)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Joint Angles (Sagittal Plane) - {task_key}", fontsize=16)

    x = df_angles['gait_cycle_%'].values

    for ax, col in zip(axes, available):
        y = df_angles[col].values
        ax.plot(x, y, color='steelblue', linewidth=2.5, label=angle_cols[col])

        # 立脚期 / 遊脚期 背景
        ax.axvspan(0,  60, alpha=0.06, color='green',  label='Stance phase')
        ax.axvspan(60, 100, alpha=0.06, color='orange', label='Swing phase')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.fill_between(x, y, np.nanmean(y), alpha=0.15, color='steelblue')

        ax.set_title(angle_cols[col])
        ax.set_ylabel('Angle (°)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel('Gait Cycle (%)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_or_show(fig, save_dir, f"{task_key}_joint_angles.png")


# ===========================================================================
# タスク間比較グラフ (今年度新規追加)
# ===========================================================================

def plot_task_comparison(torque_data_dict: dict,
                          joint_name: str = 'Knee',
                          axis: str = 'torque_y_Nm',
                          save_dir: str | None = None) -> None:
    """
    複数タスクのトルクを同一スケールでオーバーレイ表示する。

    Parameters
    ----------
    torque_data_dict : dict
        {'task01': df_torque, 'task02': df_torque, ...}
    joint_name : str
        比較する関節名 ('Hip', 'Knee', 'Ankle')
    axis : str
        比較する軸 ('torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm')

    グラフ形式:
        - タスクを色で区別 (task01=青, task02=橙, task03=緑)
        - 同一 Y スケール (全タスクのデータ範囲で統一)
        - 差分 (task01 - task03, task02 - task03) を下段に追加
    """
    TASK_COLORS = {'task01': '#1f77b4', 'task02': '#ff7f0e', 'task03': '#2ca02c'}
    AXIS_LABELS = {
        'torque_x_Nm': 'Adduction/Abduction Torque (Nm)',
        'torque_y_Nm': 'Flexion/Extension Torque (Nm)',
        'torque_z_Nm': 'Int/Ext Rotation Torque (Nm)',
    }

    valid = {k: v for k, v in torque_data_dict.items()
             if v is not None and not v.empty}
    if not valid:
        print("比較データがありません。")
        return

    fig, (ax_main, ax_diff) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 2]})
    fig.suptitle(
        f"Task Comparison: {joint_name} | {AXIS_LABELS.get(axis, axis)}",
        fontsize=16
    )

    series_dict = {}
    for task_key, df in valid.items():
        jdata = df[df['joint'] == joint_name]
        if jdata.empty:
            continue
        grouped = jdata.groupby('gait_cycle_%')[axis].sum()
        series_dict[task_key] = grouped
        color = TASK_COLORS.get(task_key, 'gray')
        ax_main.plot(grouped.index, grouped.values,
                     color=color, linewidth=2.5, label=task_key)

    ax_main.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_main.set_ylabel(AXIS_LABELS.get(axis, axis))
    ax_main.set_title(f"Absolute Torque — {joint_name}")
    ax_main.legend()
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.set_xlim(0, 100)

    # 差分プロット (task03 = ベースライン)
    if 'task03' in series_dict:
        base = series_dict['task03']
        for key in ['task01', 'task02']:
            if key in series_dict:
                diff = series_dict[key] - base
                color = TASK_COLORS.get(key, 'gray')
                ax_diff.plot(diff.index, diff.values,
                             color=color, linewidth=2.0,
                             label=f"{key} - task03 (Futto effect)")
        ax_diff.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_diff.fill_between(diff.index, diff.values, 0,
                             where=(diff.values > 0),
                             facecolor='red', alpha=0.15, label='Positive contribution')
        ax_diff.fill_between(diff.index, diff.values, 0,
                             where=(diff.values < 0),
                             facecolor='blue', alpha=0.15, label='Negative contribution')
        ax_diff.set_title(f"Δ Torque (Task − task03, Futto contribution)")
        ax_diff.legend()
        ax_diff.grid(True, linestyle='--', alpha=0.6)
    else:
        ax_diff.set_visible(False)

    ax_diff.set_xlabel('Gait Cycle (%)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"comparison_{joint_name}_{axis.replace('_Nm','')}.png"
    _save_or_show(fig, save_dir, fname)


# ===========================================================================
# 3D力場アニメーション
# (昨年度 visualize_force_field_3d.py: animate_force_field より移植)
# ===========================================================================

def animate_force_field_3d(frames_data: list, cfg: dict,
                            task_key: str, save_dir: str | None = None) -> None:
    """3D力場アニメーションを生成する。"""
    if not frames_data:
        print("フレームデータがありません。")
        return

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')

    all_pos = np.array([p for f in frames_data for p in f['pos_map'].values()])
    max_range = np.ptp(all_pos, axis=0).max() / 2.0
    mid = (all_pos.max(axis=0) + all_pos.min(axis=0)) / 2.0

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'{task_key} Force Field\nRed=Assist, Blue=Brake, Arrow=Torque')
    ax.view_init(elev=30, azim=60)  # strength_visualize と同一の視点

    lines_def = cfg.get('LINES_TO_DRAW', {})
    rubber_lines = {name: ax.plot([], [], [], linewidth=4)[0] for name in lines_def}
    scat = ax.scatter([], [], [], c='gray', s=10, alpha=0.4)
    skeleton_line, = ax.plot([], [], [], color='black', linewidth=5,
                              marker='o', markersize=6, zorder=10)

    def _gradient_color(work_val):
        white = np.ones(3)
        intensity = min(abs(work_val) / WORK_SENSITIVITY, 1.0) ** 0.5
        color = (1 - intensity) * white + intensity * (
            np.array([1., 0., 0.]) if work_val < 0 else np.array([0., 0., 1.])
        )
        return tuple(color), 0.8 + 0.2 * intensity

    def update(frame_idx):
        data = frames_data[frame_idx]
        ax.set_title(f"Gait Cycle: {int(data['step'])}%")

        # --- マーカー ---
        xs = [p[0] for p in data['pos_map'].values()]
        ys = [p[1] for p in data['pos_map'].values()]
        zs = [p[2] for p in data['pos_map'].values()]
        scat._offsets3d = (xs, ys, zs)

        # --- ゴム線 ---
        for name, line in rubber_lines.items():
            if name in data['rubber_states']:
                st  = data['rubber_states'][name]
                p1, p2 = st['p1'], st['p2']
                w   = st.get('smoothed_work', 0.0)
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
                c, a = _gradient_color(w)
                line.set_color(c); line.set_alpha(a)

        # --- 骨格線 ---
        jx, jy, jz = [], [], []
        for jname in ['Hip', 'Knee', 'Ankle']:
            if jname in data['joint_centers']:
                pt = data['joint_centers'][jname]
                if pt is not None and not np.any(np.isnan(pt)):
                    jx.append(pt[0]); jy.append(pt[1]); jz.append(pt[2])
        if jx:
            skeleton_line.set_data(jx, jy)
            skeleton_line.set_3d_properties(jz)

        # --- トルク矢印: 毎フレーム ax の quiver を全削除して再描画 ---
        # FuncAnimation + Pillow の組み合わせでは quivers リストへの
        # append/remove が正しく反映されないことがあるため、
        # ax.collections から直接削除する方式に変更する。
        for coll in list(ax.collections):
            if hasattr(coll, '_is_torque_quiver') and coll._is_torque_quiver:
                coll.remove()

        for jname, tau in data['joint_torques'].items():
            if jname not in data['joint_centers']:
                continue
            center = data['joint_centers'][jname]
            if center is None or np.any(np.isnan(center)):
                continue
            # Y軸回りのトルク (屈曲伸展) のみを矢印で表示
            u, v, w_t = 0, float(tau[1]), 0
            if abs(v) > 0.05:
                q = ax.quiver(
                    center[0], center[1], center[2],
                    u * TORQUE_SCALE, v * TORQUE_SCALE, w_t * TORQUE_SCALE,
                    color='gold', linewidth=4, arrow_length_ratio=0.3
                )
                q._is_torque_quiver = True   # 識別フラグ

        return [scat, skeleton_line] + list(rubber_lines.values())

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=ANIMATION_INTERVAL,
        blit=False,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{task_key}_force_field_3d.mp4")
        try:
            ani.save(save_path, writer='ffmpeg', fps=10)
            print(f"  -> アニメーションを保存しました: {save_path}")
        except Exception:
            gif_path = save_path.replace('.mp4', '.gif')
            try:
                ani.save(gif_path, writer='pillow', fps=10)
                print(f"  -> GIF保存: {gif_path}")
            except Exception as e:
                print(f"  -> アニメーション保存失敗: {e}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def plot_joint_power(df_power, df_torque, df_angles,
                     task_key, speed='', save_dir=None):
    """
    関節パワー P_y = tau_y x omega_y の時系列グラフを描画する。

    上段: 各関節のパワー時系列 (Hip/Knee/Ankle)
          P > 0: アシスト方向 / P < 0: 吸収方向
    中段: 参照用トルク時系列
    下段: 参照用関節角度時系列
    """
    if df_power is None or df_power.empty:
        print("パワーデータがないためグラフをスキップします。")
        return

    JOINT_COLORS = {'Hip': '#e6194b', 'Knee': '#3cb44b', 'Ankle': '#4363d8'}
    POWER_COLS = {
        'hip_power_W':   ('Hip',   '#e6194b'),
        'knee_power_W':  ('Knee',  '#3cb44b'),
        'ankle_power_W': ('Ankle', '#4363d8'),
    }
    ANGLE_COLS = {
        'hip_angle_deg':   ('Hip',   '#e6194b'),
        'knee_angle_deg':  ('Knee',  '#3cb44b'),
        'ankle_angle_deg': ('Ankle', '#4363d8'),
    }

    has_torque = df_torque is not None and not df_torque.empty
    has_angles = df_angles is not None and not df_angles.empty
    n_rows = 1 + int(has_torque) + int(has_angles)
    heights = [4] + [2] * (n_rows - 1)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12, 4 * n_rows),
        sharex=True, gridspec_kw={'height_ratios': heights}
    )
    axes = np.atleast_1d(axes)

    fig.suptitle(
        'Joint Power  P = tau_y x omega_y'
        '  |  Task: ' + task_key + '  Speed: ' + speed + 'm/s\n'
        'P > 0: Assist  /  P < 0: Absorb',
        fontsize=13
    )

    x = df_power['gait_cycle_%'].values

    # --- 上段: パワー ---
    ax_p = axes[0]
    for col, (label, color) in POWER_COLS.items():
        if col not in df_power.columns:
            continue
        y = df_power[col].values
        ax_p.plot(x, y, color=color, linewidth=2.5, label=label)
        ax_p.fill_between(x, y, 0, where=(y > 0), alpha=0.15, color=color)
        ax_p.fill_between(x, y, 0, where=(y < 0), alpha=0.08, color=color,
                          hatch='///', linewidth=0)

    ax_p.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_p.axvspan(0,   60, alpha=0.04, color='green',  label='Stance')
    ax_p.axvspan(60, 100, alpha=0.04, color='orange', label='Swing')
    ax_p.set_ylabel('Power (W)')
    ax_p.set_title('Joint Power  [P_y = tau_y x omega_y]')
    ax_p.grid(True, linestyle='--', alpha=0.6)
    ax_p.legend(loc='upper right', fontsize=8, ncol=2)
    ax_p.set_xlim(0, 100)

    row = 1

    # --- 中段: トルク (参照) ---
    if has_torque:
        ax_t = axes[row]; row += 1
        tau_g = (
            df_torque
            .groupby(['gait_cycle_%', 'joint'])['torque_y_Nm']
            .sum().unstack('joint')
        )
        for joint, color in JOINT_COLORS.items():
            if joint in tau_g.columns:
                ax_t.plot(tau_g.index, tau_g[joint],
                          color=color, linewidth=1.8, label=joint)
        ax_t.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_t.set_ylabel('Torque (Nm)')
        ax_t.set_title('Reference: Torque tau_y  [Extension(+) / Flexion(-)]')
        ax_t.grid(True, linestyle='--', alpha=0.6)
        ax_t.legend(fontsize=8)
        ax_t.set_xlim(0, 100)

    # --- 下段: 関節角度 (参照) ---
    if has_angles:
        ax_a = axes[row]
        for col, (label, color) in ANGLE_COLS.items():
            if col in df_angles.columns:
                ax_a.plot(df_angles['gait_cycle_%'], df_angles[col],
                          color=color, linewidth=1.8, label=label)
        ax_a.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_a.set_ylabel('Angle (deg)')
        ax_a.set_xlabel('Gait Cycle (%)')
        ax_a.set_title('Reference: Joint Angle  [Flexion(+) / Extension(-)]')
        ax_a.grid(True, linestyle='--', alpha=0.6)
        ax_a.legend(fontsize=8)
        ax_a.set_xlim(0, 100)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    _save_or_show(fig, save_dir, task_key + '_' + speed + 'ms_joint_power.png')


def _save_or_show(fig: plt.Figure, save_dir: str | None, filename: str) -> None:
    """save_dir が指定されていれば保存、なければ表示する。"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  -> グラフを保存しました: {path}")
        plt.close(fig)
    else:
        plt.show()