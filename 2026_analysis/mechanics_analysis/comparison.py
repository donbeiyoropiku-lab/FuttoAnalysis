# =============================================================================
# mechanics_analysis/comparison.py
#
# 役割:
#   発表資料向けに、同一フェーズ(同一速度)のタスク間 (task01/02/03) を
#   軸の範囲をそろえて比較する図を作成する。
#   個別解析 (visualizer.py) と同じ計算・描画スタイルを使い、
#   「全タスクのデータ範囲から共通の軸範囲を決める」点だけが異なる。
#
# 収録関数 (mechanics_main.py のメニュー番号に対応):
#   plot_torque_stick_comparison()  1: 関節トルク解析 + スティック図
#   plot_work_comparison()          2: ゴム仕事量 / エネルギー解析
#   plot_joint_angles_comparison()  5: 関節角度時系列
#
# 入力 task_data の形式:
#   {task_key: {'cfg': TASK_CONFIGS[task_key],
#               'df_unified': 座標統一済みの歩行周期平均データ,
#               'df_tension': 張力データ}}
#
# 軸のそろえ方:
#   1: トルクは関節ごとに全タスク共通の縦軸、スティック図は全タスク共通の
#      フレーム間隔・表示範囲
#   2: パワー・累積仕事は全タスク・全セグメントグループ共通の縦軸
#   5: 1つのグラフに全タスクを重ね描き (縦軸は関節ごとに共通)
#
# ゴムを持たないタスク (task03) はトルク・仕事量が存在しないため、
# 1 ではトルク欄に "No rubber" と表示してスティック図のみ比較し、
# 2 ではゴムのあるタスクだけを並べる。
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .physics_core import calc_all_torques, calc_work_data, calc_joint_angles
from .visualizer import _plot_stick_figure, _save_or_show

TASK_ORDER  = ['task01', 'task02', 'task03']
TASK_COLORS = {'task01': '#1f77b4', 'task02': '#ff7f0e', 'task03': '#2ca02c'}
TORQUE_COLS = ['torque_x_Nm', 'torque_y_Nm', 'torque_z_Nm']
JOINT_ORDER = ['Hip', 'Knee', 'Ankle']


# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------

def _ordered_tasks(task_data: dict) -> list:
    """task01, task02, task03 の順に並べたタスク名リストを返す。"""
    return [t for t in TASK_ORDER if t in task_data]


def _task_title(task_key: str, task_titles: dict | None) -> str:
    return (task_titles or {}).get(task_key, task_key)


def _shared_limits(arrays, pad: float = 0.08, include_zero: bool = True):
    """
    複数配列の全体から共通の軸範囲 (lo, hi) を決める。有効な値がなければ None。

    include_zero=True の場合は 0 を必ず範囲に含める (0 基準線を描くグラフ用)。
    """
    chunks = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    chunks = [c[np.isfinite(c)] for c in chunks if c.size]
    if not chunks:
        return None
    vals = np.concatenate(chunks)
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo
    if span < 1e-9:
        span = 1.0
    return lo - pad * span, hi + pad * span


def _placeholder(ax, text: str) -> None:
    """データがないパネルに説明文を表示する (軸・グリッドはそのまま残す)。"""
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha='center', va='center', color='gray', fontsize=11)


# ===========================================================================
# 1: 関節トルク解析 + スティック図 の タスク間比較
# ===========================================================================

def plot_torque_stick_comparison(task_data: dict, phase, speed: str,
                                 save_dir: str, task_titles: dict | None = None) -> None:
    """
    列=タスク、行=関節 (Hip/Knee/Ankle) のトルク時系列と、
    タスクごとのスティック図 (下段、全幅) を1枚にまとめる。

    トルクの縦軸は関節ごとに全タスク共通、スティック図は全タスク共通の
    フレーム間隔・表示範囲で描画する。
    """
    tasks = _ordered_tasks(task_data)
    if len(tasks) < 2:
        print("  -> タスクが2つ未満のため、トルク+スティック図の比較をスキップします。")
        return

    # --- トルク計算 (ゴムなしタスクは空 DataFrame → torque_by_task に入れない) ---
    torque_by_task = {}
    for tk in tasks:
        d = task_data[tk]
        t_df = calc_all_torques(d['df_unified'], d['df_tension'], d['cfg'])
        if t_df is not None and not t_df.empty:
            torque_by_task[tk] = (
                t_df.groupby(['gait_cycle_%', 'joint'])[TORQUE_COLS]
                .sum().reset_index()
            )

    joints = [j for j in JOINT_ORDER
              if any(j in df['joint'].values for df in torque_by_task.values())]

    # --- 関節ごとの共通縦軸 (全タスク・全軸成分のデータ範囲) ---
    joint_ylim = {}
    for j in joints:
        series = [df.loc[df['joint'] == j, c].values
                  for df in torque_by_task.values() for c in TORQUE_COLS]
        joint_ylim[j] = _shared_limits(series)

    # --- スティック図の共通フレーム間隔 (最もX範囲が広いタスクに合わせる) ---
    spacing = max(
        (task_data[tk]['df_unified']['x'].max() - task_data[tk]['df_unified']['x'].min()) * 1.3
        for tk in tasks
    )

    n_cols, n_joints, n_tasks = len(tasks), len(joints), len(tasks)
    fig = plt.figure(figsize=(6 * n_cols, 3.4 * n_joints + 2.8 * n_tasks))
    gs = fig.add_gridspec(n_joints + n_tasks, n_cols,
                          height_ratios=[3] * n_joints + [2] * n_tasks)
    fig.suptitle(
        f"Task Comparison: Joint Torque & Stick Figure  |  Phase {phase} ({speed} m/s)\n"
        "torque_y: Extension(+) / Flexion(-)  |  Axes are shared across tasks",
        fontsize=15
    )

    # --- トルク (行=関節, 列=タスク) ---
    for r, joint in enumerate(joints):
        for c, tk in enumerate(tasks):
            ax = fig.add_subplot(gs[r, c])
            if tk in torque_by_task:
                jdata = torque_by_task[tk][torque_by_task[tk]['joint'] == joint]
            else:
                jdata = pd.DataFrame()

            if jdata.empty:
                _placeholder(ax, "No rubber torque\n(Futto not worn / no data)")
            else:
                ax.plot(jdata['gait_cycle_%'], jdata['torque_y_Nm'], color='red',
                        linewidth=3.0, label='Extension(+)/Flexion(-)  [Y-axis]')
                ax.plot(jdata['gait_cycle_%'], jdata['torque_x_Nm'], linewidth=1.5,
                        alpha=0.7, linestyle='--', label='X-axis moment (ref)')
                ax.plot(jdata['gait_cycle_%'], jdata['torque_z_Nm'], linewidth=1.5,
                        alpha=0.7, linestyle='--', label='Z-axis moment (ref)')
                ax.legend(fontsize=7, loc='upper right')

            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlim(0, 100)
            if joint_ylim[joint] is not None:
                ax.set_ylim(joint_ylim[joint])
            ax.grid(True)
            if r == 0:
                ax.set_title(_task_title(tk, task_titles), fontsize=13)
            if c == 0:
                ax.set_ylabel(f"{joint}\nTorque (Nm)")
            else:
                ax.tick_params(labelleft=False)
            if r < n_joints - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Gait Cycle (%)")

    # --- スティック図 (行=タスク, 全幅) ---
    stick_axes = []
    for k, tk in enumerate(tasks):
        ax = fig.add_subplot(gs[n_joints + k, :])
        d = task_data[tk]
        _plot_stick_figure(ax, d['df_unified'], d['cfg'], frame_spacing=spacing)
        ax.set_title(f"{_task_title(tk, task_titles)}  -  Sagittal Plane Stick Figure "
                     "(XZ projection, 10% intervals)", fontsize=11)
        stick_axes.append(ax)

    # 全タスクの表示範囲を和集合にそろえる
    x_lo = min(ax.get_xlim()[0] for ax in stick_axes)
    x_hi = max(ax.get_xlim()[1] for ax in stick_axes)
    z_lo = min(ax.get_ylim()[0] for ax in stick_axes)
    z_hi = max(ax.get_ylim()[1] for ax in stick_axes)
    for ax in stick_axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(z_lo, z_hi)

    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    _save_or_show(fig, save_dir, f"comparison_torque_stick_Phase{phase}_{speed}ms.png")


# ===========================================================================
# 2: ゴム仕事量 / エネルギー解析 の タスク間比較
# ===========================================================================

def plot_work_comparison(task_data: dict, phase, speed: str, save_dir: str,
                         seg_groups: dict, task_titles: dict | None = None) -> None:
    """
    セグメントグループごとに、列=タスクのパワー・累積仕事の時系列を描く
    (縦軸は全タスク・全グループ共通)。あわせて総仕事量・グループ別正味仕事・
    総弾性エネルギーをまとめたサマリー図を1枚出力する。

    ゴムを持たないタスク (task03 等) は対象外。
    """
    results = {}
    for tk in _ordered_tasks(task_data):
        d = task_data[tk]
        summary, inst, cum, energy, T_cycle = calc_work_data(
            d['df_unified'], d['df_tension'], d['cfg']
        )
        if summary is not None and inst is not None:
            results[tk] = {'summary': summary, 'inst': inst, 'cum': cum,
                           'energy': energy, 'T_cycle': T_cycle}

    tasks = list(results)
    if len(tasks) < 2:
        print("  -> ゴムのあるタスクが2つ未満のため、仕事量の比較をスキップします。")
        return

    # --- 全タスク・全グループ共通の縦軸 ---
    power_arrays, work_arrays = [], []
    for res in results.values():
        for seg_list in seg_groups.values():
            power_arrays += [res['inst'][s].values for s in seg_list if s in res['inst'].columns]
            work_arrays  += [res['cum'][s].values  for s in seg_list if s in res['cum'].columns]
    ylim_power = _shared_limits(power_arrays)
    ylim_work  = _shared_limits(work_arrays)

    # --- グループごとの時系列比較 ---
    for group_name, seg_list in seg_groups.items():
        segs_by_task = {
            tk: [s for s in seg_list if s in results[tk]['inst'].columns] for tk in tasks
        }
        if not any(segs_by_task.values()):
            continue

        fig, axes = plt.subplots(2, len(tasks), figsize=(6.5 * len(tasks), 10),
                                 sharex=True, squeeze=False,
                                 gridspec_kw={'height_ratios': [6, 4]})
        fig.suptitle(
            f"Rubber Power & Work Comparison  |  Group: {group_name}  |  "
            f"Phase {phase} ({speed} m/s)\nAxes are shared across tasks and groups",
            fontsize=15
        )

        for c, tk in enumerate(tasks):
            ax_p, ax_w = axes[0, c], axes[1, c]
            segs = segs_by_task[tk]
            if segs:
                inst, cum = results[tk]['inst'], results[tk]['cum']
                x = inst.index.values
                for s in segs:
                    y = inst[s].values
                    ax_p.plot(x, y, linewidth=1.5, label=s)
                    ax_p.fill_between(x, y, 0, where=(y > 0), facecolor='red',
                                      alpha=0.2, interpolate=True)
                    ax_p.fill_between(x, y, 0, where=(y < 0), facecolor='green',
                                      alpha=0.2, interpolate=True)
                    ax_w.plot(cum.index.values, cum[s].values, linewidth=2.0, label=s)
                ax_p.legend(loc='upper right', fontsize=8)
                ax_w.legend(loc='upper right', fontsize=8)
            else:
                _placeholder(ax_p, f"No segments of group '{group_name}'")
                _placeholder(ax_w, f"No segments of group '{group_name}'")

            for ax in (ax_p, ax_w):
                ax.axhline(0, color='black', linewidth=0.5)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xlim(0, 100)
            if ylim_power is not None:
                ax_p.set_ylim(ylim_power)
            if ylim_work is not None:
                ax_w.set_ylim(ylim_work)

            ax_p.set_title(_task_title(tk, task_titles), fontsize=13)
            ax_w.set_xlabel('Gait Cycle (%)')
            if c == 0:
                ax_p.set_ylabel('Power (W)\n[P = F * v]')
                ax_w.set_ylabel('Cumulative Work (J)\n(Positive = Net Release)')
            else:
                ax_p.tick_params(labelleft=False)
                ax_w.tick_params(labelleft=False)

        fig.tight_layout(rect=[0, 0.01, 1, 0.93])
        _save_or_show(fig, save_dir,
                      f"comparison_work_{group_name}_Phase{phase}_{speed}ms.png")

    _plot_work_summary(results, phase, speed, save_dir, seg_groups, task_titles)


def _plot_work_summary(results: dict, phase, speed: str, save_dir: str,
                       seg_groups: dict, task_titles: dict | None) -> None:
    """総仕事量・グループ別正味仕事・総弾性エネルギーを1枚にまとめる。"""
    tasks = list(results)
    n = len(tasks)
    width = 0.8 / n

    # A: 総仕事量 (正 / 負 / 正味)
    total_labels = ['Positive', 'Negative', 'Net']
    total_vals = {
        tk: [results[tk]['summary']['positive_work_J'].sum(),
             results[tk]['summary']['negative_work_J'].sum(),
             results[tk]['summary']['net_work_J'].sum()]
        for tk in tasks
    }

    # B: グループ別の正の仕事 / 負の仕事
    #    (1周期を通したゴムの正味仕事はほぼ0になるため、正味ではなく正・負を別々に示す)
    group_names = [g for g, segs in seg_groups.items()
                   if any(s in results[tk]['summary']['segment'].values
                          for tk in tasks for s in segs)]

    def _group_work(tk, col):
        summ = results[tk]['summary']
        return [summ.loc[summ['segment'].isin(seg_groups[g]), col].sum()
                for g in group_names]

    group_pos = {tk: _group_work(tk, 'positive_work_J') for tk in tasks}
    group_neg = {tk: _group_work(tk, 'negative_work_J') for tk in tasks}

    # A と B は同じ縦軸 (仕事 [J]) にそろえる
    ylim_work = _shared_limits(
        [np.array(v) for v in total_vals.values()] +
        [np.array(v) for v in group_pos.values()] +
        [np.array(v) for v in group_neg.values()]
    )

    fig, (ax_a, ax_b, ax_e) = plt.subplots(
        1, 3, figsize=(20, 6.5), gridspec_kw={'width_ratios': [1, 1.3, 1.5]}
    )
    fig.suptitle(f"Rubber Work & Elastic Energy Comparison  |  "
                 f"Phase {phase} ({speed} m/s)", fontsize=15)

    for i, tk in enumerate(tasks):
        offset = (i - (n - 1) / 2) * width
        color  = TASK_COLORS.get(tk, 'gray')
        label  = _task_title(tk, task_titles)
        xs_a = np.arange(len(total_labels))
        xs_b = np.arange(len(group_names))
        ax_a.bar(xs_a + offset, total_vals[tk], width, color=color, label=label)
        ax_b.bar(xs_b + offset, group_pos[tk], width, color=color, label=label)
        ax_b.bar(xs_b + offset, group_neg[tk], width, color=color, alpha=0.45)

    for ax, labels, title in (
        (ax_a, total_labels, "Total rubber work per gait cycle"),
        (ax_b, group_names,  "Work by segment group  (solid: positive / light: negative)"),
    ):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Work (J)')
        ax.set_title(title)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
        if ylim_work is not None:
            ax.set_ylim(ylim_work)

    # C: 総弾性エネルギー (全ゴム合計) の時系列
    for tk in tasks:
        energy = results[tk]['energy']
        if energy is None or energy.empty:
            continue
        ax_e.plot(energy.index.values, energy.sum(axis=1).values, linewidth=2.5,
                  color=TASK_COLORS.get(tk, 'gray'), label=_task_title(tk, task_titles))
    ax_e.set_xlim(0, 100)
    ax_e.set_xlabel('Gait Cycle (%)')
    ax_e.set_ylabel('Total elastic energy (J)')
    ax_e.set_title('Total elastic energy (sum of all rubbers)')
    ax_e.grid(True, linestyle='--', alpha=0.6)
    ax_e.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    _save_or_show(fig, save_dir, f"comparison_work_summary_Phase{phase}_{speed}ms.png")


# ===========================================================================
# 5: 関節角度時系列 の タスク間比較
# ===========================================================================

def plot_joint_angles_comparison(task_data: dict, phase, speed: str, save_dir: str,
                                 task_titles: dict | None = None) -> None:
    """Hip / Knee / Ankle の関節角度を、全タスクを重ね描きして比較する。"""
    angles = {}
    for tk in _ordered_tasks(task_data):
        d = task_data[tk]
        df = calc_joint_angles(d['df_unified'], d['cfg'])
        if df is not None and not df.empty:
            angles[tk] = df

    if len(angles) < 2:
        print("  -> 関節角度を算出できたタスクが2つ未満のため、角度比較をスキップします。")
        return

    angle_cols = [('hip_angle_deg', 'Hip Angle'),
                  ('knee_angle_deg', 'Knee Angle')]

    # Ankle は「Foot の基準点 (Toe)」の定義がタスク間で同じ場合のみ比較できる。
    # calc_joint_angles() は JOINT_CENTER_DEFS に 'Toe' が無いと Foot セグメントの
    # 重心を Toe の代わりに使う。task01 は Futto 取付マーカー6点の重心、
    # task02 は仮想足関節と同じ2点の重心 (= ベクトル長0で全フレーム NaN) となり、
    # 実測のつま先マーカーを持つ task03 とは全く別の量になってしまう。
    missing_toe = [tk for tk in angles
                   if 'Toe' not in task_data[tk]['cfg'].get('JOINT_CENTER_DEFS', {})]
    ankle_comparable = not missing_toe
    if ankle_comparable:
        angle_cols.append(('ankle_angle_deg', 'Ankle Angle'))
    else:
        print(f"  -> Ankle角度は比較から除外しました (つま先マーカーの定義が無いタスク: "
              f"{', '.join(missing_toe)} → 足部の基準点が異なり比較不能)。")

    fig, axes = plt.subplots(len(angle_cols), 1, figsize=(12, 4.2 * len(angle_cols) + 0.6),
                             sharex=True)
    fig.suptitle(f"Joint Angle Comparison (Sagittal Plane)  |  "
                 f"Phase {phase} ({speed} m/s)", fontsize=15)

    for ax, (col, label) in zip(axes, angle_cols):
        for tk, df in angles.items():
            if col not in df.columns:
                continue
            ax.plot(df['gait_cycle_%'], df[col], linewidth=2.5,
                    color=TASK_COLORS.get(tk, 'gray'), label=_task_title(tk, task_titles))
        ax.axvspan(0,  60, alpha=0.06, color='green')
        ax.axvspan(60, 100, alpha=0.06, color='orange')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(label)
        ax.set_ylabel('Angle (deg)\nFlexion(+) / Extension(-)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel('Gait Cycle (%)   [green: stance 0-60%, orange: swing 60-100%]')
    fig.tight_layout(rect=[0, 0.04 if not ankle_comparable else 0.01, 1, 0.95])
    if not ankle_comparable:
        fig.text(0.5, 0.012,
                 f"Ankle angle omitted: no toe marker in {', '.join(missing_toe)} "
                 "(foot reference point differs from the other tasks, so values are not comparable)",
                 ha='center', fontsize=9, color='dimgray')
    _save_or_show(fig, save_dir, f"comparison_joint_angles_Phase{phase}_{speed}ms.png")
