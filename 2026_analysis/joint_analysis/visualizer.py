# =============================================================================
# joint_analysis/visualizer.py
#
# 役割:
#   算出した仮想関節座標と、元のOptiTrackマーカーを同一3D空間に重ねて
#   アニメーション表示する。仮想関節定義の妥当性を目視で確認するためのツール。
#
# 表示内容:
#   ・OptiTrack マーカー    : 色付き小球 (タブ色、ID番号表示)
#   ・仮想関節点            : 大きめの球 (関節ごとに固定色)
#   ・関節間ボーン線        : Hip-Knee-Ankle を黒太線で接続
#   ・ゴムセグメント線      : LINES_TO_DRAW が定義されていれば表示
#   ・スペースキー          : 一時停止 / 再開
#   ・← → キー            : フレーム単位の手動送り (一時停止中)
#
# 使用方法:
#   joint_analysis/main.py のメニューから呼び出す。
#   直接実行も可能 (python -m joint_analysis.visualizer)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .joint_calc import calc_joint_center_at_frame

# 関節ごとの表示色 (固定)
JOINT_COLORS = {
    'Hip':   '#e6194b',   # 赤
    'Knee':  '#3cb44b',   # 緑
    'Ankle': '#4363d8',   # 青
    'Heel':  '#f58231',   # 橙
    'Toe':   '#911eb4',   # 紫
}
JOINT_MARKER_SIZE  = 12   # 仮想関節球のサイズ
BONE_ORDER         = ['Hip', 'Knee', 'Ankle']  # ボーンを繋ぐ順序


def animate_joint_check(df_mean_cycle: pd.DataFrame,
                         df_joints: pd.DataFrame,
                         cfg: dict,
                         task_key: str,
                         phase: int,
                         speed: str) -> None:
    """
    仮想関節 + OptiTrackマーカーを重ねた3Dアニメーションを表示する。

    Parameters
    ----------
    df_mean_cycle : pd.DataFrame
        OptiTrack 平均化データ (gait_cycle_%, id, x, y, z)
    df_joints : pd.DataFrame
        calc_joint_centers() の出力 (gait_cycle_%, Hip_x, ..., Knee_x, ...)
    cfg : dict
        TASK_CONFIGS[task_key]
    task_key : str
    phase : int
    speed : str
    """
    joint_defs  = cfg.get('JOINT_CENTER_DEFS', {})
    lines_def   = cfg.get('LINES_TO_DRAW', {})
    joint_names = list(joint_defs.keys())

    frames = sorted(df_mean_cycle['gait_cycle_%'].unique())
    n_frames = len(frames)

    # ---------- 軸範囲を全マーカーデータから算出 ----------
    all_x = df_mean_cycle['x'].values
    all_y = df_mean_cycle['y'].values
    all_z = df_mean_cycle['z'].values
    max_range = np.array([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min(),
    ]).max() * 1.15
    mid = np.array([(all_x.max()+all_x.min())/2,
                    (all_y.max()+all_y.min())/2,
                    (all_z.max()+all_z.min())/2])

    # ---------- Figure / Axes ----------
    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim(mid[0]-max_range/2, mid[0]+max_range/2)
    ax.set_ylim(mid[1]-max_range/2, mid[1]+max_range/2)
    ax.set_zlim(mid[2]-max_range/2, mid[2]+max_range/2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.view_init(elev=30, azim=60)

    # ---------- マーカー描画オブジェクト ----------
    marker_ids = sorted(df_mean_cycle['id'].unique())
    cmap_m     = plt.get_cmap('tab20' if len(marker_ids) > 10 else 'tab10')
    marker_colors = {
        mid_id: cmap_m(i % cmap_m.N)
        for i, mid_id in enumerate(marker_ids)
    }
    marker_plots = {
        mid_id: ax.plot([], [], [], 'o',
                        color=marker_colors[mid_id],
                        markersize=4, linestyle='',
                        label=f'ID {int(mid_id)}')[0]
        for mid_id in marker_ids
    }

    # ---------- 仮想関節描画オブジェクト ----------
    joint_plots = {}
    for jname in joint_names:
        color = JOINT_COLORS.get(jname, 'black')
        plot, = ax.plot([], [], [], 'o',
                        color=color,
                        markersize=JOINT_MARKER_SIZE,
                        linestyle='',
                        label=f'[Joint] {jname}',
                        zorder=10)
        joint_plots[jname] = plot

    # ---------- ボーン線描画オブジェクト ----------
    bone_line, = ax.plot([], [], [], '-',
                         color='black', linewidth=3,
                         label='Bone', zorder=9)

    # ---------- ゴム線描画オブジェクト (あれば) ----------
    rubber_lines = {
        name: ax.plot([], [], [], '-', color='gray',
                      linewidth=2, alpha=0.6)[0]
        for name in lines_def
    }

    # ---------- テキスト / 凡例 ----------
    title_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes,
                           fontsize=12, va='top')
    pause_text = ax.text2D(0.50, 0.97, '',
                           transform=ax.transAxes, fontsize=10,
                           color='red', ha='center', va='top')

    # マーカー凡例 (サイドに小さく)
    marker_handles = list(marker_plots.values())
    joint_handles  = list(joint_plots.values()) + [bone_line]
    leg1 = ax.legend(handles=joint_handles,
                     loc='upper right', fontsize=8,
                     title='Virtual Joints')
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles,
              loc='lower right', fontsize=6,
              title='Markers', ncol=2)

    # ---------- アニメーション状態 ----------
    state = {'paused': False, 'frame_idx': 0}

    def _get_positions(frame_val):
        """指定フレームのマーカー座標辞書を返す。"""
        fdf = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == frame_val]
        return {
            int(row['id']): np.array([row['x'], row['y'], row['z']])
            for _, row in fdf.iterrows()
        }

    def _get_joint_positions(frame_val):
        """df_joints から指定フレームの関節座標辞書を返す。"""
        row = df_joints[df_joints['gait_cycle_%'] == frame_val]
        if row.empty:
            return {}
        row = row.iloc[0]
        jpos = {}
        for jname in joint_names:
            xc = row.get(f'{jname}_x', np.nan)
            yc = row.get(f'{jname}_y', np.nan)
            zc = row.get(f'{jname}_z', np.nan)
            if not (np.isnan(xc) or np.isnan(yc) or np.isnan(zc)):
                jpos[jname] = np.array([xc, yc, zc])
        return jpos

    def _update(frame_val):
        positions = _get_positions(frame_val)
        jpos      = _get_joint_positions(frame_val)

        # --- マーカー更新 ---
        for mid_id, plot in marker_plots.items():
            if mid_id in positions:
                p = positions[mid_id]
                plot.set_data([p[0]], [p[1]])
                plot.set_3d_properties([p[2]])
            else:
                plot.set_data([], [])
                plot.set_3d_properties([])

        # --- 仮想関節更新 ---
        for jname, plot in joint_plots.items():
            if jname in jpos:
                p = jpos[jname]
                plot.set_data([p[0]], [p[1]])
                plot.set_3d_properties([p[2]])
            else:
                plot.set_data([], [])
                plot.set_3d_properties([])

        # --- ボーン線更新 (BONE_ORDER に従って関節を線で繋ぐ) ---
        bx, by, bz = [], [], []
        for jname in BONE_ORDER:
            if jname in jpos:
                bx.append(jpos[jname][0])
                by.append(jpos[jname][1])
                bz.append(jpos[jname][2])
        if len(bx) >= 2:
            bone_line.set_data(bx, by)
            bone_line.set_3d_properties(bz)
        else:
            bone_line.set_data([], [])
            bone_line.set_3d_properties([])

        # --- ゴム線更新 ---
        for seg_name, (p1_id, p2_id) in lines_def.items():
            line = rubber_lines.get(seg_name)
            if line and p1_id in positions and p2_id in positions:
                p1, p2 = positions[p1_id], positions[p2_id]
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                line.set_3d_properties([p1[2], p2[2]])
            elif line:
                line.set_data([], [])
                line.set_3d_properties([])

        # --- タイトル更新 ---
        title_text.set_text(
            f'{task_key} | Phase{phase} ({speed}m/s) | '
            f'Gait Cycle: {frame_val:.1f}%'
        )
        return (list(marker_plots.values()) +
                list(joint_plots.values()) +
                [bone_line] +
                list(rubber_lines.values()) +
                [title_text])

    # ---------- キーボード操作 ----------
    def _on_key(event):
        if event.key == ' ':
            state['paused'] = not state['paused']
            pause_text.set_text('■ PAUSED  [Space]=resume  [←→]=step'
                                if state['paused'] else '')
            fig.canvas.draw_idle()
        elif event.key == 'right' and state['paused']:
            state['frame_idx'] = (state['frame_idx'] + 1) % n_frames
            _update(frames[state['frame_idx']])
            fig.canvas.draw_idle()
        elif event.key == 'left' and state['paused']:
            state['frame_idx'] = (state['frame_idx'] - 1) % n_frames
            _update(frames[state['frame_idx']])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', _on_key)

    # ---------- アニメーション開始 ----------
    def _anim_update(i):
        if state['paused']:
            return (list(marker_plots.values()) +
                    list(joint_plots.values()) +
                    [bone_line] +
                    list(rubber_lines.values()) +
                    [title_text, pause_text])
        state['frame_idx'] = i
        return _update(frames[i])

    ani = animation.FuncAnimation(
        fig, _anim_update,
        frames=n_frames,
        interval=80,   # ms/frame (約12fps)
        blit=False,
    )

    print("\n操作方法:")
    print("  スペースキー : 一時停止 / 再開")
    print("  ← → キー   : 一時停止中にフレーム単位で手動送り")
    print("  ウィンドウを閉じると終了します。")
    plt.tight_layout()
    plt.show()
    plt.close(fig)