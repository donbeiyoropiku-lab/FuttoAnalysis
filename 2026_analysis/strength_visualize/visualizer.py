# =============================================================================
# strength_visualize/visualizer.py
#
# 役割:
#   3Dアニメーション・静止マップ・張力グラフの描画処理を集約する。
#   TensionVisualizer クラスが描画に関するすべてのメソッドを持つ。
#
# ★ 視点角度・カラーマップ・マーカーサイズなどの描画パラメータは
#    ここだけ修正すればよい。
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D投影に必要)
import matplotlib.animation as animation

from futto_common import CONFIG as config
from .tension_calc import calculate_indicator_position, compute_segment_tension_bounds


class TensionVisualizer:
    """張力と筋活動の可視化を担うクラス。"""

    def __init__(self, strain_to_force_interp, emg_data, max_emg_vals, muscle_indicators_def):
        """
        Parameters
        ----------
        strain_to_force_interp : scipy.interpolate.interp1d or None
            io_utils.load_rubber_properties() の戻り値。
        emg_data : pd.DataFrame or None
            io_utils.load_emg_csv() の戻り値[0]。
        max_emg_vals : dict
            io_utils.load_emg_csv() の戻り値[1]。
        muscle_indicators_def : dict
            CONFIG の MUSCLE_INDICATORS に相当する辞書。
        """
        self.strain_to_force_interp = strain_to_force_interp
        self.emg_data               = emg_data
        self.max_emg_vals           = max_emg_vals
        self.muscle_indicators_def  = muscle_indicators_def
        self.paused                 = False

    # ------------------------------------------------------------------
    # アニメーション
    # ------------------------------------------------------------------

    def _init_animation(self, fig, ax, df_mean_cycle, lines_to_draw_def,
                        hide_legends=False):
        """アニメーション用のプロットオブジェクトを初期化する。

        Parameters
        ----------
        hide_legends : bool
            True のとき、Markers凡例・Muscles凡例・カラーバーをすべて非表示にする。
            GIF保存時に使用する。
        """
        self.paused = False

        marker_ids   = sorted(df_mean_cycle['id'].unique())
        cmap_markers = plt.get_cmap('tab20' if len(marker_ids) > 10 else 'tab10')
        marker_colors = {
            mid: cmap_markers(i % cmap_markers.N)
            for i, mid in enumerate(marker_ids)
        }
        plots_marker = {
            mid: ax.plot([], [], [], marker='o', color=marker_colors[mid],
                         markersize=5, linestyle='', label=f'ID {int(mid)}')[0]
            for mid in marker_ids
        }

        lines_rubber = {
            name: ax.plot([], [], [], linewidth=4)[0]
            for name in lines_to_draw_def.keys()
        }

        plots_muscle   = {}
        muscle_handles = []
        cmap_muscle_fixed = plt.get_cmap('gist_rainbow')
        num_muscles = len(self.muscle_indicators_def)

        for i, (name, _) in enumerate(self.muscle_indicators_def.items()):
            color = cmap_muscle_fixed(i / max(1, num_muscles - 1))
            plot, = ax.plot(
                [], [], [], marker='o',
                markersize=getattr(config, 'MUSCLE_MARKER_BASE_SIZE', 5),
                linestyle='', label=name, color=color
            )
            plots_muscle[name] = plot
            muscle_handles.append(plot)

        all_x = df_mean_cycle['x'].values
        all_y = df_mean_cycle['y'].values
        all_z = df_mean_cycle['z'].values
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min(),
        ]).max() * 1.1
        mid_x = (all_x.max() + all_x.min()) / 2
        mid_y = (all_y.max() + all_y.min()) / 2
        mid_z = (all_z.max() + all_z.min()) / 2

        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("Gait Cycle: Rubber Tension & Muscle Activity", fontsize=16)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=30, azim=60)

        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        if hide_legends:
            # GIF保存時: 凡例・カラーバーをすべて非表示
            ax.legend_ = None
        else:
            # 通常表示時: Markers凡例・Muscles凡例を両方表示
            marker_legend = ax.legend(
                loc='upper left', bbox_to_anchor=(0.01, 0.9), fontsize=8, title="Markers"
            )
            ax.legend(
                handles=muscle_handles, title="Muscles",
                loc='upper right', bbox_to_anchor=(0.99, 0.9), fontsize=8
            )
            ax.add_artist(marker_legend)

        cmap_tension = plt.get_cmap('plasma')
        norm_tension = plt.Normalize(vmin=0, vmax=100)

        if not hide_legends:
            sm_tension   = plt.cm.ScalarMappable(cmap=cmap_tension, norm=norm_tension)
            cbar_tension = fig.colorbar(sm_tension, ax=ax, shrink=0.5, aspect=10,
                                        pad=0.01, location='left')
            cbar_tension.set_label('Rubber Relative Tension (%)')

        def on_key_press(event):
            if event.key == ' ':
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        return plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension

    def _update_animation(self, frame_value, df_mean_cycle, tension_data, lines_to_draw_def,
                          plots_marker, lines_rubber, plots_muscle, time_text,
                          cmap_tension, segment_tension_bounds):
        """アニメーションの各フレームを更新する。"""
        if getattr(self, 'paused', False):
            return (list(plots_marker.values()) + list(lines_rubber.values())
                    + list(plots_muscle.values()) + [time_text])

        current_df = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == frame_value]
        current_positions = {
            int(row.id): np.array([row.x, row.y, row.z])
            for _, row in current_df.iterrows()
        }

        frames = sorted(df_mean_cycle['gait_cycle_%'].unique())
        idx    = frames.index(frame_value) if frame_value in frames else 0

        try:
            current_emg = (self.emg_data.iloc[idx]
                           if self.emg_data is not None else pd.Series(0))
        except Exception:
            current_emg = pd.Series(0)

        # マーカー更新
        for marker_id, plot in plots_marker.items():
            if marker_id in current_positions:
                pos = current_positions[marker_id]
                plot.set_data([pos[0]], [pos[1]])
                plot.set_3d_properties([pos[2]])

        # ゴム線更新
        for name, (id1, id2) in lines_to_draw_def.items():
            line = lines_rubber.get(name)
            if (line and id1 in current_positions and id2 in current_positions
                    and tension_data and name in tension_data):
                pos1, pos2 = current_positions[id1], current_positions[id2]
                tension = tension_data[name][idx]
                color   = 'gray'
                if name in segment_tension_bounds:
                    bounds    = segment_tension_bounds[name]
                    range_val = bounds["max"] - bounds["min"]
                    relative_tension = np.clip(
                        (tension - bounds["min"]) / (range_val + 1e-9), 0, 1
                    )
                    color = cmap_tension(relative_tension)
                line.set_color(color)
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])
            elif line:
                line.set_data([], [])
                line.set_3d_properties([])

        # 筋肉マーカー更新
        for name, muscle_info in self.muscle_indicators_def.items():
            plot = plots_muscle.get(name)
            if plot:
                muscle_pos = calculate_indicator_position(muscle_info, current_positions)
                if muscle_pos is not None and not np.isnan(muscle_pos).any():
                    emg_col     = muscle_info.get('emg_col') or muscle_info.get('emg')
                    emg_activity = 0
                    if (emg_col and self.emg_data is not None
                            and emg_col in self.emg_data.columns):
                        val     = current_emg.get(emg_col, 0)
                        max_val = self.max_emg_vals.get(name, 1.0)
                        if max_val == 0:
                            max_val = 1.0
                        emg_activity = min(1.0, max(0.0, val / max_val))

                    plot.set_data([muscle_pos[0]], [muscle_pos[1]])
                    plot.set_3d_properties([muscle_pos[2]])

                    marker_base  = getattr(config, 'MUSCLE_MARKER_BASE_SIZE', 5)
                    marker_scale = getattr(config, 'MUSCLE_MARKER_SCALE_FACTOR', 25)
                    plot.set_markersize(marker_base + emg_activity * marker_scale)
                    plot.set_visible(True)
                else:
                    plot.set_visible(False)

        time_text.set_text(f'Gait Cycle: {frame_value:.1f} %')
        return (list(plots_marker.values()) + list(lines_rubber.values())
                + list(plots_muscle.values()) + [time_text])

    def run_animation(self, df_mean_cycle, tension_data, lines_to_draw_def,
                      show=True, save_path=None):
        """
        アニメーションを表示または保存する。

        Parameters
        ----------
        df_mean_cycle : pd.DataFrame
        tension_data : dict
        lines_to_draw_def : dict
        show : bool
            True のとき plt.show() を呼ぶ。
        save_path : str or None
            指定時はGIFとして保存する。
        """
        if self.emg_data is None:
            print("EMGデータが読み込めなかったため、アニメーションを続行できません。")
            return

        fig = plt.figure(figsize=(14, 9))
        ax  = fig.add_subplot(111, projection='3d')

        segment_tension_bounds = compute_segment_tension_bounds(tension_data) if tension_data else {}

        # save_path 指定時（GIF保存）は凡例・カラーバーをすべて非表示にする
        hide_legends = save_path is not None

        plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension = \
            self._init_animation(fig, ax, df_mean_cycle, lines_to_draw_def,
                                 hide_legends=hide_legends)

        frames = sorted(df_mean_cycle['gait_cycle_%'].unique())

        def update(frame_val):
            return self._update_animation(
                frame_val, df_mean_cycle, tension_data, lines_to_draw_def,
                plots_marker, lines_rubber, plots_muscle, time_text,
                cmap_tension, segment_tension_bounds
            )

        ani = animation.FuncAnimation(
            fig, update, frames=frames,
            init_func=lambda: (list(plots_marker.values())
                               + list(lines_rubber.values())
                               + list(plots_muscle.values())
                               + [time_text]),
            blit=False,
            interval=1000 / getattr(config, 'FRAME_RATE', 100)
        )

        if save_path:
            try:
                writer = animation.PillowWriter(
                    fps=getattr(config, 'FRAME_RATE', 100) // 4
                )
                print(f"アニメーションを保存中: {save_path} ... (時間がかかる場合があります)")
                ani.save(save_path, writer=writer)
                print("アニメーションの保存が完了しました。")
            except Exception as e:
                print(f"アニメーション保存エラー: {e}")
            finally:
                plt.close(fig)
        elif show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # グラフ出力
    # ------------------------------------------------------------------

    def plot_segment_tensions(self, tension_data, seg_groups,
                               save_dir=None, task_name="", speed=""):
        """
        セグメントグループ別に張力グラフを表示・保存する。

        Parameters
        ----------
        tension_data : dict
        seg_groups : dict
            CONFIG の SEGMENT_GROUPS に相当する辞書。
        save_dir : str or None
        task_name : str
        speed : str
        """
        if not tension_data:
            return
        x_axis = np.linspace(0, 100, len(next(iter(tension_data.values()))))

        for group_name, lines in seg_groups.items():
            plt.figure(figsize=(10, 5))
            has_data = False
            for line_name in lines:
                if line_name in tension_data:
                    plt.plot(x_axis, tension_data[line_name],
                             label=line_name, linewidth=2)
                    has_data = True

            if has_data:
                plt.title(f"Tension - {group_name}")
                plt.xlabel("Gait Cycle [%]")
                plt.ylabel("Tension [N]")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()

                # 自動保存処理 (ファイル名: taskXX_0.7_FK.png)
                if save_dir:
                    safe_group_name = group_name.replace("/", "_").replace(" ", "_")
                    save_path = os.path.join(
                        save_dir, f"{task_name}_{speed}_{safe_group_name}.png"
                    )
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"✓ グラフを保存しました: {save_path}")

                plt.show()
            else:
                plt.close()

    # ------------------------------------------------------------------
    # 静止3Dマップ
    # ------------------------------------------------------------------

    def show_static_3d_maps(self, df_mean_cycle, tension_data, lines_to_draw_def,
                             save_dir=None, task_name="", speed="", show=False):
        """
        10%刻みの静止3Dマップを表示・保存する。

        Parameters
        ----------
        df_mean_cycle : pd.DataFrame
        tension_data : dict
        lines_to_draw_def : dict
        save_dir : str or None
        task_name : str
        speed : str
        show : bool
            True の場合は画面に表示し、False の場合は表示せずに図を閉じる。
        """
        phases_to_show = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        frames = sorted(df_mean_cycle['gait_cycle_%'].unique())

        segment_tension_bounds = compute_segment_tension_bounds(tension_data) if tension_data else {}

        all_x = df_mean_cycle['x'].values
        all_y = df_mean_cycle['y'].values
        all_z = df_mean_cycle['z'].values
        max_range = np.array([
            all_x.max() - all_x.min(),
            all_y.max() - all_y.min(),
            all_z.max() - all_z.min(),
        ]).max() * 1.1
        mid_x = (all_x.max() + all_x.min()) / 2
        mid_y = (all_y.max() + all_y.min()) / 2
        mid_z = (all_z.max() + all_z.min()) / 2

        marker_ids    = sorted(df_mean_cycle['id'].unique())
        cmap_markers  = plt.get_cmap('tab20' if len(marker_ids) > 10 else 'tab10')
        marker_colors = {
            mid: cmap_markers(i % cmap_markers.N)
            for i, mid in enumerate(marker_ids)
        }

        cmap_tension = plt.get_cmap('plasma')
        norm_tension = plt.Normalize(vmin=0, vmax=100)

        cmap_muscle_fixed = plt.get_cmap('gist_rainbow')
        num_muscles = len(self.muscle_indicators_def)
        muscle_colors = {
            name: cmap_muscle_fixed(i / max(1, num_muscles - 1))
            for i, name in enumerate(self.muscle_indicators_def.keys())
        }

        for target_phase in phases_to_show:
            actual_frame = min(frames, key=lambda x: abs(x - target_phase))
            idx          = frames.index(actual_frame)

            fig = plt.figure(figsize=(10, 10))
            ax  = fig.add_subplot(111, projection='3d')

            ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
            ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
            ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
            ax.set_box_aspect([1, 1, 1])
            ax.set_title(f"Gait Cycle: {actual_frame:.1f} %", fontsize=16)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')

            # ★ 画角の設定: x軸が右から左に正、水平になり、
            #    そこからZ軸周りに反時計回りに30度回転した視点
            ax.view_init(elev=30, azim=60)

            current_df = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == actual_frame]
            current_positions = {
                int(row.id): np.array([row.x, row.y, row.z])
                for _, row in current_df.iterrows()
            }

            try:
                current_emg = (self.emg_data.iloc[idx]
                               if self.emg_data is not None else pd.Series(0))
            except Exception:
                current_emg = pd.Series(0)

            # マーカー描画
            for marker_id in marker_ids:
                if marker_id in current_positions:
                    pos = current_positions[marker_id]
                    ax.plot([pos[0]], [pos[1]], [pos[2]], marker='o',
                            color=marker_colors[marker_id], markersize=5, linestyle='')

            # ゴム線描画
            for name, (id1, id2) in lines_to_draw_def.items():
                if (id1 in current_positions and id2 in current_positions
                        and tension_data and name in tension_data):
                    pos1, pos2 = current_positions[id1], current_positions[id2]
                    tension    = tension_data[name][idx]
                    color      = 'gray'
                    if name in segment_tension_bounds:
                        bounds    = segment_tension_bounds[name]
                        range_val = bounds["max"] - bounds["min"]
                        relative_tension = np.clip(
                            (tension - bounds["min"]) / (range_val + 1e-9), 0, 1
                        )
                        color = cmap_tension(relative_tension)
                    ax.plot(
                        [pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                        color=color, linewidth=4
                    )

            # 筋肉マーカー描画
            for name, muscle_info in self.muscle_indicators_def.items():
                muscle_pos = calculate_indicator_position(muscle_info, current_positions)
                if muscle_pos is not None and not np.isnan(muscle_pos).any():
                    emg_col     = muscle_info.get('emg_col') or muscle_info.get('emg')
                    emg_activity = 0
                    if (emg_col and self.emg_data is not None
                            and emg_col in self.emg_data.columns):
                        val     = current_emg.get(emg_col, 0)
                        max_val = self.max_emg_vals.get(name, 1.0)
                        if max_val == 0:
                            max_val = 1.0
                        emg_activity = min(1.0, max(0.0, val / max_val))

                    marker_base  = getattr(config, 'MUSCLE_MARKER_BASE_SIZE', 5)
                    marker_scale = getattr(config, 'MUSCLE_MARKER_SCALE_FACTOR', 25)
                    marker_size  = marker_base + emg_activity * marker_scale
                    color        = muscle_colors[name]

                    ax.plot([muscle_pos[0]], [muscle_pos[1]], [muscle_pos[2]],
                            marker='o', markersize=marker_size, color=color, linestyle='')

            sm_tension   = plt.cm.ScalarMappable(cmap=cmap_tension, norm=norm_tension)
            cbar_tension = fig.colorbar(sm_tension, ax=ax, shrink=0.5, aspect=10,
                                        pad=0.01, location='left')
            cbar_tension.set_label('Rubber Relative Tension (%)')

            plt.tight_layout()

            # 自動保存処理 (ファイル名: taskXX_0.7_10.png)
            if save_dir:
                save_path = os.path.join(
                    save_dir, f"{task_name}_{speed}_{int(actual_frame)}.png"
                )
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"✓ 3Dマップを保存しました: {save_path}")

            if show:
                print(f"[{actual_frame:.1f}%] の静止マップを表示中..."
                      " (ウィンドウを閉じると次のマップが表示されます)")
                plt.show()
            else:
                plt.close(fig)