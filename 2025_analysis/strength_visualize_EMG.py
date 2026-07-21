'''
* **目的**: 平均化された歩行周期データとゴムの物性データから、各ゴム部分にかかる張力を計算し、アニメーションとグラフで可視化します。
  * **入力**:
      * `task1_mean_cycle.csv` (Step 2の出力)
      * `rubber_strength.xlsx` (ゴムの物性データ)
      * "Subject *_task04_Phase*_average.csv"(筋電のデータ)
  * **使用スクリプト**: `strength_visualize.py`
  * **出力**:
      * 張力を色で表現した3Dアニメーション表示
      * 部位ごとの張力変化のグラフ表示
      * 筋活動も同時に表示
      *csvにデータを保存
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import seaborn as sns
import os
import config # ★ 設定ファイルをインポート

class TensionVisualizer:
    """
    張力と筋活動の計算および可視化を行うクラス。
    """
    def __init__(self, excel_path, sheet_name, emg_csv_path, muscle_indicators_def):
        """
        コンストラクタ。各種データの読み込みと前処理を行う。
        """
        self.strain_to_force_interp = self._create_strain_force_interpolator(excel_path, sheet_name)
        self.emg_data = self._load_and_process_emg(emg_csv_path, muscle_indicators_def)
        self.muscle_indicators_def = muscle_indicators_def
        self.paused = False # アニメーション一時停止用フラグ

    # --- データ読み込み・前処理 ---
    def _create_strain_force_interpolator(self, excel_path, sheet_name):
        """Excelからひずみ-荷重データを読み込み、補間関数を作成する。"""
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=3)
        except Exception as e:
            print(f"Excel読み込みエラー: {e}"); return None
        required_cols = ['ひずみ', '荷重(N)']
        if not all(col in df.columns for col in required_cols):
            print(f"Excelエラー: 必要な列が見つかりません。 ('ひずみ', '荷重(N)')"); return None
        df = df[df['ひずみ'] >= 0].sort_values('ひずみ').dropna(subset=required_cols)
        if df.empty or len(df) < 2:
            print("Excelエラー: 有効なデータ点が不足しています。"); return None
        print("ひずみ-荷重関係の作成に成功しました。")
        return interp1d(df['ひずみ'], df['荷重(N)'], kind='linear', bounds_error=False,
                        fill_value=(df['荷重(N)'].iloc[0], df['荷重(N)'].iloc[-1]))

    def _load_and_process_emg(self, emg_csv_path, muscle_indicators_def):
        """EMGデータを読み込み、正規化する。"""
        try:
            df_emg = pd.read_csv(emg_csv_path)
            print(f"EMGデータの読み込み成功: {emg_csv_path}")
        except Exception as e:
            print(f"EMG CSV読み込みエラー: {e}"); return None
        if 'GaitCycle_%' not in df_emg.columns:
            print("EMGエラー: 'GaitCycle_%' 列が見つかりません。"); return None

        df_emg = df_emg.set_index('GaitCycle_%')
        emg_columns = [indicator['emg_col'] for indicator in muscle_indicators_def.values()]
        missing_cols = [col for col in emg_columns if col not in df_emg.columns]
        if missing_cols:
            print(f"EMGエラー: 必要な列が見つかりません: {missing_cols}"); return None

        df_emg_selected = df_emg[emg_columns].copy()
        # 各列をその列の最大値で正規化 (0-1スケール)
        df_emg_normalized = df_emg_selected.apply(lambda x: x / x.max() if x.max() > 1e-9 else x, axis=0) # ゼロ除算を回避
        print("EMGデータを処理・正規化しました (最大値に基づく0-1スケール)。")
        return df_emg_normalized

    # --- 計算 ---
    def calculate_all_tensions(self, df_mean_cycle, natural_lengths_map, lines_to_draw_def):
        """歩行周期データから各ゴム部分の張力を計算する。"""
        if self.strain_to_force_interp is None: return None, None
        tension_data = {} # アニメーション/グラフ用 {name: pd.Series}
        csv_records = []  # CSV出力用 List[Dict]
        gait_cycle_perc = sorted(df_mean_cycle['gait_cycle_%'].unique())

        for name, (p1, p2) in lines_to_draw_def.items():
            natural_length = natural_lengths_map.get(name)
            if natural_length is None: continue

            df1 = df_mean_cycle[df_mean_cycle['id'] == p1].sort_values('gait_cycle_%')
            df2 = df_mean_cycle[df_mean_cycle['id'] == p2].sort_values('gait_cycle_%')

            if not df1.empty and len(df1) == len(df2):
                coords1 = df1[['x', 'y', 'z']].values
                coords2 = df2[['x', 'y', 'z']].values
                current_lengths = np.linalg.norm(coords1 - coords2, axis=1)
                strains = (current_lengths - natural_length) / natural_length
                forces = self.strain_to_force_interp(strains)
                tension_data[name] = pd.Series(forces, index=gait_cycle_perc)

                # CSV用データ作成
                for cycle, cur_len, strain, force in zip(gait_cycle_perc, current_lengths, strains, forces):
                    csv_records.append({
                        "gait_cycle_%": cycle, "segment": name, "start_marker_id": p1, "end_marker_id": p2,
                        "natural_length_mm": natural_length, "current_length_mm": cur_len,
                        "strain": strain, "tension_N": force
                    })
            else:
                print(f"警告: 張力計算スキップ ({name})。マーカーデータ不整合 (p1:{len(df1)}, p2:{len(df2)})")


        tension_df_for_csv = pd.DataFrame(csv_records)
        print("全セグメントの張力計算完了。")
        return tension_data, tension_df_for_csv
    
    # --- ▼▼▼【関数修正】'weighted_midpoint' を追加 ▼▼▼ ---
    def calculate_indicator_position(self, muscle_info, current_positions):
        """定義に基づき、筋肉インジケータの3D座標を計算する。"""
        m_type = muscle_info['type']
        m_ids = muscle_info['markers']
        points = [np.array(current_positions[mid]) for mid in m_ids if mid in current_positions]

        if len(points) != len(m_ids): return None # 必要なマーカーが不足

        if m_type == 'single': return points[0]
        if m_type == 'midpoint': return np.mean(points[:2], axis=0) # 通常の中点
        if m_type == 'centroid': return np.mean(points, axis=0)
        # --- ▼▼▼ 追加 ▼▼▼ ---
        if m_type == 'weighted_midpoint':
            if len(points) >= 2:
                weight_p1 = muscle_info.get('weight', 0.5) # 2番目の点の重み (デフォルト0.5=中点)
                weight_p0 = 1.0 - weight_p1
                return points[0] * weight_p0 + points[1] * weight_p1 # 重み付き平均
        # --- ▲▲▲ 追加 ▲▲▲ ---
        if m_type == 'offset':
            base_pos = np.mean(points, axis=0)
            ref_points = [np.array(current_positions[ref_id]) for ref_id in muscle_info.get('ref_marker', []) if ref_id in current_positions]
            if len(ref_points) != len(muscle_info.get('ref_marker', [])): return base_pos # refマーカー不足
            ref_pos = np.mean(ref_points, axis=0)
            weight = muscle_info.get('weight', 0.1)
            return base_pos + weight * (ref_pos - base_pos)
        return None # 不明なタイプ
    # --- ▲▲▲ 関数修正ここまで ▲▲▲ ---
    
    # --- 可視化 ---
    def _init_animation(self, fig, ax, df_mean_cycle, lines_to_draw_def):
        """アニメーションの初期設定 (軸範囲、プロットオブジェクト作成など)"""
        # マーカープロットオブジェクト
        marker_ids = sorted(df_mean_cycle['id'].unique())
        cmap_markers = plt.cm.get_cmap('tab20' if len(marker_ids) > 10 else 'tab10', len(marker_ids))
        marker_colors = {mid: cmap_markers(i) for i, mid in enumerate(marker_ids)}
        plots_marker = {mid: ax.plot([], [], [], marker='o', color=marker_colors[mid], markersize=5, linestyle='', label=f'ID {mid}')[0] for mid in marker_ids}

        # ゴム線プロットオブジェクト
        lines_rubber = {name: ax.plot([], [], [], linewidth=4)[0] for name in lines_to_draw_def.keys()}

        # 筋肉インジケータプロットオブジェクト
        plots_muscle = {}
        muscle_handles = []
        cmap_muscle_fixed = plt.cm.get_cmap('gist_rainbow', len(self.muscle_indicators_def))
        for i, (name, _) in enumerate(self.muscle_indicators_def.items()):
            color = cmap_muscle_fixed(i)
            plot, = ax.plot([], [], [], marker='o', markersize=config.MUSCLE_MARKER_BASE_SIZE, linestyle='', label=name, color=color)
            plots_muscle[name] = plot
            muscle_handles.append(plot)

        # 軸範囲とラベル設定
        all_x, all_y, all_z = df_mean_cycle['x'].values, df_mean_cycle['y'].values, df_mean_cycle['z'].values
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() * 1.1
        mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
        ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2); ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2); ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)
        ax.set_box_aspect([1,1,1]); ax.set_title("Gait Cycle: Rubber Tension & Muscle Activity", fontsize=16)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')

        # 仰角30度、方位角30度 (デフォルト-60から+90度回転)
        ax.view_init(elev=30, azim=-60)

        # テキストと凡例
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        marker_legend = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.9), fontsize=8, title="Markers")
        muscle_legend = ax.legend(handles=muscle_handles, title="Muscles", loc='upper right', bbox_to_anchor=(0.99, 0.9), fontsize=8)
        ax.add_artist(marker_legend) # 凡例が重ならないように

        # カラーバー (ゴム張力)
        cmap_tension = plt.cm.get_cmap('plasma'); norm_tension = plt.Normalize(vmin=0, vmax=100)
        sm_tension = plt.cm.ScalarMappable(cmap=cmap_tension, norm=norm_tension)
        cbar_tension = fig.colorbar(sm_tension, ax=ax, shrink=0.5, aspect=10, pad=0.01, location='left')
        cbar_tension.set_label('Rubber Relative Tension (%)')

        # 一時停止イベントリスナー
        def on_key_press(event):
            if event.key == ' ': self.paused = not self.paused; print("Paused" if self.paused else "Resumed")
        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # update関数で使うオブジェクトを返す
        return plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension

    def _update_animation(self, frame_value, df_mean_cycle, tension_data, lines_to_draw_def, # 引数追加
                          plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension, segment_tension_bounds): # 引数追加
        """アニメーションの各フレームを更新する"""
        if self.paused:
            return list(plots_marker.values()) + list(lines_rubber.values()) + list(plots_muscle.values())

        current_data = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == frame_value]
        positions = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()} # IDをintに

        try:
            current_emg = self.emg_data.loc[frame_value]
        except KeyError:
             current_emg = pd.Series(0, index=self.emg_data.columns)

        # マーカー更新
        for marker_id, plot in plots_marker.items():
            if marker_id in positions:
                x, y, z = positions[marker_id]
                plot.set_data([x], [y]); plot.set_3d_properties([z])

        # ゴム線更新
        for name, (p1, p2) in lines_to_draw_def.items():
            line = lines_rubber.get(name)
            if line and p1 in positions and p2 in positions and name in tension_data:
                pos1, pos2 = np.array(positions[p1]), np.array(positions[p2])
                tension = tension_data[name].get(frame_value, 0)
                color = 'gray' # デフォルト色
                if name in segment_tension_bounds:
                    bounds = segment_tension_bounds[name]
                    # ゼロ除算を回避しつつ正規化
                    range_val = bounds["max"] - bounds["min"]
                    relative_tension = np.clip((tension - bounds["min"]) / (range_val + 1e-9), 0, 1)
                    color = cmap_tension(relative_tension)
                line.set_color(color)
                line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                line.set_3d_properties([pos1[2], pos2[2]])

        # 筋肉インジケータ更新
        for name, muscle_info in self.muscle_indicators_def.items():
            plot = plots_muscle.get(name)
            if plot:
                muscle_pos = self.calculate_indicator_position(muscle_info, positions)
                if muscle_pos is not None and not np.isnan(muscle_pos).any():
                    emg_col = muscle_info['emg_col']
                    emg_activity = current_emg.get(emg_col, 0) # 正規化済み(0-1)
                    plot.set_data([muscle_pos[0]], [muscle_pos[1]])
                    plot.set_3d_properties([muscle_pos[2]])
                    marker_size = config.MUSCLE_MARKER_BASE_SIZE + emg_activity * config.MUSCLE_MARKER_SCALE_FACTOR
                    plot.set_markersize(marker_size)
                    plot.set_visible(True)
                else:
                    plot.set_visible(False)

        time_text.set_text(f'Gait Cycle: {frame_value:.1f} %')
        return list(plots_marker.values()) + list(lines_rubber.values()) + list(plots_muscle.values())

    def run_animation(self, df_mean_cycle, tension_data, lines_to_draw_def, show=True, save_path=None):
        """アニメーションを実行または保存する"""
        if self.emg_data is None: print("EMG Error."); return

        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')

        # ゴム張力の最小/最大値を計算 (updateで使うため先に計算)
        segment_tension_bounds = {}
        for name, tension_series in tension_data.items():
            if not tension_series.dropna().empty:
                min_val, max_val = tension_series.min(), tension_series.max()
                segment_tension_bounds[name] = {"min": min_val, "max": max_val}

        # アニメーション要素の初期化
        plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension = self._init_animation(
            fig, ax, df_mean_cycle, lines_to_draw_def
        )

        # FuncAnimationの呼び出し
        ani = animation.FuncAnimation(
            fig,
            self._update_animation, # update関数
            frames=sorted(df_mean_cycle['gait_cycle_%'].unique()),
            # update関数に追加で渡す引数
            fargs=(df_mean_cycle, tension_data, lines_to_draw_def,
                   plots_marker, lines_rubber, plots_muscle, time_text, cmap_tension, segment_tension_bounds),
            init_func=lambda: list(plots_marker.values()) + list(lines_rubber.values()) + list(plots_muscle.values()), # init関数は単純化
            blit=False, # blit=True は3Dプロットで問題を起こすことがある
            interval=1000/config.FRAME_RATE
            
        )

        # 表示または保存
        if save_path:
            try:
                writer = animation.PillowWriter(fps=config.FRAME_RATE // 4)
                print(f"アニメーションを保存中: {save_path} ... (時間がかかる場合があります)")
                ani.save(save_path, writer=writer)
                print("アニメーションの保存が完了しました。")
            except Exception as e:
                print(f"アニメーション保存エラー: {e}")
                print("Pillowがインストールされているか確認してください (`pip install Pillow`)。")
            finally:
                plt.close(fig) # 保存後は図を閉じる
        elif show:
            plt.show() # 通常通り表示
        else:
            plt.close(fig) # 表示も保存もしない場合は閉じる

    def calculate_global_tension_bounds(self, margin_ratio: float = 0.05) -> dict:
        """
        全タスクの平均化データから張力を計算し、SEGMENT_GROUPSごとの軸範囲(最小・最大)を算出する。
        タスク間でグラフの軸スケールを揃えて比較しやすくするために使用する。
        """
        if self.strain_to_force_interp is None:
            return {}

        seg_groups = config.SEGMENT_GROUPS
        raw_bounds = {name: [np.inf, -np.inf] for name in seg_groups}

        for other_cfg in config.TASK_CONFIGS.values():
            mean_csv_path = other_cfg.get('MEAN_CYCLE_OUTPUT_PATH')
            if not mean_csv_path or not os.path.exists(mean_csv_path):
                continue
            try:
                df_mean_cycle = pd.read_csv(mean_csv_path)
            except Exception:
                continue

            other_tension_data, _ = self.calculate_all_tensions(
                df_mean_cycle, other_cfg.get('NATURAL_LENGTHS', {}), other_cfg.get('LINES_TO_DRAW', {})
            )
            if not other_tension_data:
                continue

            for group_name, segment_list in seg_groups.items():
                for segment_name in segment_list:
                    series = other_tension_data.get(segment_name)
                    if series is None or series.dropna().empty:
                        continue
                    raw_bounds[group_name][0] = min(raw_bounds[group_name][0], series.min())
                    raw_bounds[group_name][1] = max(raw_bounds[group_name][1], series.max())

        axis_bounds = {}
        for group_name, (lo, hi) in raw_bounds.items():
            if np.isfinite(lo) and np.isfinite(hi):
                span = hi - lo
                margin = span * margin_ratio if span > 0 else (abs(hi) * margin_ratio + 1e-6)
                axis_bounds[group_name] = (lo - margin, hi + margin)
            else:
                axis_bounds[group_name] = None
        return axis_bounds

    # Front_Hip, Back_Knee, Back_Shin は1枚の図にまとめて保存する
    COMBINED_TENSION_GROUPS = ["Front_Hip", "Back_Knee", "Back_Shin"]
    COMBINED_TENSION_FILENAME = "FHBKBS"

    @staticmethod
    def _group_abbreviation(group_name: str) -> str:
        return "".join(word[0].upper() for word in group_name.split("_"))

    def _plot_tension_group_on_ax(self, ax, group_name, lines_list, tension_data, axis_bounds):
        """1つのセグメントグループの張力を指定したAxesに描画する"""
        has_data = False
        for line_name in lines_list:
            if line_name in tension_data and not tension_data[line_name].empty:
                has_data = True
                tension_series = tension_data[line_name]
                sns.lineplot(x=tension_series.index, y=tension_series.values, label=line_name, ax=ax)

        ax.set_title(group_name)
        ax.set_xlabel("Gait Cycle [%]")
        ax.set_ylabel("Tension [N]")
        ax.grid(True)
        if has_data:
            ax.legend(fontsize=9)
            if axis_bounds and axis_bounds.get(group_name):
                ax.set_ylim(*axis_bounds[group_name])
        return has_data

    def plot_segment_tensions(self, tension_data, seg_groups, task_key, axis_bounds=None):
        """部位ごとに張力グラフを表示・保存する"""
        print("セグメントグループごとの張力グラフを生成中...")
        if not tension_data:
            print("張力データがありません。グラフをスキップします。")
            return

        # タスクごとの保存先: C:\FuttoAnalysis\result\2025\{task}\graph
        output_dir = os.path.join(config.RESULT_DIR, "2025", task_key, "graph")
        save_choice = input("\n張力グラフを画像として保存しますか？ (y/n): ").lower()
        save_plots = (save_choice == 'y')
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            print(f"画像は {output_dir} に保存されます。")

        # --- 1. Front_Hip, Back_Knee, Back_Shin を1つの軸にまとめた図 (6本の曲線) ---
        combined_groups_present = [g for g in self.COMBINED_TENSION_GROUPS if g in seg_groups]
        if combined_groups_present:
            fig, ax = plt.subplots(figsize=(12, 7))
            fig.suptitle(f'Tension Profile (Combined: {", ".join(combined_groups_present)}) - Task: {task_key}', fontsize=16)

            any_data = False
            combined_lo, combined_hi = np.inf, -np.inf
            for group_name in combined_groups_present:
                for line_name in seg_groups[group_name]:
                    if line_name in tension_data and not tension_data[line_name].empty:
                        any_data = True
                        tension_series = tension_data[line_name]
                        sns.lineplot(x=tension_series.index, y=tension_series.values, label=line_name, ax=ax)
                if axis_bounds and axis_bounds.get(group_name):
                    lo, hi = axis_bounds[group_name]
                    combined_lo, combined_hi = min(combined_lo, lo), max(combined_hi, hi)

            ax.set_xlabel("Gait Cycle [%]", fontsize=14)
            ax.set_ylabel("Tension [N]", fontsize=14)
            ax.grid(True)
            if any_data:
                ax.legend(fontsize=10)
                if np.isfinite(combined_lo) and np.isfinite(combined_hi):
                    ax.set_ylim(combined_lo, combined_hi)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if any_data:
                if save_plots:
                    try:
                        save_filename = f"{task_key}_{self.COMBINED_TENSION_FILENAME}.png"
                        save_path = os.path.join(output_dir, save_filename)
                        fig.savefig(save_path, dpi=150)
                        print(f"  -> グラフ '{self.COMBINED_TENSION_FILENAME}' を保存しました。")
                    except Exception as e:
                        print(f"  -> グラフ '{self.COMBINED_TENSION_FILENAME}' の保存エラー: {e}")
                plt.show()
            else:
                plt.close(fig)
                print(f"  グラフスキップ ({self.COMBINED_TENSION_FILENAME}): データがありません。")

        # --- 2. 残りのグループは個別の図 ---
        for group_name, lines_list in seg_groups.items():
            if group_name in self.COMBINED_TENSION_GROUPS:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(f'Tension Profile: {group_name}\nTask: {task_key}', fontsize=16)
            has_data = self._plot_tension_group_on_ax(ax, group_name, lines_list, tension_data, axis_bounds)

            if has_data:
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                if save_plots:
                    try:
                        abbr = self._group_abbreviation(group_name)
                        save_filename = f"{task_key}_{abbr}.png"
                        save_path = os.path.join(output_dir, save_filename)
                        fig.savefig(save_path, dpi=150)
                        print(f"  -> グラフ '{group_name}' を保存しました。")
                    except Exception as e:
                        print(f"  -> グラフ '{group_name}' の保存エラー: {e}")
                plt.show()
            else:
                plt.close(fig) # データがない場合は空のプロットを閉じる
                print(f"  グラフスキップ ({group_name}): データがありません。")
        print("張力グラフの生成完了。")

    # --- ▼▼▼【関数修正】静止画マップに筋肉マーカーを追加 ▼▼▼ ---
    def show_static_3d_maps(self, df_mean_cycle, tension_data, lines_to_draw_def):
        """歩行周期10%刻みで静止3Dマップを表示する (筋肉マーカー付き)"""
        print("Generating static 3D maps (0-100% at 10% steps)...")
        if not tension_data and self.emg_data is None:
            print("No tension or EMG data. Skipping static maps.")
            return

        marker_ids = sorted(df_mean_cycle['id'].unique())
        cmap_markers = plt.cm.get_cmap('tab20' if len(marker_ids) > 10 else 'tab10', len(marker_ids))
        marker_colors = {mid: cmap_markers(i) for i, mid in enumerate(marker_ids)}
        segment_tension_bounds = {}
        if tension_data: # 張力データがあれば範囲計算
             for name, tension_series in tension_data.items():
                 if not tension_series.dropna().empty: segment_tension_bounds[name] = {"min": tension_series.min(), "max": tension_series.max()}

        cmap_tension = plt.cm.get_cmap('plasma')
        cmap_muscle_fixed = plt.cm.get_cmap('gist_rainbow', len(self.muscle_indicators_def)) # アニメと同じ色
        muscle_colors_fixed = {name: cmap_muscle_fixed(i) for i, (name, _) in enumerate(self.muscle_indicators_def.items())}

        target_frames = np.linspace(0, 100, 11) # 0, 10, ..., 100

        for frame_value in target_frames:
            # frame_valueに最も近い%を探す
            actual_frame_value = df_mean_cycle.iloc[(df_mean_cycle['gait_cycle_%'] - frame_value).abs().argsort()[:1]]['gait_cycle_%'].values[0]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            current_data = df_mean_cycle[df_mean_cycle['gait_cycle_%'] == actual_frame_value]

            # 軸範囲設定
            all_x, all_y, all_z = df_mean_cycle['x'].values, df_mean_cycle['y'].values, df_mean_cycle['z'].values
            max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() * 1.1
            mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
            ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2); ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2); ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)
            ax.set_box_aspect([1, 1, 1]); ax.set_title(f"Gait Cycle: {actual_frame_value:.0f}%", fontsize=16)
            ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')

            # 仰角30度、方位角30度 (デフォルト-60から+90度回転)
            ax.view_init(elev=30, azim=210)

            positions = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()}

            # マーカー描画 (マーカーIDの凡例は非表示にするため label は付けない)
            for marker_id, (x, y, z) in positions.items():
                if marker_id in marker_colors: ax.scatter(x, y, z, color=marker_colors[marker_id], s=40)

            # ゴムライン描画
            if tension_data:
                for name, (p1, p2) in lines_to_draw_def.items():
                    if p1 in positions and p2 in positions and name in tension_data:
                        pos1, pos2 = np.array(positions[p1]), np.array(positions[p2])
                        tension = tension_data[name].get(actual_frame_value, 0)
                        color = 'gray'
                        if name in segment_tension_bounds: bounds = segment_tension_bounds[name]; range_val = bounds["max"] - bounds["min"]; relative_tension = np.clip((tension - bounds["min"]) / (range_val + 1e-9), 0, 1); color = cmap_tension(relative_tension)
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color=color, linewidth=4)

            # --- ▼▼▼ 筋肉マーカー描画 ▼▼▼ ---
            if self.emg_data is not None:
                try: current_emg = self.emg_data.loc[actual_frame_value]
                except KeyError: current_emg = pd.Series(0, index=self.emg_data.columns)

                for name, muscle_info in self.muscle_indicators_def.items():
                    muscle_pos = self.calculate_indicator_position(muscle_info, positions)
                    if muscle_pos is not None and not np.isnan(muscle_pos).any():
                        emg_col = muscle_info['emg_col']
                        emg_activity = current_emg.get(emg_col, 0) # 正規化済み(0-1)
                        marker_size = config.MUSCLE_MARKER_BASE_SIZE + emg_activity * config.MUSCLE_MARKER_SCALE_FACTOR
                        color = muscle_colors_fixed.get(name, 'black') # 色を取得
                        ax.scatter(muscle_pos[0], muscle_pos[1], muscle_pos[2],
                                   s=marker_size*marker_size, # scatterのsは面積なので二乗する
                                   color=color, alpha=0.8,
                                   label=name if frame_value==0 else "") # 初回のみラベル表示
            # --- ▲▲▲ 筋肉マーカー描画 ▲▲▲ ---

            # カラーバー (張力) - 3Dマップと重ならないよう左端に配置
            if tension_data:
                sm_t = plt.cm.ScalarMappable(cmap=cmap_tension, norm=plt.Normalize(vmin=0, vmax=1))
                cbar_t = fig.colorbar(sm_t, ax=ax, shrink=0.5, aspect=10, pad=0.1, location='left')
                cbar_t.set_label('Relative Tension (0-1)')

            # 凡例 (筋活動、初回フレームのみ表示) - 3Dマップと重ならないよう右端に配置
            # マーカーIDの凡例は非表示 (マーカーにlabelを付けていないため取得されない)
            if frame_value == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, title="Muscles", loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8)

            fig.subplots_adjust(left=0.05, right=0.8)

            plt.show()
        print("Finished generating static maps.")
    # --- ▲▲▲ 関数修正ここまで ▲▲▲ ---

    

# --- ▼▼▼【main関数修正】▼▼▼ ---
def run_interactive_menu(visualizer, df_mean_cycle, tension_data, cfg, task_key):
    """ユーザーとの対話メニューを実行する関数"""
    if tension_data is None and visualizer.emg_data is None:
        print("張力データもEMGデータもありません。可視化メニューをスキップします。")
        return

    while True:
        print("\n--- 操作を選択してください ---")
        options = {}
        if tension_data is not None or visualizer.emg_data is not None:
             options['a'] = "アニメーションを表示"
             options['s'] = "アニメーションを保存 (GIF)"
        if tension_data is not None:
             options['g'] = "張力グラフを表示"
             options['m'] = "静止3Dマップを表示 (10%刻み)"
        options['q'] = "終了"

        for key, desc in options.items():
            print(f"  {key}: {desc}")

        action = input(f"実行する操作 [{'/'.join(options.keys())}]: ").lower()

        if action == 'a' and 'a' in options:
            print("\nアニメーションを表示します...")
            visualizer.run_animation(
                df_mean_cycle, tension_data if tension_data else {}, # Noneなら空辞書
                cfg['LINES_TO_DRAW'], show=True, save_path=None
            )
        elif action == 's' and 's' in options:
            video_filename = f"{task_key}_result_video.gif"
            save_path = os.path.join(config.RESULT_DIR, video_filename)
            try:
                os.makedirs(config.RESULT_DIR, exist_ok=True)
                print(f"\nアニメーションを保存します: {save_path}")
                visualizer.run_animation(
                    df_mean_cycle, tension_data if tension_data else {},
                    cfg['LINES_TO_DRAW'], show=False, save_path=save_path
                )
            except Exception as e: print(f"ファイル保存エラー: {e}")
        elif action == 'g' and 'g' in options:
            print("\nタスク間でグラフの軸を統一するため、全タスクのデータから軸範囲を算出しています...")
            axis_bounds = visualizer.calculate_global_tension_bounds()
            visualizer.plot_segment_tensions(tension_data, config.SEGMENT_GROUPS, task_key, axis_bounds)
        elif action == 'm' and 'm' in options:
            visualizer.show_static_3d_maps(df_mean_cycle, tension_data if tension_data else {}, cfg['LINES_TO_DRAW'])
        elif action == 'q':
            print("プログラムを終了します。")
            break
        else:
            print("無効な入力です。")

def main():
    """メイン実行関数"""
    while True:
        task_key = input("解析するタスク名を入力してください (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS[task_key]; break
        else: print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")

    print(f"\n--- {task_key} の解析を開始します ---")
    try: df_mean_cycle = pd.read_csv(cfg['MEAN_CYCLE_OUTPUT_PATH']); print(f"平均化データを読み込みました: {cfg['MEAN_CYCLE_OUTPUT_PATH']}")
    except Exception as e: print(f"平均化データの読み込みエラー: {e}"); return

    # EMGデータパスをグローバル設定(config.XXX)からタスク固有設定(cfg[XXX])に変更
    visualizer = TensionVisualizer(
        config.RUBBER_PROPERTIES_EXCEL_PATH,
        config.RUBBER_PROPERTIES_SHEET_NAME,
        cfg['EMG_DATA_CSV_PATH'], # ★ ここを修正
        cfg['MUSCLE_INDICATORS']
    )

    tension_data, tension_df_for_csv = None, None # 初期化
    if visualizer.strain_to_force_interp is not None:
        tension_data, tension_df_for_csv = visualizer.calculate_all_tensions(
            df_mean_cycle, cfg['NATURAL_LENGTHS'], cfg['LINES_TO_DRAW'])
        if tension_data is not None and tension_df_for_csv is not None:
            try:
                os.makedirs(os.path.dirname(cfg['TENSION_DATA_OUTPUT_PATH']), exist_ok=True)
                tension_df_for_csv.to_csv(cfg['TENSION_DATA_OUTPUT_PATH'], index=False, float_format='%.4f')
                print(f"張力データを保存しました: {cfg['TENSION_DATA_OUTPUT_PATH']}")
            except Exception as e: print(f"張力CSVの保存エラー: {e}")
    else: print("警告: ひずみ-荷重関係を作成できませんでした。張力計算はスキップされます。")

    # 対話メニューを実行 (張力データがなくてもEMGがあれば実行)
    if visualizer.emg_data is not None or tension_data is not None:
        run_interactive_menu(visualizer, df_mean_cycle, tension_data, cfg, task_key)
    else:
        print("エラー: 張力もEMGデータも処理できませんでした。可視化メニューをスキップします。")


    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()
