#optieditDで処理後

# =============================================================================
# create_anime_grad_D.py
#
# 概要:
# Step 1 (opti_edit_D.py) で作成したクリーンなマーカーデータ
# (task2_corrected_A.csv) と、歩行周期データ (task2_gait_cycles.csv) を
# 読み込み、平均化処理と可視化を行います。
#
# このスクリプトは、2つの実行モードを持ちます。
#
# モード 1: 平均歩行周期 (Mean Gait Cycle)
# 目的:
#   複数の歩行周期を抽出し、時間正規化(101ポイント)して平均化します。
#   これにより、ノイズが除去された「平均的な1歩」のデータを生成します。
# 処理:
#   1. LabChartの周期データ(hs_time)とオフセット(TIME_OFFSET)に基づき、
#      corrected_A.csv から各周期のデータを切り出します。
#   2. 基準マーカー(REFERENCE_MARKER_ID)の「絶対座標」を周期ごとに
#      正規化(_normalize_trajectory)し、平均します。
#   3. 他マーカーは、基準マーカーからの「相対座標」を周期ごとに計算・
#      正規化・平均化します。
#   4. 最後に、(3)の平均相対座標を(2)の平均絶対座標に足し戻し、
#      平均的な絶対座標の軌道(df_anim)を作成します。
#   5. v12.1の修正: _normalize_trajectory が静止マーカーでも
#      (101, 3)の形状を保つように修正され、np.stack エラーを回避します。
# 出力:
#   - 3Dアニメーション (Mean Gait Cycle Trajectory)
#   - 部位ごとのゴム長グラフ (Segment Length Over Gait Cycle)
#   - 平均化データ (MEAN_CYCLE_OUTPUT_PATH, task2_mean_cycle.csv)
#
# モード 2: 指定時間区間 (Specific Time Range)
# 目的:
#   corrected_A.csv のデータをそのままアニメーションで再生します。
#   opti_edit_D.py の追跡結果（膝の屈曲など）が正しく反映されているか
#   どうかを、生の動きで確認するために使用します。
# 出力:
#   - 3Dアニメーション (Raw Trajectory (start_t - end_t))
#
# アニメーションの色分け:
#   線分(ゴム)の色は、データ全体での最小長(min)と最大長(max)を基準に、
#   現在の長さ(cur)がどの程度伸びているか( (cur-min)/(max-min) )を
#   0(青)から1(赤系)のグラデーションで表現します。
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import seaborn as sns
import os
import config # ★ 設定ファイルをインポート

class DataProcessor:
    """データの読み込み、前処理、歩行周期の平均化を担当するクラス"""

    def __init__(self, ref_marker_id):
        self.ref_marker_id = ref_marker_id

    def load_and_prepare_data(self, opti_path, axis_mapping):
        """修正済みOptiTrackデータを読み込み、座標軸を変換する"""
        print(f"修正済みOptiTrackデータを読み込み中: {opti_path}")
        try:
            df_corrected = pd.read_csv(opti_path)
        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません: {opti_path}")
            return None
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return None

        df_plot = pd.DataFrame()
        required_cols = ['Frame', 'Time', 'id'] + list(axis_mapping.values()) # 必要な元列名
        missing_cols = [col for col in required_cols if col not in df_corrected.columns and not col.startswith('-')]
        if missing_cols:
            print(f"エラー: 入力CSVに必要な列が見つかりません: {missing_cols}")
            return None

        for col in ['Frame', 'Time', 'id']:
            df_plot[col] = df_corrected[col]
        for final_axis, source_axis in axis_mapping.items():
            col_name = final_axis.replace('final_', '')
            if source_axis.startswith('-'):
                df_plot[col_name] = -df_corrected[source_axis[1:]]
            else:
                df_plot[col_name] = df_corrected[source_axis]
        print("データ読み込みと軸マッピング完了。")
        return df_plot

    def _normalize_trajectory(self, df, num_points=101):
        """単一マーカーの軌跡を指定した点数に正規化(内挿)する"""
        time_orig = df['Time'].to_numpy()
        # ★ 周期の開始・終了時刻が同じ場合（非常にまれ）のエラー回避
        if time_orig.min() == time_orig.max():
             time_norm_abs = np.full(num_points, time_orig.min())
        else:
             time_norm_abs = np.linspace(time_orig.min(), time_orig.max(), num_points)

        df_normalized = pd.DataFrame(index=range(num_points))

        for axis in ['x', 'y', 'z']:
            axis_data = df[axis]
            if len(axis_data.unique()) == 1:
                df_normalized[axis] = axis_data.iloc[0]
            elif len(time_orig) < 2: # データ点が1つしかない場合
                df_normalized[axis] = axis_data.iloc[0]
            else:
                # ★ interp1d のエラーハンドリングを追加
                try:
                    f = interp1d(time_orig, axis_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    df_normalized[axis] = f(time_norm_abs)
                except ValueError as e:
                    print(f"  警告: 内挿エラー ({axis}軸): {e}. 線形補間で代替します。")
                    try: # 線形補間を試す
                         f_linear = interp1d(time_orig, axis_data, kind='linear', bounds_error=False, fill_value='extrapolate')
                         df_normalized[axis] = f_linear(time_norm_abs)
                    except Exception as ie:
                         print(f"  エラー: 線形補間も失敗 ({axis}軸): {ie}. NaNで埋めます。")
                         df_normalized[axis] = np.nan

        return df_normalized

    def _butter_lowpass_filter(self, data, cutoff=50, fs=config.FRAME_RATE, order=4): # ★ configからFRAME_RATE取得
        """バターワースローパスフィルタを適用する"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # ★ filtfilt のエラーハンドリング (データ長が短い場合)
        padlen = 3 * max(len(a), len(b)) # filtfiltの推奨値
        if len(data) <= padlen:
            print(f"  警告: データ長({len(data)})がフィルタの次数に対して短すぎるため、フィルタリングをスキップします。")
            return data
        try:
            return filtfilt(b, a, data)
        except Exception as e:
            print(f"  警告: フィルタ適用中にエラーが発生しました: {e}. フィルタリングをスキップします。")
            return data


    def calculate_mean_gait_cycle(self, df_plot, cycles_path, time_offset,
                                  filter_start_time=None, filter_end_time=None):
        """
        複数歩行周期から平均軌道を計算する。
        オプションで指定時間内の周期のみにフィルタリングする。
        """
        try:
            df_cycles_all = pd.read_csv(cycles_path)
        except FileNotFoundError:
            print(f"エラー: 歩行周期ファイルが見つかりません: {cycles_path}"); return None
        except Exception as e:
            print(f"歩行周期ファイルの読み込みエラー: {e}"); return None

        # モード3（指定区間平均化）のためのフィルタリング
        if filter_start_time is not None and filter_end_time is not None:
            print(f"歩行周期を指定範囲内 ({filter_start_time:.2f}s - {filter_end_time:.2f}s, OptiTrack時間) でフィルタリング...")
            df_cycles_all['opti_start_time'] = df_cycles_all['hs_time'] - time_offset
            df_cycles_all['opti_end_time'] = df_cycles_all['next_hs_time'] - time_offset
            valid_cycles_mask = (df_cycles_all['opti_start_time'] >= filter_start_time) & \
                                (df_cycles_all['opti_end_time'] <= filter_end_time)
            df_cycles = df_cycles_all[valid_cycles_mask].copy()
            print(f"  指定範囲内に完全に含まれる {len(df_cycles)} 歩行周期が見つかりました (全 {len(df_cycles_all)} 周期中)。")
            if df_cycles.empty: print("エラー: 指定範囲内に有効な歩行周期がありません。"); return None
        else:
            df_cycles = df_cycles_all # モード1

        all_cycles_rel = {int(mid): [] for mid in df_plot['id'].unique()} # ★ IDをintに
        all_cycles_ref_abs = []
        valid_cycle_count = 0

        print("歩行周期データの抽出と正規化を開始...")
        for _, cycle in df_cycles.iterrows():
            start_time = float(cycle.get('opti_start_time', cycle['hs_time'] - time_offset))
            end_time = float(cycle.get('opti_end_time', cycle['next_hs_time'] - time_offset))

            # 抽出範囲を少し広げて、境界でのデータ欠落を防ぐ (例: 前後1フレーム)
            buffer = 1.0 / config.FRAME_RATE # ★ configからFRAME_RATE取得
            cycle_df = df_plot[(df_plot['Time'] >= start_time - buffer) & (df_plot['Time'] <= end_time + buffer)].copy()

            ref_traj_cycle = cycle_df[cycle_df['id'] == self.ref_marker_id]
            if ref_traj_cycle.empty or len(ref_traj_cycle) < 4:
                # print(f"  警告: Cycle ({start_time:.2f}-{end_time:.2f}s) 基準マーカー不足のためスキップ。")
                continue # 基準マーカーがない周期はスキップ

            # 基準マーカーの絶対座標を正規化
            normalized_ref = self._normalize_trajectory(ref_traj_cycle)
            if normalized_ref.isnull().values.any(): # 内挿失敗チェック
                 # print(f"  警告: Cycle ({start_time:.2f}-{end_time:.2f}s) 基準マーカー正規化失敗のためスキップ。")
                 continue
            all_cycles_ref_abs.append(normalized_ref)

            # 他マーカーの相対座標を計算・正規化
            valid_marker_in_cycle = 0
            for marker_id_float, group in cycle_df.groupby('id'):
                marker_id = int(marker_id_float) # ★ IDをintに
                if marker_id == self.ref_marker_id or len(group) < 4: continue

                # merge_asof で時間的に最も近い基準マーカー位置を結合
                # suffixesに空文字('')以外を指定して列名衝突を回避
                merged = pd.merge_asof(
                    group.sort_values('Time'),
                    ref_traj_cycle[['Time', 'x', 'y', 'z']].sort_values('Time'),
                    on='Time', direction='nearest', suffixes=('_marker', '_ref')
                )
                if merged.empty or merged[['x_ref', 'y_ref', 'z_ref']].isnull().values.any(): continue

                # 相対座標を計算
                rel_group = pd.DataFrame({'Time': merged['Time']})
                for axis in ['x', 'y', 'z']:
                    rel_group[axis] = merged[f'{axis}_marker'] - merged[f'{axis}_ref']

                if not rel_group.empty:
                    normalized_rel = self._normalize_trajectory(rel_group)
                    if not normalized_rel.isnull().values.any(): # 内挿失敗チェック
                        if marker_id in all_cycles_rel: # 念のため存在確認
                             all_cycles_rel[marker_id].append(normalized_rel)
                             valid_marker_in_cycle += 1

            if valid_marker_in_cycle > 0: # この周期で少なくとも1つのマーカーが処理できたらカウント
                 valid_cycle_count += 1


        print(f"\n--- 処理された歩行周期の診断 ({valid_cycle_count} 周期) ---")
        print(f"  基準マーカー (ID {self.ref_marker_id}): {len(all_cycles_ref_abs)} 周期分のデータを取得")
        for mid, traj_list in all_cycles_rel.items():
            if mid != self.ref_marker_id:
                print(f"  マーカー ID {mid}: {len(traj_list)} 周期分のデータを取得")
        print("------------------------------------\n")

        if len(all_cycles_ref_abs) < 2:
            print(f"エラー: 平均化に必要な有効歩行周期が不足しています (基準マーカーで {len(all_cycles_ref_abs)} 周期)。最低2周期必要です。")
            return None

        # 平均化処理
        print("歩行周期データを平均化中...")
        try:
            stacked_ref_trajs = np.stack([df[['x','y','z']].values for df in all_cycles_ref_abs])
            mean_ref_abs_traj = np.mean(stacked_ref_trajs, axis=0)
        except ValueError as e:
            print(f"エラー: 基準マーカーの平均化中に形状不一致が発生しました: {e}")
            # 各DataFrameの形状を出力してデバッグ
            # for i, df in enumerate(all_cycles_ref_abs): print(f"  Ref cycle {i} shape: {df[['x','y','z']].values.shape}")
            return None

        mean_rel_trajectories = {}
        for mid, traj_list in all_cycles_rel.items():
            if mid != self.ref_marker_id and len(traj_list) >= 2: # 平均化には最低2周期必要
                try:
                    stacked_trajs = np.stack([df[['x','y','z']].values for df in traj_list])
                    mean_rel_trajectories[mid] = np.mean(stacked_trajs, axis=0)
                except ValueError as e:
                     print(f"警告: マーカー ID {mid} の平均化中に形状不一致: {e}。スキップします。")
                     # for i, df in enumerate(traj_list): print(f"  Marker {mid} cycle {i} shape: {df[['x','y','z']].values.shape}")
            elif mid != self.ref_marker_id and len(traj_list) < 2:
                 print(f"  警告: マーカー ID {mid} は有効周期が {len(traj_list)} 個しかないため、平均化から除外します。")


        if not mean_rel_trajectories and len(all_cycles_ref_abs) >= 2 :
             print("警告: 基準マーカー以外のマーカーで平均化可能なデータがありませんでした。基準マーカーのみのデータを出力します。")
        elif not mean_rel_trajectories and len(all_cycles_ref_abs) < 2:
             print("エラー: 平均化可能なデータがありません。")
             return None


        # 平均化された絶対座標を再構築
        mean_df_list = []
        gait_cycle_perc = np.linspace(0, 100, 101) # 101点 (0% から 100%)

        df_ref = pd.DataFrame(mean_ref_abs_traj, columns=['x','y','z'])
        df_ref['gait_cycle_%'] = gait_cycle_perc
        df_ref['id'] = self.ref_marker_id
        mean_df_list.append(df_ref)

        for mid, mean_rel_traj in mean_rel_trajectories.items():
            abs_traj = mean_rel_traj + mean_ref_abs_traj
            df_temp = pd.DataFrame(abs_traj, columns=['x','y','z'])
            df_temp['gait_cycle_%'] = gait_cycle_perc
            df_temp['id'] = mid
            mean_df_list.append(df_temp)

        if not mean_df_list:
             print("エラー: 平均化データリストが空です。")
             return None

        df_mean = pd.concat(mean_df_list, ignore_index=True)

        # フィルタリング (オプション)
        # print("Applying low-pass filter...")
        final_df_list_filtered = []
        for mid, group in df_mean.groupby('id'):
            group_filtered = group.copy()
            # for axis in ['x','y','z']:
            #     group_filtered[axis] = self._butter_lowpass_filter(group[axis].values)
            final_df_list_filtered.append(group_filtered)

        df_final = pd.concat(final_df_list_filtered, ignore_index=True)

        # --- ▼▼▼ デバッグ出力追加 ▼▼▼ ---
        print("\n--- 平均化データ整合性チェック ---")
        if not df_final.empty:
            gait_percentages = sorted(df_final['gait_cycle_%'].unique())
            print(f"  歩行周期 % 範囲: {gait_percentages[0]:.1f}% から {gait_percentages[-1]:.1f}%")
            print(f"  % ポイント数: {len(gait_percentages)}")

            ref_marker_data = df_final[df_final['id'] == self.ref_marker_id]
            pos_0_percent = ref_marker_data[np.isclose(ref_marker_data['gait_cycle_%'], 0.0)][['x','y','z']].values
            pos_50_percent = ref_marker_data[np.isclose(ref_marker_data['gait_cycle_%'], 50.0)][['x','y','z']].values

            if pos_0_percent.size > 0 and pos_50_percent.size > 0:
                print(f"  基準マーカー ({self.ref_marker_id}) 0% 時点位置:  x={pos_0_percent[0,0]:.1f}, y={pos_0_percent[0,1]:.1f}, z={pos_0_percent[0,2]:.1f}")
                print(f"  基準マーカー ({self.ref_marker_id}) 50% 時点位置: x={pos_50_percent[0,0]:.1f}, y={pos_50_percent[0,1]:.1f}, z={pos_50_percent[0,2]:.1f}")
                # z が表示上の X軸 (AXIS_MAPPING)
                if pos_0_percent[0, 0] > pos_50_percent[0, 0]: # 表示X軸(元Z軸)で比較
                     print("  ---> 警告: 0% 時点の X(元Z)座標が 50% 時点より大きいようです。平均化データで位相がズレている可能性があります。")
            else:
                print("  0% または 50% 時点の基準マーカー位置を取得できませんでした。")
        else:
            print("  平均化データが空です。")
        print("---------------------------------\n")
        # --- ▲▲▲ デバッグ出力追加 ▲▲▲ ---

        return df_final


class Visualizer:
    """3Dアニメーションとグラフの描画を担当するクラス"""

    def animate_3d_plot(self, df_anim, title, time_col_name, is_cycle, lines_to_draw_def): # ★ lines_to_draw_def を追加
        """3Dアニメーション（ゴムの伸び率を色で表現）を作成する"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        marker_ids = sorted(df_anim['id'].unique())
        num_markers = len(marker_ids)
        if num_markers <= 10: cmap = plt.cm.get_cmap('tab10', num_markers)
        elif num_markers <= 20: cmap = plt.cm.get_cmap('tab20', num_markers)
        else: cmap = plt.cm.get_cmap('gist_rainbow', num_markers)
        marker_colors = {mid: cmap(i) for i, mid in enumerate(marker_ids)}
        plots = {mid: ax.plot([], [], [], marker='o', color=marker_colors[mid], label=f'ID {int(mid)}')[0] for mid in marker_ids} # ★ IDをintに

        # ★ lines_to_draw_def を使用
        lines = {name: ax.plot([], [], [], linewidth=3)[0] for name in lines_to_draw_def.keys()}
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

        palette = sns.color_palette("winter", n_colors=256)
        min_lengths, max_lengths = self._calculate_length_bounds(df_anim, lines_to_draw_def) # ★ lines_to_draw_def を渡す

        def init():
            all_x, all_y, all_z = df_anim['x'].values, df_anim['y'].values, df_anim['z'].values
            max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() * 1.1 # 範囲を少し広げる
            mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            ax.set_box_aspect([1,1,1])
            ax.set_title(title); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
            return list(plots.values()) + list(lines.values())

        def update(frame_value):
            # isclose を使って浮動小数点数比較を安全に行う
            current_data = df_anim[np.isclose(df_anim[time_col_name], frame_value)]
            if is_cycle:
                time_str = f'Gait Cycle: {frame_value:.1f} %'
            else:
                time_val = current_data["Time"].iloc[0] if not current_data.empty else 0
                time_str = f'Time: {time_val:.3f}s ({frame_value})' # フレーム番号も表示

            positions = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()} # ★ IDをintに

            for marker_id, plot in plots.items():
                if marker_id in positions:
                    x, y, z = positions[marker_id]
                    plot.set_data([x], [y]); plot.set_3d_properties([z])

            # ★ lines_to_draw_def を使用
            for name, (p1, p2) in lines_to_draw_def.items():
                line = lines.get(name)
                if line and p1 in positions and p2 in positions:
                    pos1, pos2 = np.array(positions[p1]), np.array(positions[p2])
                    cur_len = np.linalg.norm(pos1 - pos2)
                    color = 'gray' # デフォルト
                    min_len = min_lengths.get(name, 0)
                    max_len = max_lengths.get(name, 1)
                    range_len = max_len - min_len
                    if range_len > 1e-8: # ゼロ除算回避
                        stretch_ratio = np.clip((cur_len - min_len) / range_len, 0, 1)
                        color_idx = int(stretch_ratio * (len(palette) - 1))
                        color = palette[color_idx]

                    line.set_color(color)
                    line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                    line.set_3d_properties([pos1[2], pos2[2]])

            time_text.set_text(time_str)
            return list(plots.values()) + list(lines.values())

        # ★ is_cycle=False の場合、time_col_name='Frame' なので frames 引数を修正
        frames_to_iterate = sorted(df_anim[time_col_name].unique())
        ani = animation.FuncAnimation(fig, update, frames=frames_to_iterate,
                                      init_func=init, blit=False, interval=1000/config.FRAME_RATE) # ★ configからFRAME_RATE取得
        plt.show()

    def _calculate_length_bounds(self, df, lines_to_draw_def): # ★ lines_to_draw_def を追加
        """データ全体から各線分の最小長と最大長を計算する"""
        min_lengths, max_lengths = {}, {}
        unique_ids = df['id'].unique() # データに存在するIDリスト
        # ★ lines_to_draw_def を使用
        for name, (p1, p2) in lines_to_draw_def.items():
            # IDがデータに存在するか確認
            if p1 not in unique_ids or p2 not in unique_ids:
                 print(f"警告: 長さ計算スキップ ({name})。ID {p1} または {p2} がデータにありません。")
                 min_lengths[name] = 0; max_lengths[name] = 1
                 continue

            coords1 = df[df['id'] == p1].sort_values(df.columns[0])[['x','y','z']].values # 最初の列(Frame or gait_cycle_%)でソート
            coords2 = df[df['id'] == p2].sort_values(df.columns[0])[['x','y','z']].values

            if coords1.shape[0] == coords2.shape[0] and coords1.shape[0] > 0:
                lengths = np.linalg.norm(coords1 - coords2, axis=1)
                min_lengths[name] = lengths.min()
                max_lengths[name] = lengths.max()
            else:
                print(f"警告: 長さ計算スキップ ({name})。座標形状不一致 ({coords1.shape[0]} vs {coords2.shape[0]})。")
                min_lengths[name] = 0; max_lengths[name] = 1
        return min_lengths, max_lengths

    def plot_segment_lengths(self, df_mean_cycle, seg_groups, lines_to_draw_def): # ★ 引数追加
        """部位ごとに線分長のグラフを描画する"""
        print("セグメント長のグラフを生成中...")
        if df_mean_cycle.empty or 'gait_cycle_%' not in df_mean_cycle.columns:
            print("平均化データが不正なため、グラフをスキップします。")
            return

        gait_perc = sorted(df_mean_cycle['gait_cycle_%'].unique())
        unique_ids = df_mean_cycle['id'].unique() # データに存在するID

        for seg_name, lines_list in seg_groups.items():
            plt.figure(figsize=(10, 5))
            has_data_in_seg = False
            for line_name in lines_list:
                # ★ lines_to_draw_def を使用
                if line_name not in lines_to_draw_def: continue

                p1, p2 = lines_to_draw_def[line_name]
                if p1 not in unique_ids or p2 not in unique_ids:
                    # print(f"  グラフスキップ ({line_name}): ID {p1} or {p2} がデータにありません。")
                    continue

                df1 = df_mean_cycle[df_mean_cycle['id'] == p1].sort_values('gait_cycle_%')
                df2 = df_mean_cycle[df_mean_cycle['id'] == p2].sort_values('gait_cycle_%')

                if not df1.empty and not df2.empty and df1.shape[0] == df2.shape[0]:
                    lengths = np.linalg.norm(df1[['x','y','z']].values - df2[['x','y','z']].values, axis=1)
                    # np.isclose を使って gait_perc と df1/df2 のインデックスを一致させる
                    if len(gait_perc) == len(lengths):
                         sns.lineplot(x=gait_perc, y=lengths, label=line_name)
                         has_data_in_seg = True
                    # else:
                         # print(f"  グラフスキップ ({line_name}): gait_perc({len(gait_perc)})とlengths({len(lengths)})の長さ不一致。")


            if has_data_in_seg:
                plt.title(f"{seg_name} Segment Length Over Gait Cycle")
                plt.xlabel("Gait Cycle (%)"); plt.ylabel("Segment Length (mm)")
                plt.legend(); plt.grid(True)
                plt.show()
            else:
                plt.close() # データがない場合は閉じる
                print(f"  グラフスキップ ({seg_name}): 有効なデータがありません。")
        print("セグメント長のグラフ生成完了。")


def main():
    """メイン実行関数"""
    # 1. タスク選択
    while True:
        task_key = input("解析するタスク名を入力してください (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS[task_key]
            print(f"\n--- {task_key} の設定を読み込みました ---")
            break
        else:
            print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")

    # 2. processor 初期化 (設定から基準マーカー取得)
    processor = DataProcessor(ref_marker_id=cfg['REFERENCE_MARKER_ID'])
    visualizer = Visualizer()

    # 3. データの読み込みと準備
    df_plot = processor.load_and_prepare_data(cfg['OUTPUT_CSV_PATH'], config.AXIS_MAPPING) # 共通設定を使用
    if df_plot is None: return # エラーなら終了

    # 4. 実行モードの選択ループ
    while True:
        print("\n--- 操作を選択してください ---")
        print("  1: 平均歩行周期 (全周期)")
        print("  2: 指定時間区間アニメーション (生データ)")
        print("  3: 指定区間平均化 (範囲指定)")
        print("  q: 終了")
        mode = input("実行するモード [1/2/3/q]: ").lower()

        df_anim = None # アニメーション/グラフ用データフレーム
        title = ""
        time_col_name = ""
        is_cycle = False
        output_path = "" # 保存先パス

        if mode == '1':
            df_anim = processor.calculate_mean_gait_cycle(df_plot, cfg['LABCHART_CYCLES_PATH'], config.TIME_OFFSET) # 共通設定を使用
            if df_anim is not None and not df_anim.empty:
                title = f'{task_key} Mean Gait Cycle Trajectory (All Cycles)'
                time_col_name = 'gait_cycle_%'
                is_cycle = True
                output_path = cfg['MEAN_CYCLE_OUTPUT_PATH'] # モード1の保存先

        elif mode == '2':
            try:
                start_t = float(input("アニメーション開始時刻 (秒) を入力: "))
                end_t = float(input("アニメーション終了時刻 (秒) を入力: "))
                if start_t >= end_t: print("エラー: 終了時刻は開始時刻より後にしてください。"); continue
            except ValueError: print("エラー: 数値を入力してください。"); continue

            df_anim = df_plot[(df_plot['Time'] >= start_t) & (df_plot['Time'] <= end_t)]
            if df_anim.empty: print("指定された時間範囲にデータが見つかりませんでした。"); continue
            else:
                title = f'{task_key} Raw Trajectory ({start_t:.2f}s - {end_t:.2f}s)'
                time_col_name = 'Frame' # フレーム番号でアニメーション
                is_cycle = False
                # モード2はCSV保存しない

        elif mode == '3':
            try:
                start_t = float(input("平均化する範囲の開始時刻 (秒) を入力 (例: 20.0): "))
                end_t = float(input("平均化する範囲の終了時刻 (秒) を入力 (例: 40.0): "))
                if start_t >= end_t: print("エラー: 終了時刻は開始時刻より後にしてください。"); continue
            except ValueError: print("エラー: 数値を入力してください。"); continue

            df_anim = processor.calculate_mean_gait_cycle(
                df_plot, cfg['LABCHART_CYCLES_PATH'], config.TIME_OFFSET, # 共通設定を使用
                filter_start_time=start_t, filter_end_time=end_t
            )
            if df_anim is not None and not df_anim.empty:
                title = f'{task_key} Mean Gait Cycle Trajectory (Ranged: {start_t:.1f}s - {end_t:.1f}s)'
                time_col_name = 'gait_cycle_%'
                is_cycle = True
                output_path = cfg['MEAN_CYCLE_RANGED_OUTPUT_PATH'] # モード3の保存先

        elif mode == 'q':
            print("プログラムを終了します。")
            break # while ループを抜ける
        else:
            print("無効な入力です。1, 2, 3, q のいずれかを入力してください。")
            continue # モード選択に戻る

        # --- アニメーション/グラフ表示 (df_anim が生成された場合) ---
        if df_anim is not None and not df_anim.empty:
            # CSV保存 (モード1または3の場合)
            if output_path:
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) # フォルダ作成
                    df_anim.to_csv(output_path, index=False)
                    print(f"平均化データを保存しました: {output_path}")
                except Exception as e: print(f"平均化データのCSV保存エラー: {e}")

            # アニメーション表示
            visualizer.animate_3d_plot(df_anim, title, time_col_name, is_cycle, cfg['LINES_TO_DRAW']) # ★ cfgから渡す

            # グラフ表示 (平均化モードの場合のみ)
            if is_cycle:
                visualizer.plot_segment_lengths(df_anim, config.SEGMENT_GROUPS, cfg['LINES_TO_DRAW']) # ★ configとcfgから渡す
        elif mode != 'q': # エラーメッセージ (終了以外の場合)
            print("エラー: 選択されたモードの処理に失敗したか、データが生成されませんでした。")

    print(f"\n--- {task_key} の解析終了 ---")


if __name__ == '__main__':
    main()