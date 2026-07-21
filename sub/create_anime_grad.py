'''
#create_anime_grad.py

入力：
opti_edit_A.pyで出力したoptiのクリーンなマーカーデータ
CORRECTED_OPTI_CSV_PATH = r"C:\FuttoAnalysis\opti\20250731\task1_corrected_A.csv"

visualize_labchart.pyで出力した歩行周期データ
LABCHART_CYCLES_PATH = r"C:\FuttoAnalysis\labchart\20250731\task1_gait_cycles.csv"

**出力**:
      * `task1_mean_cycle.csv` (平均化されたマーカーデータ)
      * 平均歩行周期のアニメーション表示
      * 部位ごとの長さ変化のグラフ表示
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import seaborn as sns
import os

# --- ▼▼▼ 設定 ▼▼▼ ---
# 入力ファイルパス
CORRECTED_OPTI_CSV_PATH = r"C:\FuttoAnalysis\opti\20251111\try_corrected.csv"
#r"C:\FuttoAnalysis\opti\20251020\task2_corrected_A.csv"
LABCHART_CYCLES_PATH = r"C:\FuttoAnalysis\labchart\20251020\task2_gait_cycles.csv"


# 出力ファイルパス (平均化データ用)
MEAN_CYCLE_OUTPUT_PATH = r"C:\FuttoAnalysis\opti\20251020\task2_mean_cycle.csv"

# 定数
TIME_OFFSET = 0.0  # LabChartデータとOptiTrackデータの時間オフセット(秒)
FRAME_RATE = 120    # OptiTrackのフレームレート(Hz)
REFERENCE_MARKER_ID = 22638 # 体全体の移動を代表するマーカーID (例: 腰)

# 座標軸のマッピング (OptiTrack座標系 -> 表示したい座標系)
AXIS_MAPPING = {"final_x": 'x', "final_y": 'y', "final_z": 'z'}

# 描画する線分の定義 (マーカーIDのペア)
LINES_TO_DRAW = {
    "Front_Upper_In": (24774, 24768), "Front_Upper_Out": (24770, 24768),
    "Front_Knee_Upper_Out": (24768, 24754), "Front_Knee_Upper_In": (24768, 24782),
    "Front_Knee_Lower_Out": (24754, 24772), "Front_Knee_Lower_In": (24782, 24772),
    "Front_Shin": (24772, 24758), "Toe_Out": (24758, 24756), "Toe_In": (24758, 24764),
    "Back_Upper_In": (24778, 24776), "Back_Upper_Out": (24762, 24776),
    "Back_Thigh_Out": (24776, 24754), "Back_Thigh_In": (24776, 24782),
    "Back_Knee_Out": (24754, 24752), "Back_Knee_In": (24782, 24752),
    "Back_Shin_In": (24752, 24766), "Back_Shin_Out": (24752, 24760),
}
'''
#task1
LINES_TO_DRAW = {
    "Front_Upper_In": (15810, 15808), "Front_Upper_Out": (15796, 15808),
    "Front_Knee_Upper_Out": (15808, 15794), "Front_Knee_Upper_In": (15808, 15800),
    "Front_Knee_Lower_Out": (15794, 15798), "Front_Knee_Lower_In": (15800, 15798),
    "Front_Shin": (15798, 15802), "Toe_Out": (15802, 15812), "Toe_In": (15802, 15806),
    "Back_Upper_In": (15814, 15816), "Back_Upper_Out": (15804, 15816),
    "Back_Thigh_Out": (15816, 15794), "Back_Thigh_In": (15816, 15800),
    "Back_Knee_Out": (15794, 15818), "Back_Knee_In": (15800, 15818),
    "Back_Shin_In": (15818, 15792), "Back_Shin_Out": (15818, 15820),
}
'''


# 部位ごとの線分グループ定義 (グラフ描画用)
SEGMENT_GROUPS = {
    "Front_Knee": ["Front_Knee_Upper_Out", "Front_Knee_Upper_In", "Front_Knee_Lower_Out", "Front_Knee_Lower_In"],
    "Front_Ankle": ["Front_Shin", "Toe_Out", "Toe_In"], "Back_Knee": ["Back_Knee_Out", "Back_Knee_In"],
    "Back_Thigh": ["Back_Thigh_Out", "Back_Thigh_In"], "Back_Shin": ["Back_Shin_In", "Back_Shin_Out"],
    "Front_Hip": ["Front_Upper_In", "Front_Upper_Out"], "Back_Hip": ["Back_Upper_In", "Back_Upper_Out"]
}
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


class DataProcessor:
    """データの読み込み、前処理、歩行周期の平均化を担当するクラス"""

    def __init__(self, ref_marker_id):
        self.ref_marker_id = ref_marker_id

    def load_and_prepare_data(self, opti_path, axis_mapping):
        """修正済みOptiTrackデータを読み込み、座標軸を変換する"""
        print(f"Loading corrected OptiTrack data from: {opti_path}")
        df_corrected = pd.read_csv(opti_path)
        df_plot = pd.DataFrame()
        for col in ['Frame', 'Time', 'id']:
            df_plot[col] = df_corrected[col]
        for final_axis, source_axis in axis_mapping.items():
            col_name = final_axis.replace('final_', '')
            if source_axis.startswith('-'):
                df_plot[col_name] = -df_corrected[source_axis[1:]]
            else:
                df_plot[col_name] = df_corrected[source_axis]
        print("Data loading and axis mapping complete.")
        return df_plot

    def _normalize_trajectory(self, df, num_points=101):
        """単一マーカーの軌跡を指定した点数に正規化(内挿)する"""
        time_orig = df['Time'].to_numpy()
        time_norm_abs = np.linspace(time_orig.min(), time_orig.max(), num_points)
        
        df_normalized = pd.DataFrame()
        for axis in ['x', 'y', 'z']:
            if len(df[axis].unique()) == 1:
                df_normalized[axis] = df[axis].iloc[0]
            else:
                f = interp1d(time_orig, df[axis], kind='cubic', bounds_error=False, fill_value='extrapolate')
                df_normalized[axis] = f(time_norm_abs)
        return df_normalized

    def _butter_lowpass_filter(self, data, cutoff=50, fs=FRAME_RATE, order=4):
        """バターワースローパスフィルタを適用する"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)


    
    def calculate_mean_gait_cycle(self, df_plot, cycles_path, time_offset):
        """複数歩行周期から平均軌道を計算する"""
        try:
            df_cycles = pd.read_csv(cycles_path)
        except FileNotFoundError:
            print(f"Error: Gait cycle file not found at {cycles_path}")
            return None

        all_cycles_rel = {mid: [] for mid in df_plot['id'].unique()}
        all_cycles_ref_abs = []

        print("Extracting and normalizing gait cycles...")
        for _, cycle in df_cycles.iterrows():
            start_time = float(cycle['hs_time']) - time_offset
            end_time = float(cycle['next_hs_time']) - time_offset
            cycle_df = df_plot[(df_plot['Time'].between(start_time, end_time))].copy()

            ref_traj_cycle = cycle_df[cycle_df['id'] == self.ref_marker_id]
            if ref_traj_cycle.empty or len(ref_traj_cycle) < 4:
                continue

            all_cycles_ref_abs.append(self._normalize_trajectory(ref_traj_cycle))

            for marker_id, group in cycle_df.groupby('id'):
                if marker_id == self.ref_marker_id or len(group) < 4:
                    continue
                
                # --- ▼▼▼ KeyError修正箇所 ▼▼▼ ---
                merged = pd.merge_asof(
                    group.sort_values('Time'),
                    ref_traj_cycle[['Time', 'x', 'y', 'z']].sort_values('Time'),
                    on='Time',
                    direction='nearest',
                    suffixes=('', '_ref')
                )
                
                if merged.empty:
                    continue

                # 新しいDataFrameを作成し、相対座標を計算
                # これによりインデックスの不整合が解消される
                rel_group = pd.DataFrame()
                rel_group['Time'] = merged['Time']
                for axis in ['x', 'y', 'z']:
                    rel_group[axis] = merged[axis] - merged[f'{axis}_ref']
                
                if not rel_group.empty:
                    all_cycles_rel[marker_id].append(self._normalize_trajectory(rel_group))
                # --- ▲▲▲ 修正ここまで ▲▲▲ ---

        print("\n--- Cycle Count Diagnosis ---")
        print("Number of valid cycles found for each marker ID:")
        print(f"  - ID {self.ref_marker_id} (Reference): {len(all_cycles_ref_abs)} cycles")
        for mid, traj_list in all_cycles_rel.items():
            if mid != self.ref_marker_id:
                print(f"  - ID {mid}: {len(traj_list)} cycles")
        print("---------------------------\n")

        if len(all_cycles_ref_abs) < 2:
            print(f"Error: Not enough valid cycles for the reference marker (ID {self.ref_marker_id}). Found {len(all_cycles_ref_abs)}, need at least 2.")
            return None

        stacked_ref_trajs = np.stack([df[['x','y','z']].values for df in all_cycles_ref_abs])
        mean_ref_abs_traj = np.mean(stacked_ref_trajs, axis=0)

        mean_rel_trajectories = {}
        for mid, traj_list in all_cycles_rel.items():
            if mid != self.ref_marker_id and len(traj_list) > 1:
                stacked_trajs = np.stack([df[['x','y','z']].values for df in traj_list])
                mean_rel_trajectories[mid] = np.mean(stacked_trajs, axis=0)
        
        if not mean_rel_trajectories:
            print("Warning: No other markers had enough data (at least 2 cycles) to be averaged.")
            # この場合でも基準マーカーのみのアニメーションは作成せずに処理を終える場合は以下のreturnを有効化
            # return None

        print("Averaging cycles...")
        mean_df_list = []
        gait_cycle_perc = np.linspace(0, 100, 101)

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
            
        df_anim = pd.concat(mean_df_list, ignore_index=True)

        print("Applying low-pass filter...")
        final_df_list = []
        for mid, group in df_anim.groupby('id'):
            group_filtered = group.copy()
            for axis in ['x','y','z']:
                group_filtered[axis] = group[axis].values #self._butter_lowpass_filter(group[axis].values)
            final_df_list.append(group_filtered)

        return pd.concat(final_df_list, ignore_index=True)




class Visualizer:
    """3Dアニメーションとグラフの描画を担当するクラス"""

    def animate_3d_plot(self, df_anim, title, time_col_name, is_cycle, frame_rate):
        """3Dアニメーション（ゴムの伸び率を色で表現）を作成する"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- ▼▼▼ 色分け修正箇所 ▼▼▼ ---
        marker_ids = sorted(df_anim['id'].unique())
        
        # マーカーの数に応じてカラーマップを選択
        num_markers = len(marker_ids)
        if num_markers <= 10:
            cmap = plt.cm.get_cmap('tab10', num_markers)
        elif num_markers <= 20:
            cmap = plt.cm.get_cmap('tab20', num_markers)
        else: # 20個以上の場合
            cmap = plt.cm.get_cmap('gist_rainbow', num_markers)
        
        # 各マーカーIDに色を割り当て
        marker_colors = {mid: cmap(i) for i, mid in enumerate(marker_ids)}
        
        # 色を指定してプロットオブジェクトを作成
        plots = {mid: ax.plot([], [], [], marker='o', color=marker_colors[mid], label=f'ID {mid}')[0] for mid in marker_ids}
        # --- ▲▲▲ 修正ここまで ▲▲▲ ---

        lines = {name: ax.plot([], [], [], linewidth=3)[0] for name in LINES_TO_DRAW.keys()}
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

        palette = sns.color_palette("winter", n_colors=256)
        
        # 伸び率正規化のための最小/最大長を計算
        min_lengths, max_lengths = self._calculate_length_bounds(df_anim)

        def init():
            all_x, all_y, all_z = df_anim['x'].values, df_anim['y'].values, df_anim['z'].values
            max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max()
            mid_x, mid_y, mid_z = (all_x.max()+all_x.min())/2, (all_y.max()+all_y.min())/2, (all_z.max()+all_z.min())/2
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            ax.set_box_aspect([1,1,1])
            ax.set_title(title); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
            return list(plots.values()) + list(lines.values())

        def update(frame_value):
            current_data = df_anim[df_anim[time_col_name] == frame_value]
            if is_cycle:
                time_str = f'Gait Cycle: {frame_value:.1f} %'
            else:
                time_val = current_data["Time"].iloc[0] if not current_data.empty else 0
                time_str = f'Time: {time_val:.3f}s'

            positions = {row['id']: (row['x'], row['y'], row['z']) for _, row in current_data.iterrows()}

            for marker_id, plot in plots.items():
                if marker_id in positions:
                    x, y, z = positions[marker_id]
                    plot.set_data([x], [y]); plot.set_3d_properties([z])

            for name, (p1, p2) in LINES_TO_DRAW.items():
                if p1 in positions and p2 in positions:
                    pos1, pos2 = np.array(positions[p1]), np.array(positions[p2])
                    cur_len = np.linalg.norm(pos1 - pos2)
                    
                    stretch_ratio = np.clip((cur_len - min_lengths[name]) / (max_lengths[name] - min_lengths[name] + 1e-8), 0, 1)
                    color_idx = int(stretch_ratio * (len(palette) - 1))
                    
                    lines[name].set_color(palette[color_idx])
                    lines[name].set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
                    lines[name].set_3d_properties([pos1[2], pos2[2]])

            time_text.set_text(time_str)
            return list(plots.values()) + list(lines.values())

        ani = animation.FuncAnimation(fig, update, frames=sorted(df_anim[time_col_name].unique()),
                                      init_func=init, blit=False, interval=1000/frame_rate)
        plt.show()

    def _calculate_length_bounds(self, df):
        """データ全体から各線分の最小長と最大長を計算する"""
        min_lengths, max_lengths = {}, {}
        for name, (p1, p2) in LINES_TO_DRAW.items():
            coords1 = df[df['id'] == p1][['x','y','z']].values
            coords2 = df[df['id'] == p2][['x','y','z']].values
            if coords1.shape[0] == coords2.shape[0] and coords1.shape[0] > 0:
                lengths = np.linalg.norm(coords1 - coords2, axis=1)
                min_lengths[name] = lengths.min()
                max_lengths[name] = lengths.max()
            else: # データがない場合はデフォルト値
                min_lengths[name] = 0
                max_lengths[name] = 1
        return min_lengths, max_lengths

    def plot_segment_lengths(self, df_mean_cycle, seg_groups):
        """部位ごとに線分長のグラフを描画する"""
        gait_perc = sorted(df_mean_cycle['gait_cycle_%'].unique())
        for seg_name, lines_list in seg_groups.items():
            plt.figure(figsize=(10, 5))
            for line_name in lines_list:
                if line_name not in LINES_TO_DRAW: continue
                
                p1, p2 = LINES_TO_DRAW[line_name]
                df1 = df_mean_cycle[df_mean_cycle['id'] == p1].sort_values('gait_cycle_%')
                df2 = df_mean_cycle[df_mean_cycle['id'] == p2].sort_values('gait_cycle_%')

                if not df1.empty and not df2.empty:
                    lengths = np.linalg.norm(df1[['x','y','z']].values - df2[['x','y','z']].values, axis=1)
                    sns.lineplot(x=gait_perc, y=lengths, label=line_name)
            
            plt.title(f"{seg_name} Segment Length Over Gait Cycle")
            plt.xlabel("Gait Cycle (%)"); plt.ylabel("Segment Length (mm)")
            plt.legend(); plt.grid(True)
            plt.show()

def main():
    """メイン実行関数"""
    processor = DataProcessor(ref_marker_id=REFERENCE_MARKER_ID)
    visualizer = Visualizer()

    # データの読み込みと準備
    try:
        df_plot = processor.load_and_prepare_data(CORRECTED_OPTI_CSV_PATH, AXIS_MAPPING)
    except FileNotFoundError:
        print(f"Error: Corrected OptiTrack file not found at {CORRECTED_OPTI_CSV_PATH}")
        return

    # 実行モードの選択
    while True:
        mode = input("\nSelect analysis mode (1: Mean Gait Cycle, 2: Specific Time Range): ")
        if mode in ['1', '2']: break
        print("Invalid input. Please enter 1 or 2.")

    if mode == '1':
        df_anim = processor.calculate_mean_gait_cycle(df_plot, LABCHART_CYCLES_PATH, TIME_OFFSET)
        if df_anim is not None:
            # 平均化データをCSVに保存
            try:
                df_anim.to_csv(MEAN_CYCLE_OUTPUT_PATH, index=False)
                print(f"Mean cycle data saved to: {MEAN_CYCLE_OUTPUT_PATH}")
            except Exception as e:
                print(f"Could not save mean cycle data. Error: {e}")

            visualizer.animate_3d_plot(df_anim, 'Mean Gait Cycle Trajectory', 'gait_cycle_%', is_cycle=True, frame_rate=FRAME_RATE)
            visualizer.plot_segment_lengths(df_anim, SEGMENT_GROUPS)

    elif mode == '2':
        try:
            start_t = float(input("Enter animation start time (seconds): "))
            end_t = float(input("Enter animation end time (seconds): "))
        except ValueError:
            print("Invalid time format. Please enter numbers.")
            return

        df_anim = df_plot[(df_plot['Time'] >= start_t) & (df_plot['Time'] <= end_t)]
        if df_anim.empty:
            print("No data found in the specified time range.")
        else:
            visualizer.animate_3d_plot(df_anim, f'Raw Trajectory ({start_t}s - {end_t}s)', 'Frame', is_cycle=False, frame_rate=FRAME_RATE)

if __name__ == '__main__':
    main()
