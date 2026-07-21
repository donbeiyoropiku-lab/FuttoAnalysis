# =============================================================================
# create_anime_grad.py (代表データ確認機能付き 完全版)
#
# モード:
#   1: ★ 新機能: 保存済みデータのアニメーション確認 (代表データ / 平均化データ)
#   2: 指定時間区間アニメーション (生データ) - トラッキング結果の動作確認用
#   3: 指定区間平均化 (範囲を手動指定) - 特定区間のテスト用
#   4: 5フェーズ一括平均化処理 (自動で分割保存) - 本番用
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import os
import glob
import sys

# --- CONFIG.py のパス解決 ---
# このスクリプトは C:\FuttoAnalysis\2026_analysis\ に置かれている想定。
# CONFIG.py は C:\FuttoAnalysis\2026_analysis\futto_common\ に存在する。
_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futto_common')
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import CONFIG

class DataProcessor:
    def __init__(self, ref_marker_id):
        self.ref_marker_id = ref_marker_id

    def load_and_prepare_data(self, opti_path, axis_mapping):
        print(f"OptiTrackデータを読み込み中: {opti_path}")
        try:
            df_corrected = pd.read_csv(opti_path)
        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return None

        df_plot = pd.DataFrame()
        for col in ['Frame', 'Time', 'id']:
            if col in df_corrected.columns:
                df_plot[col] = df_corrected[col]
        for final_axis, source_axis in axis_mapping.items():
            col_name = final_axis.replace('final_', '')
            if source_axis.startswith('-'):
                if source_axis[1:] in df_corrected.columns:
                    df_plot[col_name] = -df_corrected[source_axis[1:]]
            else:
                if source_axis in df_corrected.columns:
                    df_plot[col_name] = df_corrected[source_axis]
        return df_plot

    def _normalize_trajectory(self, df, num_points=101):
        time_orig = df['Time'].to_numpy()
        if time_orig.min() == time_orig.max():
             time_norm_abs = np.full(num_points, time_orig.min())
        else:
             time_norm_abs = np.linspace(time_orig.min(), time_orig.max(), num_points)

        df_normalized = pd.DataFrame(index=range(num_points))

        for axis in ['x', 'y', 'z']:
            axis_data = df[axis]
            if len(axis_data.unique()) == 1 or len(time_orig) < 2:
                df_normalized[axis] = axis_data.iloc[0]
            else:
                try:
                    f = interp1d(time_orig, axis_data, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    df_normalized[axis] = f(time_norm_abs)
                except ValueError:
                    f_linear = interp1d(time_orig, axis_data, kind='linear', bounds_error=False, fill_value='extrapolate')
                    df_normalized[axis] = f_linear(time_norm_abs)
        return df_normalized

    def calculate_mean_gait_cycle(self, df_plot, cycles_path, time_offset, filter_start_time=None, filter_end_time=None):
        try:
            df_cycles_all = pd.read_csv(cycles_path)
        except Exception as e:
            print(f"歩行周期ファイルの読み込みエラー: {e}"); return None

        if filter_start_time is not None and filter_end_time is not None:
            df_cycles_all['opti_start_time'] = df_cycles_all['hs_time'] - time_offset
            df_cycles_all['opti_end_time'] = df_cycles_all['next_hs_time'] - time_offset
            valid_cycles_mask = (df_cycles_all['opti_start_time'] >= filter_start_time) & \
                                (df_cycles_all['opti_end_time'] <= filter_end_time)
            df_cycles = df_cycles_all[valid_cycles_mask].copy()
            if df_cycles.empty: 
                print(f"警告: 指定範囲 ({filter_start_time}s - {filter_end_time}s) に有効な歩行周期がありません。")
                return None
        else:
            df_cycles = df_cycles_all

        all_cycles_rel = {int(mid): [] for mid in df_plot['id'].unique()}
        all_cycles_ref_abs = []
        valid_cycle_count = 0

        for _, cycle in df_cycles.iterrows():
            start_time = float(cycle.get('opti_start_time', cycle['hs_time'] - time_offset))
            end_time = float(cycle.get('opti_end_time', cycle['next_hs_time'] - time_offset))
            buffer = 1.0 / CONFIG.FRAME_RATE
            cycle_df = df_plot[(df_plot['Time'] >= start_time - buffer) & (df_plot['Time'] <= end_time + buffer)].copy()

            ref_traj_cycle = cycle_df[cycle_df['id'] == self.ref_marker_id]
            if ref_traj_cycle.empty or len(ref_traj_cycle) < 4: continue

            normalized_ref = self._normalize_trajectory(ref_traj_cycle)
            if normalized_ref.isnull().values.any(): continue
            all_cycles_ref_abs.append(normalized_ref)

            valid_marker_in_cycle = 0
            for marker_id_float, group in cycle_df.groupby('id'):
                marker_id = int(marker_id_float)
                if marker_id == self.ref_marker_id or len(group) < 4: continue

                merged = pd.merge_asof(
                    group.sort_values('Time'),
                    ref_traj_cycle[['Time', 'x', 'y', 'z']].sort_values('Time'),
                    on='Time', direction='nearest', suffixes=('_marker', '_ref')
                )
                
                rel_group = pd.DataFrame({'Time': merged['Time']})
                for axis in ['x', 'y', 'z']:
                    rel_group[axis] = merged[f'{axis}_marker'] - merged[f'{axis}_ref']

                if not rel_group.empty:
                    normalized_rel = self._normalize_trajectory(rel_group)
                    if not normalized_rel.isnull().values.any():
                        all_cycles_rel[marker_id].append(normalized_rel)
                        valid_marker_in_cycle += 1

            if valid_marker_in_cycle > 0: valid_cycle_count += 1

        if len(all_cycles_ref_abs) < 2: return None

        stacked_ref_trajs = np.stack([df[['x','y','z']].values for df in all_cycles_ref_abs])
        mean_ref_abs_traj = np.mean(stacked_ref_trajs, axis=0)

        mean_rel_trajectories = {}
        for mid, traj_list in all_cycles_rel.items():
            if mid != self.ref_marker_id and len(traj_list) >= 2:
                stacked_trajs = np.stack([df[['x','y','z']].values for df in traj_list])
                mean_rel_trajectories[mid] = np.mean(stacked_trajs, axis=0)

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

        return pd.concat(mean_df_list, ignore_index=True)


class Visualizer:
    def animate_3d_plot(self, df_anim, title, time_col_name, is_cycle, lines_to_draw):
        print(f"アニメーションを描画中... ({title})")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 軸の範囲設定
        x_min, x_max = df_anim['x'].quantile(0.01), df_anim['x'].quantile(0.99)
        y_min, y_max = df_anim['y'].quantile(0.01), df_anim['y'].quantile(0.99)
        z_min, z_max = df_anim['z'].quantile(0.01), df_anim['z'].quantile(0.99)
        
        max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
        if max_range == 0 or np.isnan(max_range): max_range = 500
        
        mid_x = (x_max+x_min) * 0.5
        mid_y = (y_max+y_min) * 0.5
        mid_z = (z_max+z_min) * 0.5
        
        # 少し引いた視点にするために1.2倍
        ax.set_xlim(mid_x - max_range*1.2, mid_x + max_range*1.2)
        ax.set_ylim(mid_y - max_range*1.2, mid_y + max_range*1.2)
        ax.set_zlim(mid_z - max_range*1.2, mid_z + max_range*1.2)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
        scatter = ax.scatter([], [], [], c='blue', s=30, alpha=0.6)
        lines = {name: ax.plot([], [], [], color='red', linewidth=1.5, alpha=0.7)[0] for name in lines_to_draw.keys()}
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        frames = sorted(df_anim[time_col_name].unique())
        
        def update(frame_val):
            current_df = df_anim[df_anim[time_col_name] == frame_val]
            
            scatter._offsets3d = (current_df['x'].values, current_df['y'].values, current_df['z'].values)
            
            for line_name, (id1, id2) in lines_to_draw.items():
                p1 = current_df[current_df['id'] == id1]
                p2 = current_df[current_df['id'] == id2]
                if not p1.empty and not p2.empty:
                    x_data = [p1['x'].values[0], p2['x'].values[0]]
                    y_data = [p1['y'].values[0], p2['y'].values[0]]
                    z_data = [p1['z'].values[0], p2['z'].values[0]]
                    lines[line_name].set_data(x_data, y_data)
                    lines[line_name].set_3d_properties(z_data)
                else:
                    lines[line_name].set_data([], [])
                    lines[line_name].set_3d_properties([])
                    
            unit = '%' if is_cycle else 's'
            time_text.set_text(f'{time_col_name}: {frame_val:.2f} {unit}')
            
            return [scatter, time_text] + list(lines.values())

        # interval: 生データなら約10ms (100Hz), 平均周期なら40ms
        interval_ms = 10 if not is_cycle else 40
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
        plt.show()


def main():
    while True:
        task_key = input("解析するタスク名を入力してください (task01, task02, task03): ").lower()
        if task_key in CONFIG.TASK_CONFIGS:
            cfg = CONFIG.TASK_CONFIGS[task_key]
            break
        else:
            print("エラー: 設定ファイルにタスクが見つかりません。")

    # task03はFutto非着用のためLINES_TO_DRAWが空 → アニメーションはマーカー点のみ表示
    # AXIS_MAPPING: CONFIGに定義がなければ恒等写像をデフォルト使用
    axis_mapping = getattr(CONFIG, 'AXIS_MAPPING', {
        'final_x': 'x', 'final_y': 'y', 'final_z': 'z'
    })

    processor = DataProcessor(ref_marker_id=cfg['REFERENCE_MARKER_ID'])
    visualizer = Visualizer() 
    
    while True:
        print("\n--- 操作を選択してください ---")
        print("  1: ★ 保存済みデータのアニメーション確認 (代表データ / 平均データ)")
        print("  2: 指定時間区間アニメーション (生データ)")
        print("  3: 指定区間平均化 (範囲を手動指定)")
        print("  4: 5フェーズ一括平均化処理 (自動で分割保存)")
        print("  q: 終了")
        mode = input("実行するモード [1/2/3/4/q]: ").lower()

        if mode == '1':
            # モード1: 保存済みCSVの再生
            print("\n--- 保存済みデータの確認 ---")
            base_path = cfg.get('MEAN_CYCLE_BASE_PATH')
            dir_name = os.path.dirname(base_path)
            file_pattern = f"{os.path.basename(base_path)}*.csv"
            
            # ディレクトリ内の _representative.csv などを検索
            files = glob.glob(os.path.join(dir_name, file_pattern))
            
            if not files:
                print(f"エラー: {dir_name} 内に保存されたデータが見つかりません。")
                continue
                
            print("見つかったファイル:")
            for i, f in enumerate(files):
                print(f"  {i+1}: {os.path.basename(f)}")
                
            try:
                idx = int(input(f"確認するファイルの番号を選択 (1-{len(files)}): ")) - 1
                if 0 <= idx < len(files):
                    target_file = files[idx]
                    print(f"\n{os.path.basename(target_file)} を読み込み中...")
                    df_anim = pd.read_csv(target_file)
                    title = f'{task_key} : {os.path.basename(target_file)}'
                    visualizer.animate_3d_plot(df_anim, title, 'gait_cycle_%', True, cfg.get('LINES_TO_DRAW', {}))
                else:
                    print("無効な番号です。")
            except ValueError:
                print("数値を入力してください。")

        elif mode in ['2', '3', '4']:
            # モード2〜4は元のOptiTrackデータ(.csv)が必要なのでここでロード
            if 'df_plot' not in locals() or df_plot is None:
                df_plot = processor.load_and_prepare_data(cfg['OUTPUT_CSV_PATH'], axis_mapping)
                if df_plot is None: continue

            if mode == '2':
                # モード2: 指定区間の生データアニメーション
                print("\n--- 生データアニメーション ---")
                try:
                    start_t = float(input(f"開始時刻を入力 (例: {cfg['T1_WALK_START']}): "))
                    end_t = float(input(f"終了時刻を入力 (例: {cfg['T1_WALK_START'] + 5}): "))
                except ValueError:
                    print("エラー: 数値を入力してください。"); continue
                    
                df_anim = df_plot[(df_plot['Time'] >= start_t) & (df_plot['Time'] <= end_t)].copy()
                if df_anim.empty:
                    print("エラー: 指定された区間にデータがありません。"); continue
                    
                title = f'{task_key} Raw Data ({start_t}s - {end_t}s)'
                visualizer.animate_3d_plot(df_anim, title, 'Time', False, cfg.get('LINES_TO_DRAW', {}))
                
            elif mode == '3':
                # モード3: 手動での区間指定と平均化
                print("\n--- 手動区間平均化 ---")
                try:
                    start_t = float(input("抽出開始時刻を入力 (例: 40.0): "))
                    end_t = float(input("抽出終了時刻を入力 (例: 100.0): "))
                except ValueError:
                    print("エラー: 数値を入力してください。"); continue
                    
                df_anim = processor.calculate_mean_gait_cycle(
                    df_plot, cfg['LABCHART_CYCLES_PATH'], CONFIG.TIME_OFFSET,
                    filter_start_time=start_t, filter_end_time=end_t
                )
                
                if df_anim is not None and not df_anim.empty:
                    title = f'{task_key} Mean Gait Cycle ({start_t}s - {end_t}s)'
                    
                    save_path = cfg['MEAN_CYCLE_BASE_PATH'] + f"_manual_{start_t}-{end_t}.csv"
                    ans = input(f"データを保存しますか？ [y/N] (パス: {os.path.basename(save_path)}): ").lower()
                    if ans == 'y':
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        df_anim.to_csv(save_path, index=False)
                        print("✓ 保存しました。")
                    
                    visualizer.animate_3d_plot(df_anim, title, 'gait_cycle_%', True, cfg.get('LINES_TO_DRAW', {}))
                    
            elif mode == '4':
                # モード4: 全フェーズの一括処理と保存
                print("\n--- 5フェーズの一括処理を開始します ---")
                # PHASESはタスク共通のグローバル設定 (CONFIG.PHASES)
                phases = CONFIG.PHASES
                if not phases:
                    print("エラー: CONFIG.py に PHASES が定義されていません。")
                    continue

                for phase_id, phase_info in phases.items():
                    phase_name = phase_info['name']
                    start_t    = phase_info['start']
                    end_t      = phase_info['end']
                    print(f"\n>> 処理中: Phase {phase_id} ({phase_name}, {start_t}s - {end_t}s)")
                    
                    df_anim = processor.calculate_mean_gait_cycle(
                        df_plot, cfg['LABCHART_CYCLES_PATH'], CONFIG.TIME_OFFSET,
                        filter_start_time=start_t, filter_end_time=end_t
                    )
                    
                    if df_anim is not None and not df_anim.empty:
                        base_path = cfg['MEAN_CYCLE_BASE_PATH']
                        # phase_name に "/" が含まれるとWindowsでパス区切りになるため除去
                        safe_name = phase_name.replace('/', '')
                        # 例: task01_mean_cycle_Phase1_0.7ms.csv
                        output_path = f"{base_path}_Phase{phase_id}_{safe_name}.csv"
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        df_anim.to_csv(output_path, index=False)
                        print(f"✓ 保存完了: {os.path.basename(output_path)}")
                    else:
                        print(f"✗ 失敗: {phase_name} のデータ生成に失敗しました。")
                
                print("\n全フェーズのバッチ処理が完了しました！")
                
        elif mode == 'q':
            print("プログラムを終了します。")
            break
        else:
            print("無効な入力です。1, 2, 3, 4, q のいずれかを入力してください。")

if __name__ == '__main__':
    main()