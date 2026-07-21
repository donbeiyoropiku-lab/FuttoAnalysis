'''
pre_analysis_checker.py

### **使用方法**

1.  **ファイルの保存:**
    * 上記のコードを **`pre_analysis_checker.py`** という名前で保存します。

2.  **`task2.csv` の解析:**
    * スクリプト上部の設定部分で、`RAW_OPTI_CSV_PATH` が `task2.csv` の正しいパスになっていることを確認します。
        ```python
        RAW_OPTI_CSV_PATH = r"C:\FuttoAnalysis\opti\20251020\task2.csv"
        ```
    * ターミナル（コマンドプロンプトやPowerShell）でスクリプトを実行します。
        ```bash
        python pre_analysis_checker.py
        ```
    * **コンソールに出力される結果を確認:** 提案された安定時間と15個のマーカーIDをメモします。
    * **3DマップでIDを特定:** 表示された3Dマップを回転させ、各マーカーをクリックしてIDを確認し、`LINES_TO_DRAW` に設定すべきIDの対応関係を特定します。
    * 完了したら、3Dマップのウィンドウを閉じます。

3.  **`task3.csv` の解析:**
    * `pre_analysis_checker.py` の `RAW_OPTI_CSV_PATH` の行を `task3.csv` のパスに書き換えます。
        ```python
        RAW_OPTI_CSV_PATH = r"C:\FuttoAnalysis\opti\20260217\task01.csv"
        ```
    * 再度、ターミナルでスクリプトを実行します。
        ```bash
        python pre_analysis_checker.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from collections import defaultdict

# --- ▼▼▼ 設定 ▼▼▼ ---
RAW_OPTI_CSV_PATH = r"C:\FuttoAnalysis\opti\20260217\task01.csv"
STATIC_CANDIDATE_WINDOW_END = 12.0
SEGMENT_COUNTS = {'Hip': 4, 'Thigh': 1, 'Knee': 4, 'Shank': 1, 'Foot': 5}
SEGMENT_ORDER = ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']
# --- ▲▲▲ 設定ここまで ▲▲▲ ---


class PreAnalysisChecker:
    """
    OptiTrack生データを解析し、安定区間、安定IDを見つけ、
    インタラクティブ3Dプロット、マーカー存在CSV、
    および config.py 用のテンプレートテキスト(LINES_TO_DRAW含む)を出力する。
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df_long = self._load_opti_data_to_long()
        if self.df_long is None or self.df_long.empty:
            print("エラー: データ読み込み失敗またはデータが空です。")
            return

        self.stable_window, self.stable_ids = self._find_stable_window()

        if self.stable_window and self.stable_ids:
            self.stable_df = self.df_long[
                (self.df_long['Time'] >= self.stable_window[0]) &
                (self.df_long['Time'] <= self.stable_window[1]) &
                (self.df_long['id'].isin(self.stable_ids))
            ].copy()

            if not self.stable_df.empty:
                self.mean_pos = self.stable_df.groupby('id')[['x', 'y', 'z']].mean()
                if not self.mean_pos.empty:
                    self.segments, self.id_to_segment = self._segment_markers_by_y(self.mean_pos)
                    # config テキスト生成 (mean_posが必要)
                    self._generate_config_text(self.mean_pos)
                    self._create_interactive_3d_plot()
                else:
                     print("エラー: 平均座標を計算できませんでした。")
            else:
                 print("エラー: 安定区間からデータを抽出できませんでした。")
        else:
             print("安定区間が見つからなかったため、configテンプレート生成と3Dプロットはスキップします。")

        self._create_wide_csv()

    # --- データ読み込み、安定区間探索、セグメント分け (変更なし) ---
    def _load_opti_data_to_long(self):
        # (... v21 と同様 ...)
        if not os.path.exists(self.file_path): print(f"エラー: ファイルなし: {self.file_path}"); return None
        print(f"'{os.path.basename(self.file_path)}' を読み込み中...")
        rows = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for _ in range(43): next(f) # ヘッダー
                for line_num, line in enumerate(f, 44):
                    parts = line.strip().split(',')
                    try:
                        if len(parts) < 5: continue
                        frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4]); base_col = 5
                        if len(parts) >= base_col + n_markers * 4:
                            for i in range(n_markers):
                                x = float(parts[base_col + 4*i]) * 1000.0; y = float(parts[base_col + 4*i + 1]) * 1000.0
                                z = float(parts[base_col + 4*i + 2]) * 1000.0; mid = int(parts[base_col + 4*i + 3])
                                rows.append((frame, t, mid, x, y, z))
                    except (ValueError, IndexError): continue
            if not rows: print("エラー: 有効行なし。"); return None
            out_df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"]); print(f"読み込み成功: {len(out_df)} 行"); return out_df
        except Exception as e: print(f"読み込みエラー: {e}"); return None

    def _find_stable_window(self):
        # (... v21 と同様 ...)
        print("\n--- 安定区間の探索 ---")
        static_candidate_df = self.df_long[self.df_long['Time'] <= STATIC_CANDIDATE_WINDOW_END].copy()
        if static_candidate_df.empty: print("エラー: 指定時間内にデータなし。"); return None, None
        ids_per_time = static_candidate_df.groupby('Time')['id'].apply(set)
        sets_of_15 = ids_per_time[ids_per_time.apply(len) == 15]
        if sets_of_15.empty: print("警告: 15個同時取得の時間なし。"); return None, None
        try: most_common_set = sets_of_15.apply(frozenset).value_counts().idxmax()
        except ValueError: print("警告: 15個セットが見つかりませんでした。"); return None, None
        stable_times = sets_of_15[sets_of_15.apply(frozenset) == most_common_set].index
        if len(stable_times) < 10: print("警告: 安定区間短すぎ。"); return None, None
        start_time, end_time = stable_times.min(), stable_times.max()
        stable_ids = sorted(list(most_common_set))
        print("\n【結果】")
        print(f"✅ 安定区間検出: {start_time:.3f}s - {end_time:.3f}s")
        print(f"   安定マーカーID ({len(stable_ids)}個):"); print(f"   {stable_ids}")
        return (start_time, end_time), stable_ids

    def _segment_markers_by_y(self, mean_pos_df):
        # (... v21 と同様 ...)
        print("\n--- Y座標による自動セグメント分け ---")
        if sum(SEGMENT_COUNTS.values()) != 15: print("エラー: SEGMENT_COUNTS 合計≠15。"); return {}, {}
        if set(mean_pos_df.index) != set(self.stable_ids):
             print("警告: 安定IDと平均座標ID不一致。")
             ids_to_sort = list(set(mean_pos_df.index).intersection(self.stable_ids))
             if len(ids_to_sort) < 15: print(f"エラー: セグメント分けID不足 ({len(ids_to_sort)}/15)。"); return {}, {}
             sorted_ids = mean_pos_df.loc[ids_to_sort].sort_values('y', ascending=False).index.tolist()
        else: sorted_ids = mean_pos_df.sort_values('y', ascending=False).index.tolist()
        segments = {}; id_to_segment = {}; current_index = 0
        for name in SEGMENT_ORDER:
            count = SEGMENT_COUNTS[name]
            if current_index + count > len(sorted_ids):
                 print(f"エラー: セグメント '{name}' 割り当てID不足。"); segment_ids = sorted_ids[current_index:]; count = len(segment_ids)
            else: segment_ids = sorted_ids[current_index : current_index + count]
            segments[name] = segment_ids; [id_to_segment.update({mid: name}) for mid in segment_ids]
            print(f"  {name} ({count}個): {segment_ids}"); current_index += count
            if current_index >= len(sorted_ids): break
        return segments, id_to_segment

    # --- ▼▼▼【関数修正】configテキスト生成 (LINES_TO_DRAW 自動割り当て) ▼▼▼ ---
    def _generate_config_text(self, mean_pos_df): # mean_pos_df を引数で受け取る
        """ config.py に貼り付けるためのテキストを生成・出力する """
        print("\n" + "="*30)
        print(" config.py 用テンプレート")
        print("="*30 + "\n")
        print("# --- ▼▼▼ 以下を config.py の taskX 設定に貼り付け ▼▼▼ ---")
        print("# (パスや時刻は手動で確認・修正してください)")

        # --- SEGMENTS ---
        print("\n'SEGMENTS': {")
        seg_items = [f"    '{name}': {self.segments.get(name, [])}," for name in SEGMENT_ORDER]
        print('\n'.join(seg_items))
        print("},")

        # --- KEYFRAME_MAP (セグメント順) ---
        print("\n'KEYFRAME_MAP': { # ★ 要手動確認/修正 ★")
        map_items = []
        for seg_name in SEGMENT_ORDER:
            segment_ids = self.segments.get(seg_name, [])
            if segment_ids:
                 map_items.append(f"    # --- {seg_name} ---")
                 for mid in segment_ids:
                      if mid in self.stable_ids: map_items.append(f"    {mid}: {mid},")
        print('\n'.join(map_items))
        print("},")

        # --- LINES_TO_DRAW (ルールベース自動割り当て) ---
        print("\n'LINES_TO_DRAW': { # ★ 要手動確認/修正 (ルールベース自動割り当て) ★")
        lines_draw_items = []
        ids = {} # 各ルールで特定したIDを格納する辞書

        try:
            # --- ルールに基づいてIDを特定 ---
            # セグメントごとの座標データを取得
            segment_coords = {}
            for seg_name, seg_ids in self.segments.items():
                segment_coords[seg_name] = mean_pos_df.loc[seg_ids]

            # Hip (z小2点)
            hip_z_sorted = segment_coords['Hip'].sort_values('z')
            hip_z_low2 = hip_z_sorted.iloc[:2]
            ids['hip_z_low_x_high'] = hip_z_low2.sort_values('x', ascending=False).index[0] # x最大
            ids['hip_z_low_x_low'] = hip_z_low2.sort_values('x', ascending=True).index[0]  # x最小

            # Hip (z大2点)
            hip_z_high2 = hip_z_sorted.iloc[2:]
            ids['hip_z_high_x_high'] = hip_z_high2.sort_values('x', ascending=False).index[0] # x最大
            ids['hip_z_high_x_low'] = hip_z_high2.sort_values('x', ascending=True).index[0]  # x最小

            # Knee
            knee_coords = segment_coords['Knee']
            ids['knee_y_high'] = knee_coords.sort_values('y', ascending=False).index[0] # y最大
            ids['knee_y_low'] = knee_coords.sort_values('y', ascending=True).index[0]  # y最小
            ids['knee_x_low'] = knee_coords.sort_values('x', ascending=True).index[0]  # x最小
            ids['knee_x_high'] = knee_coords.sort_values('x', ascending=False).index[0] # x最大

            # Thigh (1点のみ)
            ids['thigh'] = segment_coords['Thigh'].index[0]

            # Shank (1点のみ)
            ids['shank'] = segment_coords['Shank'].index[0]

            # Foot
            foot_coords = segment_coords['Foot']
            ids['foot_y_high'] = foot_coords.sort_values('y', ascending=False).index[0] # y最大

            # Foot (x小2点)
            foot_x_low2 = foot_coords.sort_values('x').iloc[:2]
            ids['foot_x_low_z_low'] = foot_x_low2.sort_values('z').index[0] # z最小
            ids['foot_x_low_z_high'] = foot_x_low2.sort_values('z', ascending=False).index[0] # z最大

            # Foot (x大2点)
            foot_x_high2 = foot_coords.sort_values('x', ascending=False).iloc[:2]
            ids['foot_x_high_z_low'] = foot_x_high2.sort_values('z').index[0] # z最小
            ids['foot_x_high_z_high'] = foot_x_high2.sort_values('z', ascending=False).index[0] # z最大

            # --- LINES_TO_DRAW 辞書を作成 ---
            lines_draw_items.append(f'    "Front_Upper_In": ({ids["hip_z_low_x_high"]}, {ids["knee_y_high"]}),')
            lines_draw_items.append(f'    "Front_Upper_Out": ({ids["hip_z_low_x_low"]}, {ids["knee_y_high"]}),')
            lines_draw_items.append(f'    "Front_Knee_Upper_Out": ({ids["knee_y_high"]}, {ids["knee_x_low"]}),')
            lines_draw_items.append(f'    "Front_Knee_Upper_In": ({ids["knee_y_high"]}, {ids["knee_x_high"]}),')
            lines_draw_items.append(f'    "Front_Knee_Lower_Out": ({ids["knee_y_low"]}, {ids["knee_x_low"]}),')
            lines_draw_items.append(f'    "Front_Knee_Lower_In": ({ids["knee_y_low"]}, {ids["knee_x_high"]}),')
            lines_draw_items.append(f'    "Front_Shin": ({ids["knee_y_low"]}, {ids["foot_y_high"]}),')
            lines_draw_items.append(f'    "Toe_Out": ({ids["foot_y_high"]}, {ids["foot_x_low_z_low"]}),')
            lines_draw_items.append(f'    "Toe_In": ({ids["foot_y_high"]}, {ids["foot_x_high_z_low"]}),') # x大 z小
            lines_draw_items.append(f'    "Back_Upper_In": ({ids["hip_z_high_x_high"]}, {ids["thigh"]}),')
            lines_draw_items.append(f'    "Back_Upper_Out": ({ids["hip_z_high_x_low"]}, {ids["thigh"]}),')
            lines_draw_items.append(f'    "Back_Thigh_Out": ({ids["thigh"]}, {ids["knee_x_low"]}),')
            lines_draw_items.append(f'    "Back_Thigh_In": ({ids["thigh"]}, {ids["knee_x_high"]}),')
            lines_draw_items.append(f'    "Back_Knee_Out": ({ids["knee_x_low"]}, {ids["shank"]}),') # Back_knee_Out
            lines_draw_items.append(f'    "Back_Knee_In": ({ids["knee_x_high"]}, {ids["shank"]}),') # Back_knee_In
            lines_draw_items.append(f'    "Back_Shin_Out": ({ids["shank"]}, {ids["foot_x_low_z_high"]}),') # Shankと x小z大
            lines_draw_items.append(f'    "Back_Shin_In": ({ids["shank"]}, {ids["foot_x_high_z_high"]}),') # Shankと x大z大

        except (KeyError, IndexError) as e:
            print(f"エラー: LINES_TO_DRAW 自動割り当て中にエラーが発生しました: {e}")
            print("       セグメントのマーカー数が不足しているか、想定外の位置関係の可能性があります。")
            lines_draw_items.append("    # --- 自動割り当て失敗 ---")

        print('\n'.join(lines_draw_items))
        print("},")

        # --- MUSCLE_INDICATORS (暫定 - IDのみ更新) ---
        print("\n'MUSCLE_INDICATORS': { # ★ 要手動確認/修正 (IDは自動割り当て) ★")
        muscle_definitions_template = { # (変更なし)
            "Tibialis Anterior": {'emg':"L_TA_mean", 'type':'midpoint', 'ids_key':['knee_y_low','foot_y_high']},
            "Soleus": {'emg':"L_SOL_mean", 'type':'centroid', 'ids_key':['shank','foot_x_high_z_high','foot_x_low_z_high']}, # Shank, Foot x大z高, Foot x小z高
            "Rectus Femoris": {'emg':"L_RF_mean", 'type':'centroid', 'ids_key':['hip_z_low_x_high','knee_y_high','hip_z_low_x_low']}, # Hip z低x高, Knee y高, Hip z低x低
            "Vastus Lateralis": {'emg':"L_VL_mean", 'type':'weighted_midpoint', 'ids_key':['hip_z_low_x_low','knee_y_high'], 'weight':0.25}, # Hip z低x低, Knee y高
            "Biceps Femoris": {'emg':"L_BF_mean", 'type':'midpoint', 'ids_key':['thigh','knee_x_low']}, # Thigh, Knee x低
            "Semitendinosus": {'emg':"L_ST_mean", 'type':'single', 'ids_key':['thigh']}, # Thigh
            "Gluteus Maximus Area": {'emg':"L_GM_mean", 'type':'offset', 'ids_key':['hip_z_high_x_high','thigh','hip_z_high_x_low'], 'ref_ids_key':['hip_z_high_x_high','thigh'], 'weight':0.3}, # Hip z高x高, Thigh, Hip z高x低
            "Iliopsoas Area": {'emg':"L_ILIO_mean", 'type':'midpoint', 'ids_key':['hip_z_low_x_high','hip_z_low_x_low']}, # Hip z低x高, Hip z低x低
        }
        muscle_items = []
        for name, definition in muscle_definitions_template.items():
            marker_ids_val = [ids.get(key, "???") for key in definition['ids_key']]
            item_str = f'    "{name}": {{\'emg_col\': "{definition["emg"]}", \'type\': \'{definition["type"]}\', \'markers\': {marker_ids_val}'
            if 'weight' in definition: item_str += f', \'weight\': {definition["weight"]}'
            if 'ref_ids_key' in definition:
                ref_ids_val = [ids.get(key, "???") for key in definition['ref_ids_key']]
                item_str += f', \'ref_marker\': {ref_ids_val}'
            item_str += '},'
            muscle_items.append(item_str)
        print('\n'.join(muscle_items))
        print("}")

        print("\n# --- ▲▲▲ ここまでをコピー ▲▲▲ ---")
        print("\n" + "="*30 + "\n")
    # --- ▲▲▲ 関数修正ここまで ▲▲▲ ---


    def _create_interactive_3d_plot(self):
        """ インタラクティブ3Dプロット作成 """
        # (... v21 と同様 ...)
        print("\n--- 3Dマップの生成 ---")
        print("グラフ上のマーカーをクリックするとIDが表示されます。")
        fig = plt.figure(figsize=(14, 10)); ax = fig.add_subplot(111, projection='3d')
        if not hasattr(self, 'mean_pos') or self.mean_pos.empty: print("エラー: 平均座標なし。プロット不可。"); return
        mean_pos = self.mean_pos; unique_ids = sorted(mean_pos.index);
        if not unique_ids: print("エラー: ユニークIDなし。プロット不可。"); return
        cmap = plt.cm.get_cmap('tab20' if len(unique_ids) > 10 else 'tab10', len(unique_ids))
        id_colors = {uid: cmap(i) for i, uid in enumerate(unique_ids)}; color_array = [id_colors[uid] for uid in mean_pos.index]
        points = ax.scatter(mean_pos['x'], mean_pos['y'], mean_pos['z'], c=color_array, s=50, picker=5)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'ID {mid}', markerfacecolor=id_colors[mid], markersize=8) for mid in unique_ids]
        ax.legend(handles=legend_elements, title="Marker IDs", bbox_to_anchor=(1.02, 1), loc='upper left')
        annotation = ax.text(0, 0, 0, "", bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5), visible=False)
        def on_pick(event):
            inds = event.ind;
            if not inds: return True
            ind = inds[0]; xs, ys, zs = event.artist._offsets3d; x, y, z = xs[ind], ys[ind], zs[ind]
            marker_id = mean_pos.index[ind]; annotation.set_position((x, y, z)); annotation.set_text(f"ID: {marker_id}"); annotation.set_visible(True)
            fig.canvas.draw_idle(); return True
        fig.canvas.mpl_connect('pick_event', on_pick)
        ax.set_title(f'Interactive 3D Marker Map ({os.path.basename(self.file_path)})')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        all_coords = mean_pos[['x', 'y', 'z']].values; x_coords, y_coords, z_coords = all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]
        x_range = x_coords.max() - x_coords.min(); y_range = y_coords.max() - y_coords.min(); z_range = z_coords.max() - z_coords.min()
        max_range = np.array([x_range, y_range, z_range]).max()
        mid_x, mid_y, mid_z = (x_coords.max() + x_coords.min())/2, (y_coords.max() + y_coords.min())/2, (z_coords.max() + z_coords.min())/2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2); ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2); ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def _create_wide_csv(self):
        """ Wide形式CSV作成 """
        # (... v21 と同様 ...)
        if self.df_long is None or self.df_long.empty: print("データなし、Wide形式CSVスキップ。"); return
        try:
            pivot_df = self.df_long.pivot_table(index='Time', columns='id', aggfunc='size', fill_value=0)
            pivot_df = (pivot_df > 0).astype(int); base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(os.path.dirname(self.file_path), f"{base_name}_marker_presence.csv")
            pivot_df.to_csv(output_path); print(f"\nWide形式マーカー存在CSV出力: {output_path}")
        except Exception as e: print(f"Wide形式CSV出力エラー: {e}")


if __name__ == "__main__":
    if not os.path.exists(RAW_OPTI_CSV_PATH):
        print(f"エラー: 指定ファイルパスなし: {RAW_OPTI_CSV_PATH}")
    else:
        try:
            checker = PreAnalysisChecker(RAW_OPTI_CSV_PATH)
            if hasattr(checker, 'df_long') and checker.df_long is not None: print("\nPre-analysis check completed.")
            else: print("\nPre-analysis check failed or interrupted.")
        except Exception as e: print(f"\nスクリプト実行中に予期せぬエラー: {e}")