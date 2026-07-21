'''
pre_analysis_checker.py  (CONFIG連携版)

### 変更点
- TASK_KEY を指定するだけで CONFIG.py からマーカー数・セグメント設定を自動読み込み
- ハードコードの切り替え作業が不要になった
- task01(15個) / task02(8個) / task03(5個) すべて自動対応

### 使用方法
1. TASK_KEY を解析したいタスク名に変更して実行するだけ。
2. 出力されるCONFIG.pyテンプレートを CONFIG.py の該当タスク設定に貼り付ける。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from collections import defaultdict

# --- CONFIG.py のパス解決 ---
# このスクリプトは C:\FuttoAnalysis\2026_analysis\ に置かれている想定。
# CONFIG.py は C:\FuttoAnalysis\2026_analysis\futto_common\ に存在する。
_COMMON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futto_common')
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import CONFIG  # 設定モジュール

# =============================================================================
# ▼▼▼ ここだけ変更 ▼▼▼
TASK_KEY = 'task03'   # 'task01' / 'task02' / 'task03'
# ▲▲▲ ここだけ変更 ▲▲▲
# =============================================================================

# --- CONFIG から設定を自動取得 ---
_cfg = CONFIG.TASK_CONFIGS[TASK_KEY]
RAW_OPTI_CSV_PATH         = _cfg['OPTI_CSV_PATH']
STATIC_CANDIDATE_WINDOW_END = 38  # 最初の40s静止立位より少し手前

# マーカー数はセグメント定義から自動計算
_seg_counts_raw = {seg: len(ids) for seg, ids in _cfg['SEGMENTS'].items()}
EXPECTED_MARKER_COUNT = sum(_seg_counts_raw.values())
SEGMENT_COUNTS = _seg_counts_raw
SEGMENT_ORDER  = list(_cfg['SEGMENTS'].keys())  # CONFIG定義順を維持

# ノイズ除去範囲 (CONFIG に PLAUSIBLE_BOUNDS があれば優先)
PLAUSIBLE_BOUNDS = _cfg.get(
    'PLAUSIBLE_BOUNDS',
    {'x': (-400, 400), 'y': (0, 1200), 'z': (-1000, 1000)}
)

print(f"[{TASK_KEY}] 設定を CONFIG から読み込みました。")
print(f"  期待マーカー数: {EXPECTED_MARKER_COUNT}")
print(f"  セグメント: {SEGMENT_COUNTS}")
print(f"  入力ファイル: {RAW_OPTI_CSV_PATH}")


class PreAnalysisChecker:
    """
    OptiTrack生データを解析し、安定区間・安定IDを見つけ、
    インタラクティブ3Dプロット、マーカー存在CSV、
    CONFIG.py 用テンプレートテキストを出力する。
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.task_key  = TASK_KEY
        self.df_long   = self._load_opti_data_to_long()
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
                    self._generate_CONFIG_text(self.mean_pos)
                    self._create_interactive_3d_plot()
                else:
                    print("エラー: 平均座標を計算できませんでした。")
            else:
                print("エラー: 安定区間からデータを抽出できませんでした。")
        else:
            print("安定区間が見つからなかったため、CONFIGテンプレート生成と3Dプロットはスキップします。")

        self._create_wide_csv()

    # -------------------------------------------------------------------------
    def _load_opti_data_to_long(self):
        if not os.path.exists(self.file_path):
            print(f"エラー: ファイルなし: {self.file_path}"); return None
        print(f"'{os.path.basename(self.file_path)}' を読み込み中...")
        rows = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for _ in range(43): next(f)
                for line_num, line in enumerate(f, 44):
                    parts = line.strip().split(',')
                    try:
                        if len(parts) < 5: continue
                        frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4])
                        base_col = 5
                        if len(parts) >= base_col + n_markers * 4:
                            for i in range(n_markers):
                                x   = float(parts[base_col + 4*i])     * 1000.0
                                y   = float(parts[base_col + 4*i + 1]) * 1000.0
                                z   = float(parts[base_col + 4*i + 2]) * 1000.0
                                mid = int(parts[base_col + 4*i + 3])
                                rows.append((frame, t, mid, x, y, z))
                    except (ValueError, IndexError): continue
            if not rows: print("エラー: 有効行なし。"); return None
            out_df = pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])

            initial_count = len(out_df)
            out_df = out_df[
                (out_df['x'] >= PLAUSIBLE_BOUNDS['x'][0]) & (out_df['x'] <= PLAUSIBLE_BOUNDS['x'][1]) &
                (out_df['y'] >= PLAUSIBLE_BOUNDS['y'][0]) & (out_df['y'] <= PLAUSIBLE_BOUNDS['y'][1]) &
                (out_df['z'] >= PLAUSIBLE_BOUNDS['z'][0]) & (out_df['z'] <= PLAUSIBLE_BOUNDS['z'][1])
            ].copy()
            print(f"読み込み完了: {initial_count} 行 (範囲外ノイズ {initial_count - len(out_df)} 行を除外 -> 残り {len(out_df)} 行)")
            return out_df
        except Exception as e: print(f"読み込みエラー: {e}"); return None

    # -------------------------------------------------------------------------
    def _find_stable_window(self):
        print("\n--- 安定区間の探索 ---")
        static_candidate_df = self.df_long[self.df_long['Time'] <= STATIC_CANDIDATE_WINDOW_END].copy()
        if static_candidate_df.empty: print("エラー: 指定時間内にデータなし。"); return None, None

        ids_per_time  = static_candidate_df.groupby('Time')['id'].apply(set)
        sets_of_target = ids_per_time[ids_per_time.apply(len) == EXPECTED_MARKER_COUNT]
        if sets_of_target.empty:
            print(f"警告: {EXPECTED_MARKER_COUNT}個同時取得の時間なし。"); return None, None
        try:
            most_common_set = sets_of_target.apply(frozenset).value_counts().idxmax()
        except ValueError:
            print(f"警告: {EXPECTED_MARKER_COUNT}個セットが見つかりませんでした。"); return None, None

        stable_times = sets_of_target[sets_of_target.apply(frozenset) == most_common_set].index
        if len(stable_times) < 10: print("警告: 安定区間短すぎ。"); return None, None

        start_time, end_time = stable_times.min(), stable_times.max()
        stable_ids = sorted(list(most_common_set))
        print("\n【結果】")
        print(f"✅ 安定区間検出: {start_time:.3f}s - {end_time:.3f}s")
        print(f"   安定マーカーID ({len(stable_ids)}個): {stable_ids}")
        return (start_time, end_time), stable_ids

    # -------------------------------------------------------------------------
    def _segment_markers_by_y(self, mean_pos_df):
        print("\n--- Y座標による自動セグメント分け ---")
        expected = sum(SEGMENT_COUNTS.values())
        if expected != EXPECTED_MARKER_COUNT:
            print(f"エラー: SEGMENT_COUNTS 合計({expected})≠{EXPECTED_MARKER_COUNT}。"); return {}, {}

        if set(mean_pos_df.index) != set(self.stable_ids):
            print("警告: 安定IDと平均座標ID不一致。")
            ids_to_sort = list(set(mean_pos_df.index).intersection(self.stable_ids))
            if len(ids_to_sort) < EXPECTED_MARKER_COUNT:
                print(f"エラー: セグメント分けID不足 ({len(ids_to_sort)}/{EXPECTED_MARKER_COUNT})。"); return {}, {}
            sorted_ids = mean_pos_df.loc[ids_to_sort].sort_values('y', ascending=False).index.tolist()
        else:
            sorted_ids = mean_pos_df.sort_values('y', ascending=False).index.tolist()

        segments = {}; id_to_segment = {}; current_index = 0
        for name in SEGMENT_ORDER:
            count = SEGMENT_COUNTS[name]
            if count == 0:
                # 空セグメント(Thigh等)はスキップ
                segments[name] = []
                print(f"  {name} (0個): [] (スキップ)")
                continue
            if current_index + count > len(sorted_ids):
                print(f"エラー: セグメント '{name}' 割り当てID不足。")
                segment_ids = sorted_ids[current_index:]; count = len(segment_ids)
            else:
                segment_ids = sorted_ids[current_index : current_index + count]
            segments[name] = segment_ids
            for mid in segment_ids: id_to_segment[mid] = name
            print(f"  {name} ({count}個): {segment_ids}")
            current_index += count
            if current_index >= len(sorted_ids): break

        return segments, id_to_segment

    # -------------------------------------------------------------------------
    def _generate_CONFIG_text(self, mean_pos_df):
        """ CONFIG.py に貼り付けるためのテキストを生成・出力する """
        print("\n" + "="*60)
        print(f" CONFIG.py 用テンプレート [{self.task_key}]")
        print("="*60 + "\n")
        print("# --- ▼▼▼ 以下を CONFIG.py の taskX 設定に貼り付け ▼▼▼ ---")
        print("# (パスや時刻は手動で確認・修正してください)\n")

        # --- SEGMENTS ---
        print("'SEGMENTS': {")
        for name in SEGMENT_ORDER:
            print(f"    '{name}': {self.segments.get(name, [])},")
        print("},")

        # --- KEYFRAME_MAP ---
        print("\n'KEYFRAME_MAP': { # ★ 要手動確認/修正 ★")
        for seg_name in SEGMENT_ORDER:
            segment_ids = self.segments.get(seg_name, [])
            if segment_ids:
                print(f"    # --- {seg_name} ---")
                for mid in segment_ids:
                    if mid in self.stable_ids:
                        print(f"    {mid}: {mid},")
        print("},")

        # --- LINES_TO_DRAW ---
        print("\n'LINES_TO_DRAW': { # ★ 要手動確認/修正 (ルールベース自動割り当て) ★")
        lines_draw_items = []
        ids = {}

        # セグメントごとの座標を準備
        segment_coords = {}
        for seg_name, seg_ids in self.segments.items():
            if seg_ids:
                segment_coords[seg_name] = mean_pos_df.loc[seg_ids]

        if EXPECTED_MARKER_COUNT <= 5:
            # ===== task03 (5点: Hip/Knee/Ankle/Heel/Toe 直接配置) =====
            # LINES_TO_DRAW は Futto なしのため空
            lines_draw_items.append("    # Futto非着用のためゴム接続線なし")
        else:
            # ===== task01(15点) / task02(8点以上) =====
            try:
                # Hip (z小2点)
                hip_z_sorted = segment_coords['Hip'].sort_values('z')
                hip_z_low2   = hip_z_sorted.iloc[:2]
                ids['hip_z_low_x_high'] = hip_z_low2.sort_values('x', ascending=False).index[0]
                ids['hip_z_low_x_low']  = hip_z_low2.sort_values('x', ascending=True).index[0]

                # Hip (z大2点) — task01(4点Hip)のみ有効
                if len(segment_coords['Hip']) >= 4:
                    hip_z_high2 = hip_z_sorted.iloc[2:]
                    ids['hip_z_high_x_high'] = hip_z_high2.sort_values('x', ascending=False).index[0]
                    ids['hip_z_high_x_low']  = hip_z_high2.sort_values('x', ascending=True).index[0]

                # Knee
                knee_coords = segment_coords['Knee']
                ids['knee_y_high'] = knee_coords.sort_values('y', ascending=False).index[0]
                ids['knee_y_low']  = knee_coords.sort_values('y', ascending=True).index[0]
                ids['knee_x_low']  = knee_coords.sort_values('x', ascending=True).index[0]
                ids['knee_x_high'] = knee_coords.sort_values('x', ascending=False).index[0]

                if 'Thigh' in segment_coords and segment_coords['Thigh'] is not None:
                    ids['thigh'] = segment_coords['Thigh'].index[0]
                if 'Shank' in segment_coords and segment_coords['Shank'] is not None:
                    ids['shank'] = segment_coords['Shank'].index[0]

                # Foot
                foot_coords = segment_coords['Foot']
                ids['foot_y_high']    = foot_coords.sort_values('y', ascending=False).index[0]
                foot_x_low2  = foot_coords.sort_values('x').iloc[:2]
                ids['foot_x_low_z_low']  = foot_x_low2.sort_values('z').index[0]
                ids['foot_x_low_z_high'] = foot_x_low2.sort_values('z', ascending=False).index[0]
                foot_x_high2 = foot_coords.sort_values('x', ascending=False).iloc[:2]
                ids['foot_x_high_z_low']  = foot_x_high2.sort_values('z').index[0]
                ids['foot_x_high_z_high'] = foot_x_high2.sort_values('z', ascending=False).index[0]

                lines_draw_items.append(f'    "Front_Upper_In": ({ids["hip_z_low_x_high"]}, {ids["knee_y_high"]}),')
                lines_draw_items.append(f'    "Front_Upper_Out": ({ids["hip_z_low_x_low"]}, {ids["knee_y_high"]}),')
                lines_draw_items.append(f'    "Front_Knee_Upper_Out": ({ids["knee_y_high"]}, {ids["knee_x_low"]}),')
                lines_draw_items.append(f'    "Front_Knee_Upper_In": ({ids["knee_y_high"]}, {ids["knee_x_high"]}),')
                lines_draw_items.append(f'    "Front_Knee_Lower_Out": ({ids["knee_y_low"]}, {ids["knee_x_low"]}),')
                lines_draw_items.append(f'    "Front_Knee_Lower_In": ({ids["knee_y_low"]}, {ids["knee_x_high"]}),')
                lines_draw_items.append(f'    "Front_Shin": ({ids["knee_y_low"]}, {ids["foot_y_high"]}),')
                lines_draw_items.append(f'    "Toe_Out": ({ids["foot_y_high"]}, {ids["foot_x_low_z_low"]}),')
                lines_draw_items.append(f'    "Toe_In": ({ids["foot_y_high"]}, {ids["foot_x_high_z_low"]}),')
                if 'thigh' in ids:
                    lines_draw_items.append(f'    "Back_Upper_In": ({ids["hip_z_high_x_high"]}, {ids["thigh"]}),')
                    lines_draw_items.append(f'    "Back_Upper_Out": ({ids["hip_z_high_x_low"]}, {ids["thigh"]}),')
                    lines_draw_items.append(f'    "Back_Thigh_Out": ({ids["thigh"]}, {ids["knee_x_low"]}),')
                    lines_draw_items.append(f'    "Back_Thigh_In": ({ids["thigh"]}, {ids["knee_x_high"]}),')
                if 'shank' in ids:
                    lines_draw_items.append(f'    "Back_Knee_Out": ({ids["knee_x_low"]}, {ids["shank"]}),')
                    lines_draw_items.append(f'    "Back_Knee_In": ({ids["knee_x_high"]}, {ids["shank"]}),')
                    lines_draw_items.append(f'    "Back_Shin_Out": ({ids["shank"]}, {ids["foot_x_low_z_high"]}),')
                    lines_draw_items.append(f'    "Back_Shin_In": ({ids["shank"]}, {ids["foot_x_high_z_high"]}),')

            except (KeyError, IndexError) as e:
                print(f"エラー: LINES_TO_DRAW 自動割り当て中にエラー: {e}")
                lines_draw_items.append("    # --- 自動割り当て失敗 (手動で設定してください) ---")

        print('\n'.join(lines_draw_items))
        print("},")

        # --- MUSCLE_INDICATORS ---
        print("\n'MUSCLE_INDICATORS': { # ★ 要手動確認/修正 ★")
        if EXPECTED_MARKER_COUNT <= 5:
            # task03 用: 関節直接配置での近似定義
            hip_id   = self.segments.get('Hip',   [None])[0]
            knee_id  = self.segments.get('Knee',  [None])[0]
            foot_ids = self.segments.get('Foot',  [])
            shank_id = self.segments.get('Shank', [None])[0]
            heel_id  = foot_ids[0] if len(foot_ids) > 0 else None
            toe_id   = foot_ids[1] if len(foot_ids) > 1 else None

            print(f'    "Tibialis Anterior":  {{\'emg_col\': "L_TA_mean",  \'type\': \'midpoint\',          \'markers\': [{knee_id}, {shank_id}]}},')
            print(f'    "Soleus":             {{\'emg_col\': "L_SOL_mean", \'type\': \'centroid\',          \'markers\': [{knee_id}, {heel_id}, {toe_id}]}},')
            print(f'    "Rectus Femoris":     {{\'emg_col\': "L_RF_mean",  \'type\': \'midpoint\',          \'markers\': [{hip_id}, {knee_id}]}},')
            print(f'    "Vastus Lateralis":   {{\'emg_col\': "L_VL_mean",  \'type\': \'weighted_midpoint\', \'markers\': [{hip_id}, {knee_id}], \'weight\': 0.25}},')
            print(f'    "Biceps Femoris":     {{\'emg_col\': "L_BF_mean",  \'type\': \'midpoint\',          \'markers\': [{hip_id}, {knee_id}]}},')
            print(f'    "Semitendinosus":     {{\'emg_col\': "L_ST_mean",  \'type\': \'midpoint\',          \'markers\': [{hip_id}, {knee_id}]}},')
            print(f'    "Gluteus Maximus Area": {{\'emg_col\': "L_GM_mean", \'type\': \'single\', \'markers\': [{hip_id}]}},')
            print(f'    "Iliopsoas Area":     {{\'emg_col\': "L_ILIO_mean",\'type\': \'single\',            \'markers\': [{hip_id}]}},')
        else:
            # task01/02 用: 既存のルールベース定義
            muscle_definitions_template = {
                "Tibialis Anterior":    {'emg': "L_TA_mean",   'type': 'midpoint',          'ids_key': ['knee_y_low', 'foot_y_high']},
                "Soleus":               {'emg': "L_SOL_mean",  'type': 'centroid',          'ids_key': ['shank', 'foot_x_high_z_high', 'foot_x_low_z_high']},
                "Rectus Femoris":       {'emg': "L_RF_mean",   'type': 'centroid',          'ids_key': ['hip_z_low_x_high', 'knee_y_high', 'hip_z_low_x_low']},
                "Vastus Lateralis":     {'emg': "L_VL_mean",   'type': 'weighted_midpoint', 'ids_key': ['hip_z_low_x_low', 'knee_y_high'], 'weight': 0.25},
                "Biceps Femoris":       {'emg': "L_BF_mean",   'type': 'midpoint',          'ids_key': ['thigh', 'knee_x_low']},
                "Semitendinosus":       {'emg': "L_ST_mean",   'type': 'single',            'ids_key': ['thigh']},
                "Gluteus Maximus Area": {'emg': "L_GM_mean",   'type': 'offset',            'ids_key': ['hip_z_high_x_high', 'thigh', 'hip_z_high_x_low'], 'ref_ids_key': ['hip_z_high_x_high', 'thigh'], 'weight': 0.3},
                "Iliopsoas Area":       {'emg': "L_ILIO_mean", 'type': 'midpoint',          'ids_key': ['hip_z_low_x_high', 'hip_z_low_x_low']},
            }
            for name, definition in muscle_definitions_template.items():
                marker_ids_val = [ids.get(key, "???") for key in definition['ids_key']]
                item_str = f'    "{name}": {{\'emg_col\': "{definition["emg"]}", \'type\': \'{definition["type"]}\', \'markers\': {marker_ids_val}'
                if 'weight' in definition:       item_str += f', \'weight\': {definition["weight"]}'
                if 'ref_ids_key' in definition:
                    ref_ids_val = [ids.get(key, "???") for key in definition['ref_ids_key']]
                    item_str += f', \'ref_marker\': {ref_ids_val}'
                item_str += '},'
                print(item_str)
        print("}")

        print("\n# --- ▲▲▲ ここまでをコピー ▲▲▲ ---")
        print("\n" + "="*60 + "\n")

    # -------------------------------------------------------------------------
    def _create_interactive_3d_plot(self):
        print("\n--- 3Dマップの生成 ---")
        fig = plt.figure(figsize=(14, 10)); ax = fig.add_subplot(111, projection='3d')
        if not hasattr(self, 'mean_pos') or self.mean_pos.empty: print("エラー: 平均座標なし。プロット不可。"); return
        mean_pos = self.mean_pos
        unique_ids = sorted(mean_pos.index)
        cmap = plt.cm.get_cmap('tab20' if len(unique_ids) > 10 else 'tab10', len(unique_ids))
        id_colors = {uid: cmap(i) for i, uid in enumerate(unique_ids)}
        color_array = [id_colors[uid] for uid in mean_pos.index]
        ax.scatter(mean_pos['x'], mean_pos['y'], mean_pos['z'], c=color_array, s=50, picker=5)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'ID {mid}',
                           markerfacecolor=id_colors[mid], markersize=8) for mid in unique_ids]
        ax.legend(handles=legend_elements, title="Marker IDs", bbox_to_anchor=(1.02, 1), loc='upper left')
        annotation = ax.text(0, 0, 0, "", bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5), visible=False)

        def on_pick(event):
            inds = event.ind
            if not inds: return True
            ind = inds[0]; xs, ys, zs = event.artist._offsets3d
            x, y, z = xs[ind], ys[ind], zs[ind]
            marker_id = mean_pos.index[ind]
            annotation.set_position((x, y, z)); annotation.set_text(f"ID: {marker_id}"); annotation.set_visible(True)
            fig.canvas.draw_idle(); return True

        fig.canvas.mpl_connect('pick_event', on_pick)
        ax.set_title(f'Interactive 3D Marker Map [{self.task_key}] ({os.path.basename(self.file_path)})')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        all_coords = mean_pos[['x', 'y', 'z']].values
        x_c, y_c, z_c = all_coords[:,0], all_coords[:,1], all_coords[:,2]
        max_range = np.array([x_c.max()-x_c.min(), y_c.max()-y_c.min(), z_c.max()-z_c.min()]).max()
        mid_x, mid_y, mid_z = (x_c.max()+x_c.min())/2, (y_c.max()+y_c.min())/2, (z_c.max()+z_c.min())/2
        ax.set_xlim(mid_x-max_range/2, mid_x+max_range/2)
        ax.set_ylim(mid_y-max_range/2, mid_y+max_range/2)
        ax.set_zlim(mid_z-max_range/2, mid_z+max_range/2)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    # -------------------------------------------------------------------------
    def _create_wide_csv(self):
        if self.df_long is None or self.df_long.empty: print("データなし、Wide形式CSVスキップ。"); return
        try:
            pivot_df = self.df_long.pivot_table(index='Time', columns='id', aggfunc='size', fill_value=0)
            pivot_df = (pivot_df > 0).astype(int)
            base_name   = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(os.path.dirname(self.file_path), f"{base_name}_marker_presence.csv")
            pivot_df.to_csv(output_path)
            print(f"\nWide形式マーカー存在CSV出力: {output_path}")
        except Exception as e: print(f"Wide形式CSV出力エラー: {e}")


# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(RAW_OPTI_CSV_PATH):
        print(f"エラー: 指定ファイルパスなし: {RAW_OPTI_CSV_PATH}")
    else:
        try:
            checker = PreAnalysisChecker(RAW_OPTI_CSV_PATH)
            if hasattr(checker, 'df_long') and checker.df_long is not None:
                print("\nPre-analysis check completed.")
            else:
                print("\nPre-analysis check failed or interrupted.")
        except Exception as e:
            import traceback
            print(f"\nスクリプト実行中に予期せぬエラー: {e}")
            traceback.print_exc()