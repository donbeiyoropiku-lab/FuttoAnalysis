# =========================================================================
# 
# 概要:
#   歩行周期60%時点でのマーカー割り当て状況と、捨てられた生データを3D可視化し、
#   トラッキングが崩れる原因（誤ったRaw_IDの取得）を目視で特定するためのデバッガ。
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import CONFIG

TARGET_PHASE = 0.59


def load_opti_data(file_path):
    print(f"生データ '{os.path.basename(file_path)}' を読み込み中...")
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(43): next(f) # ヘッダー
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5: continue
                frame, t, n_markers = int(parts[1]), float(parts[2]), int(parts[4]); base_col = 5
                if len(parts) >= base_col + n_markers * 4:
                    for i in range(n_markers):
                        x, y, z = float(parts[base_col + 4*i])*1000, float(parts[base_col + 4*i + 1])*1000, float(parts[base_col + 4*i + 2])*1000
                        mid = int(parts[base_col + 4*i + 3])
                        rows.append((frame, t, mid, x, y, z))
        return pd.DataFrame(rows, columns=["Frame", "Time", "id", "x", "y", "z"])
    except Exception as e: 
        print(f"読み込みエラー: {e}")
        return None

def main():
    print("=== 歩行周期% デバッグビューア ===")
    
    # CONFIGからの設定読み込み
    if 'task1' in CONFIG.TASK_CONFIGS:
        cfg = CONFIG.TASK_CONFIGS['task1']
    elif 'task01' in CONFIG.TASK_CONFIGS:
        cfg = CONFIG.TASK_CONFIGS['task01']
    else:
        print("エラー: CONFIG.py に 'task1' または 'task01' が見つかりません。")
        return

    opti_csv_path = cfg['OPTI_CSV_PATH']
    cycles_path = cfg['LABCHART_CYCLES_PATH']
    output_base = cfg['OUTPUT_CSV_PATH'].replace('.csv', '')
    corrected_path = output_base + '.csv'
    debug_path = output_base + '_debug_mapping.csv'
    
    # ファイル存在確認
    for path in [opti_csv_path, cycles_path, corrected_path, debug_path]:
        if not os.path.exists(path):
            print(f"エラー: ファイルが見つかりません -> {path}")
            print("先に opti_edit_auto_gait.py を実行してファイルを作成してください。")
            return

    df_raw = load_opti_data(opti_csv_path)
    df_cycles = pd.read_csv(cycles_path)
    df_corrected = pd.read_csv(corrected_path)
    df_debug = pd.read_csv(debug_path)
    
    lines_to_draw = cfg.get('LINES_TO_DRAW', {})
    plots = []
    
    # 指定された n (0~4) についてループ
    for n in range(5):
        t_base = 60 + 60 * n
        valid_cycles = df_cycles[df_cycles['hs_time'] >= t_base]
        
        if valid_cycles.empty:
            print(f"Time >= {t_base}s の歩行周期が見つかりません。")
            continue
            
        target_cycle = valid_cycles.iloc[0]
        t_start = target_cycle['hs_time']
        t_end = target_cycle['next_hs_time']
        
        # 指定された位相の時刻を計算
        t_target = t_start + TARGET_PHASE * (t_end - t_start)
        
        # 直近のFrameを取得
        frame_idx = df_corrected.iloc[(df_corrected['Time'] - t_target).abs().argmin()]['Frame']
        actual_time = df_corrected[df_corrected['Frame'] == frame_idx]['Time'].iloc[0]
        
        print(f"\n--- n={n} (Base {t_base}s) ---")
        print(f"歩行周期: {t_start:.3f}s - {t_end:.3f}s")
        print(f"{TARGET_PHASE * 100:.1f}% 時刻: {t_target:.3f}s -> 対象Frame: {int(frame_idx)} (Time: {actual_time:.3f}s)")
        
        frame_corr = df_corrected[df_corrected['Frame'] == frame_idx]
        frame_debug = df_debug[df_debug['Frame'] == frame_idx]
        frame_raw = df_raw[df_raw['Frame'] == frame_idx]
        
        tid_to_raw = dict(zip(frame_debug['Template_ID'], frame_debug['Raw_ID']))
        
        fig = go.Figure()
        
        # 1. 修正されたマーカー (TID)
        texts = []
        for _, row in frame_corr.iterrows():
            tid = int(row['id'])
            raw_id = tid_to_raw.get(tid, "Unknown")
            texts.append(f"TID: {tid}<br>(Raw: {raw_id})")
            
        fig.add_trace(go.Scatter3d(
            x=frame_corr['x'], y=frame_corr['y'], z=frame_corr['z'],
            mode='markers+text',
            marker=dict(size=8, color='green', symbol='circle'),
            text=texts,
            textposition='middle right',
            textfont=dict(color='darkgreen', size=12),
            name="Corrected (TID)"
        ))
        
        # 2. 使われなかった生データ (ノイズ・ゴースト候補)
        used_raw_ids = set()
        for raw_val in tid_to_raw.values():
            if str(raw_val).isdigit() or isinstance(raw_val, (int, float)):
                try: used_raw_ids.add(int(float(raw_val)))
                except: pass
        
        unmatched_raw = frame_raw[~frame_raw['id'].isin(used_raw_ids)]
        if not unmatched_raw.empty:
            fig.add_trace(go.Scatter3d(
                x=unmatched_raw['x'], y=unmatched_raw['y'], z=unmatched_raw['z'],
                mode='markers+text',
                marker=dict(size=4, color='gray', opacity=0.5),
                text=[f"Raw: {int(rid)}" for rid in unmatched_raw['id']],
                textposition='middle left',
                textfont=dict(color='gray', size=10),
                name="Unmatched Raw"
            ))
            
        # 3. 線分の描画
        corr_dict = {int(row['id']): (row['x'], row['y'], row['z']) for _, row in frame_corr.iterrows()}
        for line_name, (id1, id2) in lines_to_draw.items():
            if id1 in corr_dict and id2 in corr_dict:
                p1 = corr_dict[id1]; p2 = corr_dict[id2]
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                    mode='lines', line=dict(color='red', width=2),
                    name=line_name, showlegend=False
                ))
                
        fig.update_layout(
            title=f"n={n} (Base {t_base}s) / {TARGET_PHASE * 100:.1f}% Phase / Frame: {int(frame_idx)}",
            scene=dict(
                xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=0.5, z=0.5)) # 横から見やすい視点
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        plots.append(fig)
        
    print("\nブラウザで5つのプロットタブが開きます。")
    print("確認が終わったら、本来の15974や15972が『どのRaw ID』になっているかを教えてください！")
    for fig in plots:
        fig.show()

if __name__ == "__main__":
    main()