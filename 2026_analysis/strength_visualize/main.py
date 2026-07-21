# =============================================================================
# strength_visualize/main.py
#
# 役割:
#   プログラムのエントリーポイント。
#   - ユーザーへの入力プロンプト
#   - 各モジュールの呼び出し順序の制御
#   - 対話メニューの提供
#
# 実行方法:
#   (A) パッケージフォルダ内から直接実行
#       cd C:\FuttoAnalysis\2026_analysis\strength_visualize
#       python main.py
#
#   (B) 親フォルダから -m オプションで実行
#       cd C:\FuttoAnalysis\2026_analysis
#       python -m strength_visualize.main
# =============================================================================

import os
import sys
 
# ------------------------------------------------------------------
# (A) の直接実行に対応するため、パッケージルートをパスに追加する。
# (B) の -m 実行では不要だが、あっても害はない。
# ------------------------------------------------------------------
if __name__ == '__main__' and __package__ is None:
    from pathlib import Path
    # このファイルの親ディレクトリ (= 2026_analysis) を sys.path に追加
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'strength_visualize'
 
from pathlib import Path
 
from futto_common import CONFIG as config
from futto_common.io_utils import (
    load_opti_csv,
    load_rubber_properties,
    load_emg_csv,
    save_tension_csv,
    build_paths,
)
from .tension_calc import calculate_all_tensions
from .visualizer import TensionVisualizer
 


# ===========================================================================
# 対話メニュー
# ===========================================================================

def run_interactive_menu(visualizer, df_mean_cycle, tension_data,
                          cfg, task_key, speed_str, year="2026"):
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
            options['g'] = "張力グラフを表示し、自動保存"
            options['m'] = "静止3Dマップを表示し、自動保存 (10%刻み)"
        options['q'] = "終了"

        for key, desc in options.items():
            print(f"  {key}: {desc}")

        action = input(f"実行する操作 [{'/'.join(options.keys())}]: ").lower()

        if action == 'a' and 'a' in options:
            print("\nアニメーションを表示します...")
            visualizer.run_animation(
                df_mean_cycle,
                tension_data if tension_data else {},
                cfg.get('LINES_TO_DRAW', {}),
                show=True, save_path=None
            )

        elif action == 's' and 's' in options:
            video_filename = f"{task_key}_{speed_str}_video.gif"
            save_path = os.path.join(
                getattr(config, 'RESULT_DIR', '.'),
                year, task_key, speed_str, video_filename
            )
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"\nアニメーションを保存します: {save_path}")
                visualizer.run_animation(
                    df_mean_cycle,
                    tension_data if tension_data else {},
                    cfg.get('LINES_TO_DRAW', {}),
                    show=False, save_path=save_path
                )
            except Exception as e:
                print(f"ファイル保存エラー: {e}")

        elif action == 'g' and 'g' in options:
            seg_groups = getattr(
                config, 'SEGMENT_GROUPS',
                {'All Lines': list(cfg.get('LINES_TO_DRAW', {}).keys())}
            )
            # 保存用ディレクトリの自動生成 (2026/taskXX/0.7/graph)
            graph_save_dir = os.path.join(
                getattr(config, 'RESULT_DIR', '.'),
                year, task_key, speed_str, "graph"
            )
            os.makedirs(graph_save_dir, exist_ok=True)
            visualizer.plot_segment_tensions(
                tension_data, seg_groups,
                save_dir=graph_save_dir, task_name=task_key, speed=speed_str
            )

        elif action == 'm' and 'm' in options:
            # 保存用ディレクトリの自動生成 (2026/taskXX/0.7/gait_cycle)
            map_save_dir = os.path.join(
                getattr(config, 'RESULT_DIR', '.'),
                year, task_key, speed_str, "gait_cycle"
            )
            os.makedirs(map_save_dir, exist_ok=True)
            visualizer.show_static_3d_maps(
                df_mean_cycle,
                tension_data if tension_data else {},
                cfg.get('LINES_TO_DRAW', {}),
                save_dir=map_save_dir, task_name=task_key, speed=speed_str
            )

        elif action == 'q':
            print("プログラムを終了します。")
            break

        else:
            print("無効な入力です。")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=========================================================")
    print(" 張力＆筋電可視化プログラム (対話メニュー搭載・5フェーズ対応)")
    print("=========================================================")

    # --- タスク選択 ---
    task = input("タスク名 (task01 または task02): ").strip().lower()
    if not hasattr(config, 'TASK_CONFIGS') or task not in config.TASK_CONFIGS:
        print(f"エラー: CONFIG.py に {task} の設定が見つかりません。")
        return
    cfg = config.TASK_CONFIGS[task]

    # --- フェーズ選択 ---
    phase_input = input("フェーズ番号 (1-5): ").strip()
    if not phase_input.isdigit() or \
            int(phase_input) not in getattr(config, 'PHASES', {1:1, 2:2, 3:3, 4:4, 5:5}):
        print("無効なフェーズ番号です。")
        return

    phase      = int(phase_input)
    phase_info = getattr(config, 'PHASES', {}).get(phase, {'name': f'{0.5+phase*0.2}m/s'})
    speed      = phase_info['name'].replace('m/s', '')

    # --- 被験者選択 ---
    print("--- 被験者を選択してください ---")
    subjects = getattr(config, 'SUBJECTS', ['Ide'])
    for i, name in enumerate(subjects):
        print(f"{i+1}: {name}")
    try:
        sub_num_input = int(input(f"番号 (1-{len(subjects)}): "))
        subject       = subjects[sub_num_input - 1]
    except (ValueError, IndexError):
        print("無効な入力のため、デフォルトの被験者を使用します。")
        subject = subjects[0]

    # --- パス生成 ---
    paths = build_paths(
        cfg       = cfg,
        task_key  = task,
        phase     = phase,
        speed     = speed,
        subject   = subject,
        base_dir  = getattr(config, 'BASE_DIR', r"C:\Users\ihika\2026_experiment"),
        result_dir= getattr(config, 'RESULT_DIR', r"C:\FuttoAnalysis\result"),
    )

    # --- CONFIG から設定値を取得 ---
    natural_lengths  = cfg.get('NATURAL_LENGTHS', {})
    if not natural_lengths:
        print(f"警告: CONFIG.py の {task} 内に NATURAL_LENGTHS が設定されていません。")

    force_multiplier = cfg.get('FORCE_MULTIPLIER', 1.0)
    lines_to_draw    = cfg.get('LINES_TO_DRAW', {})
    muscle_indicators= cfg.get('MUSCLE_INDICATORS', {})

    rubber_excel_path = getattr(config, 'RUBBER_PROPERTIES_EXCEL_PATH',
                                r"C:\FuttoAnalysis\rubber_strength.xlsx")
    sheet_name        = getattr(config, 'RUBBER_PROPERTIES_SHEET_NAME', 'Sheet1')

    # --- データ読み込み ---
    df_mean_cycle = load_opti_csv(paths['opti_csv'])
    if df_mean_cycle is None:
        return

    strain_to_force_interp = load_rubber_properties(rubber_excel_path, sheet_name)
    emg_data, max_emg_vals = load_emg_csv(str(paths['emg_csv']), muscle_indicators)

    # --- Visualizer 構築 ---
    visualizer = TensionVisualizer(
        strain_to_force_interp = strain_to_force_interp,
        emg_data               = emg_data,
        max_emg_vals           = max_emg_vals,
        muscle_indicators_def  = muscle_indicators,
    )

    # --- 張力計算 ---
    tension_data = None
    if strain_to_force_interp is not None:
        print(f"ゴムの張力計算を実行中... (Multiplier: x{force_multiplier})")
        tension_data, tension_df_for_csv = calculate_all_tensions(
            df_mean_cycle      = df_mean_cycle,
            natural_lengths    = natural_lengths,
            lines_to_draw_def  = lines_to_draw,
            strain_to_force_interp = strain_to_force_interp,
            force_multiplier   = force_multiplier,
        )
        if tension_data and not tension_df_for_csv.empty:
            save_tension_csv(tension_df_for_csv, paths['tension_out'])
    else:
        print("エラー: 張力計算モデルが初期化できませんでした。")

    # --- 対話メニュー ---
    run_interactive_menu(visualizer, df_mean_cycle, tension_data, cfg, task, speed)


if __name__ == '__main__':
    main()