# =============================================================================
# cot_analysis/run_cot_analysis.py
#
# CPET生データから全被験者・全日程・全タスクのCOTを算出し、
# COT-Fr重ね描きグラフと速度別箱ひげ図を result/ に出力する。
#
# 使い方:
#   python run_cot_analysis.py
#
# 入力:
#   C:\FuttoAnalysis\CPET\{subject}\{day}\{task}.txt
#   (CONFIG.SUBJECTS_MULTI x CONFIG.DAYS x CONFIG.TASKS の全組み合わせを走査。
#    存在しないファイルはスキップする)
#
# 出力:
#   C:\FuttoAnalysis\result\2026\cot\cot_all_subjects.csv
#   C:\FuttoAnalysis\result\2026\cot\cot_froude_overlay.png
#   C:\FuttoAnalysis\result\2026\cot\cot_boxplot_by_speed.png
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "futto_common"))

import pandas as pd
import CONFIG

from cot_analysis.cot_loader import load_cpet_txt
from cot_analysis.cot_calc import (
    compute_stage_vo2_means, compute_rest_vo2_mean, compute_cot, compute_froude,
)
from cot_analysis.cot_plot import plot_cot_froude_overlay, plot_cot_boxplot_by_speed


def run():
    speeds = CONFIG.COT_SPEED_STAGES
    n_stages = len(speeds)
    rows = []

    for subject in CONFIG.SUBJECTS_MULTI:
        profile = CONFIG.SUBJECT_PROFILES.get(subject, {})
        weight_kg = profile.get('weight_kg')
        leg_length_cm = profile.get('leg_length_cm')
        if weight_kg is None or leg_length_cm is None:
            print(f"[スキップ] {subject}: CONFIG.SUBJECT_PROFILES に体重/脚長が未設定です。")
            continue
        leg_length_m = leg_length_cm / 100.0

        for day in CONFIG.DAYS:
            for task in CONFIG.TASKS:
                cpet_path = CONFIG.get_raw_path('cpet', subject, day, task)
                if not cpet_path.exists():
                    print(f"[スキップ] {subject}/{day}/{task}: ファイルが見つかりません ({cpet_path})")
                    continue

                print(f"処理中: {subject}/{day}/{task}")
                df, recorded_weight = load_cpet_txt(cpet_path)

                if recorded_weight is not None and abs(recorded_weight - weight_kg) > 2.0:
                    print(f"  [注意] CPETファイル記録体重({recorded_weight}kg)とCONFIG設定値"
                          f"({weight_kg}kg)が2kg以上異なります。要確認。")

                vo2_stage = compute_stage_vo2_means(
                    df, CONFIG.WALKING_START_SEC, CONFIG.STAGE_DURATION_SEC,
                    CONFIG.STAGE_TAIL_WINDOW_SEC, n_stages,
                )
                rest_vo2 = compute_rest_vo2_mean(df, CONFIG.REST_WINDOW_SEC)
                if rest_vo2 == rest_vo2:  # not NaN
                    print(f"  安静時VO2平均 (QC参考値): {rest_vo2:.1f} ml/min")

                cot = compute_cot(vo2_stage, weight_kg, speeds, g=CONFIG.COT_G)
                fr = compute_froude(speeds, leg_length_m, g=CONFIG.COT_G)

                for i, speed in enumerate(speeds):
                    rows.append({
                        'subject': subject, 'day': day, 'task': task,
                        'speed': speed, 'froude': fr[i],
                        'vo2_ml_min': vo2_stage[i], 'cot': cot[i],
                    })

    if not rows:
        print("\nエラー: 処理できたデータがありません。"
              "CPET生データの配置と CONFIG.SUBJECT_PROFILES の体重/脚長設定を確認してください。")
        return

    df_all = pd.DataFrame(rows)

    out_dir = Path(CONFIG.RESULT_DIR) / "2026" / "cot"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cot_all_subjects.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nCOTデータを保存しました: {csv_path}")

    plot_cot_froude_overlay(df_all, save_path=out_dir / "cot_froude_overlay.png")
    plot_cot_boxplot_by_speed(df_all, save_path=out_dir / "cot_boxplot_by_speed.png")

    print("\n完了。")


if __name__ == "__main__":
    run()
