# =============================================================================
# joint_analysis/main.py
#
# 役割:
#   対話式メニューでタスク・フェーズ・被験者を選び、
#   仮想関節座標（Hip / Knee / Ankle）を歩行周期平均データとして算出・保存し、
#   3Dアニメーションで確認できる。
#
# メニュー:
#   c: 仮想関節を算出してCSVに保存
#   v: 算出済み仮想関節 + マーカーを3Dアニメーションで確認
#   b: 算出 → アニメーション確認を続けて実行
#   q: 終了
#
# 実行方法:
#   cd C:\FuttoAnalysis\2026_analysis\joint_analysis
#   python main.py
# =============================================================================

import sys
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'joint_analysis'

from futto_common import CONFIG as config
from futto_common.io_utils import load_opti_csv, build_paths
from .joint_calc import calc_joint_centers, interpolate_missing_frames, save_joint_csv
from .visualizer import animate_joint_check


# ---------------------------------------------------------------------------
# 共通: タスク・フェーズ・被験者選択 + データ読み込み
# ---------------------------------------------------------------------------

def _select_and_load():
    """
    タスク・フェーズ・被験者を対話選択し、
    OptiTrackデータと設定を返す。
    失敗時は None を返す。
    """
    # --- タスク選択 ---
    available = [t for t in config.TASKS if t in config.TASK_CONFIGS]
    print(f"\n利用可能なタスク: {', '.join(available)}")
    task = input("タスク名を入力してください: ").strip().lower()
    if task not in config.TASK_CONFIGS:
        print(f"エラー: '{task}' は CONFIG に登録されていません。")
        return None
    cfg = config.TASK_CONFIGS[task]

    joint_defs = cfg.get('JOINT_CENTER_DEFS')
    if not joint_defs:
        print(f"エラー: CONFIG の {task} に 'JOINT_CENTER_DEFS' が定義されていません。")
        return None
    print(f"  -> 定義済み関節: {list(joint_defs.keys())}")

    # --- フェーズ選択 ---
    phases = getattr(config, 'PHASES', {})
    print("\n利用可能なフェーズ:")
    for k, v in phases.items():
        print(f"  {k}: {v['name']}  ({v['start']}s〜{v['end']}s)")
    phase_input = input("フェーズ番号を入力してください (1-5): ").strip()
    if not phase_input.isdigit() or int(phase_input) not in phases:
        print("無効なフェーズ番号です。")
        return None
    phase = int(phase_input)
    speed = phases[phase]['name'].replace('m/s', '')

    # --- 被験者選択 ---
    subjects = getattr(config, 'SUBJECTS', ['Ide'])
    if len(subjects) == 1:
        subject = subjects[0]
        print(f"\n被験者: {subject}")
    else:
        print("\n被験者一覧:")
        for i, s in enumerate(subjects):
            print(f"  {i+1}: {s}")
        try:
            idx     = int(input(f"番号を選択してください (1-{len(subjects)}): ")) - 1
            subject = subjects[idx]
        except (ValueError, IndexError):
            subject = subjects[0]
            print(f"  -> デフォルトの被験者 '{subject}' を使用します。")

    # --- パス生成 ---
    paths = build_paths(
        cfg        = cfg,
        task_key   = task,
        phase      = phase,
        speed      = speed,
        subject    = subject,
        base_dir   = getattr(config, 'BASE_DIR',   r"C:\Users\ihika\2026_experiment"),
        result_dir = getattr(config, 'RESULT_DIR', r"C:\FuttoAnalysis\result"),
    )
    joint_output = (
        Path(getattr(config, 'RESULT_DIR', r"C:\FuttoAnalysis\result"))
        / "2026" / task / speed
        / f"{task}_Phase{phase}_{speed}ms_joint_centers.csv"
    )

    # --- OptiTrackデータ読み込み ---
    print(f"\nOptiTrackデータを読み込みます...")
    df_mean = load_opti_csv(paths['opti_csv'])
    if df_mean is None:
        return None

    required_cols = {'gait_cycle_%', 'id', 'x', 'y', 'z'}
    missing_cols  = required_cols - set(df_mean.columns)
    if missing_cols:
        print(f"エラー: 入力CSVに必要な列がありません: {missing_cols}")
        return None
    print(f"  -> {df_mean['gait_cycle_%'].nunique()} フレーム読み込み完了")

    return {
        'task':         task,
        'cfg':          cfg,
        'joint_defs':   joint_defs,
        'phase':        phase,
        'speed':        speed,
        'df_mean':      df_mean,
        'joint_output': joint_output,
    }


# ---------------------------------------------------------------------------
# 処理A: 仮想関節算出 → CSV保存
# ---------------------------------------------------------------------------

def _calc_and_save(ctx: dict):
    """仮想関節を算出してCSVに保存し、df_joints を返す。"""
    print(f"\n仮想関節座標を算出中...")
    df_joints = calc_joint_centers(ctx['df_mean'], ctx['joint_defs'])
    df_joints = interpolate_missing_frames(df_joints)

    # 結果サマリー
    print("\n--- 算出結果サマリー (先頭5行) ---")
    print(df_joints.head().to_string(index=False))
    print(f"\n  総フレーム数: {len(df_joints)}")
    for joint in ctx['joint_defs'].keys():
        cols = [f'{joint}_x', f'{joint}_y', f'{joint}_z']
        nan_count = df_joints[cols].isna().sum().sum()
        if nan_count == 0:
            print(
                f"  {joint:8s}: "
                f"X変動={df_joints[f'{joint}_x'].max()-df_joints[f'{joint}_x'].min():.1f}mm  "
                f"Y変動={df_joints[f'{joint}_y'].max()-df_joints[f'{joint}_y'].min():.1f}mm  "
                f"Z変動={df_joints[f'{joint}_z'].max()-df_joints[f'{joint}_z'].min():.1f}mm"
            )
        else:
            print(f"  {joint:8s}: NaN={nan_count} フレーム残存")

    save_joint_csv(df_joints, ctx['joint_output'])
    print(f"\n完了。出力先: {ctx['joint_output']}")
    return df_joints


# ---------------------------------------------------------------------------
# 処理B: 3Dアニメーション確認
# ---------------------------------------------------------------------------

def _show_animation(ctx: dict, df_joints):
    """仮想関節 + マーカーを3Dアニメーションで表示する。"""
    print("\n3Dアニメーションを表示します...")
    animate_joint_check(
        df_mean_cycle = ctx['df_mean'],
        df_joints     = df_joints,
        cfg           = ctx['cfg'],
        task_key      = ctx['task'],
        phase         = ctx['phase'],
        speed         = ctx['speed'],
    )


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" 仮想関節座標算出 + 確認プログラム")
    print("=" * 60)

    while True:
        print("\n--- メニュー ---")
        print("  c: 仮想関節を算出してCSVに保存")
        print("  v: 算出済みCSVを読み込んで3Dアニメーション確認")
        print("  b: 算出 → そのままアニメーション確認 (c + v を連続実行)")
        print("  q: 終了")
        mode = input("選択 >> ").strip().lower()

        if mode == 'q':
            print("終了します。")
            break

        elif mode in ('c', 'b'):
            ctx = _select_and_load()
            if ctx is None:
                continue
            df_joints = _calc_and_save(ctx)
            if mode == 'b':
                _show_animation(ctx, df_joints)

        elif mode == 'v':
            ctx = _select_and_load()
            if ctx is None:
                continue

            import pandas as pd
            joint_path = ctx['joint_output']
            if not joint_path.exists():
                print(f"\n警告: 保存済みCSVが見つかりません: {joint_path}")
                ans = input("今すぐ算出しますか？ (y/n): ").strip().lower()
                if ans == 'y':
                    df_joints = _calc_and_save(ctx)
                else:
                    continue
            else:
                df_joints = pd.read_csv(joint_path)
                print(f"  -> CSVを読み込みました: {joint_path}")

            _show_animation(ctx, df_joints)

        else:
            print("無効な入力です。")


if __name__ == '__main__':
    main()