"""
emg_pipeline/run_preprocess.py
================================
EMG 前処理パイプラインの実行スクリプト

使い方:
  # 対話モード
  python emg_pipeline/run_preprocess.py

  # 1件指定
  python emg_pipeline/run_preprocess.py --subjects Ide --task task01 --phase 3

  # 全フェーズ
  python emg_pipeline/run_preprocess.py --subjects Ide --task task01 --all_phases

  # 全タスク × 全フェーズ（被験者1名）
  python emg_pipeline/run_preprocess.py --subjects Ide --all_tasks --all_phases

  # 全タスク × 全フェーズ × 複数被験者
  python emg_pipeline/run_preprocess.py --subjects Ide Tanaka Suzuki --all_tasks --all_phases

出力先（被験者名を含む）:
  C:\\FuttoAnalysis\\result\\2026\\{subject}\\{task}\\{speed}\\
    {task}_Phase{N}_{speed}ms_emg_normalized.csv  ← 正規化済み（後続解析用）
    {task}_Phase{N}_{speed}ms_emg_enveloped.csv   ← 包絡線（μV）
    plots/
      all_channels_Ph{N}_{speed}.png              ← 全チャンネル一覧
      steps_{ch}_Ph{N}_{speed}.png                ← 前処理ステップ確認図

サマリー:
  C:\\FuttoAnalysis\\result\\2026\\emg_preprocess_summary.csv  ← 全被験者×タスク×フェーズの処理結果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2] / "futto_common"))

try:
    import CONFIG as CFG
    RESULT_DIR = CFG.RESULT_DIR
    BASE_DIR   = str(CFG.BASE_DIR)
    FS_EMG     = CFG.FS_EMG
    PHASES     = CFG.PHASES
    TASKS      = CFG.TASKS
    SUBJECTS   = CFG.SUBJECTS
except ModuleNotFoundError:
    RESULT_DIR = r"C:\FuttoAnalysis\result"
    BASE_DIR   = r"C:\Users\ihika\2026_experiment"
    FS_EMG     = 2000
    PHASES     = {
        1: {'name': '0.7m/s', 'start': 40.0,  'end': 100.0},
        2: {'name': '0.9m/s', 'start': 100.0, 'end': 160.0},
        3: {'name': '1.1m/s', 'start': 160.0, 'end': 220.0},
        4: {'name': '1.3m/s', 'start': 220.0, 'end': 280.0},
        5: {'name': '1.5m/s', 'start': 280.0, 'end': 340.0},
    }
    TASKS    = ['task01', 'task02', 'task03']
    SUBJECTS = ['Ide']

from emg_pipeline.emg_loader import (
    load_cometa_txt, extract_phase, get_emg_array,
    build_emg_raw_path, PHASE_INTERVALS,
)
from emg_pipeline.emg_preprocessor import (
    EMGPreprocessor, plot_preprocessing_steps, plot_all_channels_normalized,
)


# =============================================================================
# ヘルパー
# =============================================================================

def phase_to_speed(phase: int) -> str:
    """フェーズ番号 → 速度文字列（例: 3 → '1.1'）"""
    if phase in PHASES:
        return PHASES[phase]['name'].replace('m/s', '')
    return str(phase)


def build_out_dir(result_dir: str, subject: str, task_key: str, speed: str) -> Path:
    """
    出力ディレクトリパスを生成する。
    被験者ごとにフォルダが分かれるため、複数被験者でも上書きされない。

    例: C:\\FuttoAnalysis\\result\\2026\\Ide\\task01\\1.1\\
    """
    return Path(result_dir) / "2026" / subject / task_key / speed


# =============================================================================
# 1件処理（デバッグ・単発実行用）
# =============================================================================

def run_single(
    task_key     : str,
    phase        : int,
    subject      : str  = "Ide",
    result_dir   : str  = RESULT_DIR,
    base_dir     : str  = BASE_DIR,
    save_plots   : bool = True,
    save_all_steps: bool = False,
) -> dict:
    """
    1被験者 × 1タスク × 1フェーズ を処理する。

    複数フェーズを処理したい場合は run_batch() を使うこと
    （run_batch は同じ .txt を1回しか読まないため効率的）。
    """
    speed = phase_to_speed(phase)
    print(f"\n{'='*55}")
    print(f"  被験者: {subject}  タスク: {task_key}  Phase{phase} ({speed}m/s)")
    print(f"{'='*55}")

    raw_path = build_emg_raw_path(task_key, subject, base_dir)
    print(f"  入力: {raw_path}")

    if not raw_path.exists():
        print(f"  [スキップ] ファイルが存在しません: {raw_path}")
        return {'status': 'skip', 'subject': subject, 'task': task_key, 'phase': phase}

    # ファイル読み込み
    df_all  = load_cometa_txt(raw_path)
    df_base = extract_phase(df_all, phase=0)    # 静止前（ベースライン）
    df_walk = extract_phase(df_all, phase=phase) # 歩行区間

    if len(df_walk) == 0:
        print(f"  [スキップ] データが空です。")
        return {'status': 'empty', 'subject': subject, 'task': task_key, 'phase': phase}

    # 出力先（被験者名を含む）
    out_dir  = build_out_dir(result_dir, subject, task_key, speed)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 前処理
    proc   = EMGPreprocessor(fs=FS_EMG)
    result = proc.run(df_walk, df_base, task_key=task_key, phase=phase, speed=speed)
    print(f"\n  {result.summary()}")

    # CSV 保存
    proc.save_csv(result, out_dir, step='normalized')
    proc.save_csv(result, out_dir, step='enveloped')
    if save_all_steps:
        proc.save_all_steps(result, out_dir)

    # プロット
    if save_plots:
        t0 = float(df_walk['Time_s'].iloc[0])
        plot_all_channels_normalized(
            result,
            save_path=plot_dir / f"all_channels_Ph{phase}_{speed}.png",
            t_range=(t0, t0 + 10.0),
        )
        for ch_name in ['L_GM', 'R_SOL']:
            if ch_name in result.channel_names:
                idx = result.channel_names.index(ch_name)
                plot_preprocessing_steps(
                    result, channel_idx=idx,
                    save_path=plot_dir / f"steps_{ch_name}_Ph{phase}_{speed}.png",
                    t_range=(t0, t0 + 3.0),
                )

    print(f"  出力先: {out_dir}")
    return {
        'status'      : 'ok',
        'subject'     : subject,
        'task'        : task_key,
        'phase'       : phase,
        'speed'       : speed,
        'n_channels'  : result.N_ch,
        'n_samples'   : result.T,
        'duration_s'  : result.T / result.fs,
        'peak_mean_uV': float(result.peak_values.mean()),
        'out_dir'     : str(out_dir),
    }


# =============================================================================
# バッチ処理（複数被験者 × 複数タスク × 複数フェーズ）
# =============================================================================

def run_batch(
    task_keys  : list[str],
    phases     : list[int],
    subjects   : list[str] = None,
    result_dir : str  = RESULT_DIR,
    base_dir   : str  = BASE_DIR,
    save_plots : bool = True,
) -> list[dict]:
    """
    複数被験者 × 複数タスク × 複数フェーズ を一括処理する。

    処理順序: 被験者 → タスク → フェーズ
    最適化  : 同一（被験者 × タスク）の .txt は1回だけ読み込む

    出力構造:
      {result_dir}/2026/{subject}/{task}/{speed}/
        ├── *_emg_normalized.csv
        ├── *_emg_enveloped.csv
        └── plots/
    """
    import pandas as pd

    if subjects is None:
        subjects = SUBJECTS

    summaries = []

    for subject in subjects:
        print(f"\n{'#'*55}")
        print(f"#  被験者: {subject}")
        print(f"{'#'*55}")

        for tk in task_keys:
            # ── 同じ .txt を1回だけ読む ──────────────────────────
            raw_path = build_emg_raw_path(tk, subject, base_dir)
            print(f"\n{'='*55}")
            print(f"  タスク: {tk}")
            print(f"  入力 : {raw_path}")
            print(f"{'='*55}")

            if not raw_path.exists():
                print(f"  [スキップ] ファイルが存在しません")
                for ph in phases:
                    summaries.append({
                        'status': 'skip', 'subject': subject,
                        'task': tk, 'phase': ph,
                    })
                continue

            try:
                df_all = load_cometa_txt(raw_path)
            except Exception as e:
                print(f"  [エラー] ファイル読み込み失敗: {e}")
                for ph in phases:
                    summaries.append({
                        'status': 'error', 'subject': subject,
                        'task': tk, 'phase': ph, 'error': str(e),
                    })
                continue

            # ベースライン（静止前 0〜40s）は1回だけ切り出す
            df_base = extract_phase(df_all, phase=0)
            proc    = EMGPreprocessor(fs=FS_EMG)

            # ── フェーズごとに処理 ────────────────────────────────
            for ph in phases:
                speed = phase_to_speed(ph)
                print(f"\n  --- Phase {ph} ({speed} m/s) ---")
                try:
                    df_walk = extract_phase(df_all, phase=ph)
                    if len(df_walk) == 0:
                        print(f"  [スキップ] データが空")
                        summaries.append({
                            'status': 'empty', 'subject': subject,
                            'task': tk, 'phase': ph, 'speed': speed,
                        })
                        continue

                    # 被験者名を含む出力先
                    out_dir  = build_out_dir(result_dir, subject, tk, speed)
                    plot_dir = out_dir / "plots"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    result = proc.run(
                        df_walk=df_walk, df_baseline=df_base,
                        task_key=tk, phase=ph, speed=speed,
                    )

                    proc.save_csv(result, out_dir, step='normalized')
                    proc.save_csv(result, out_dir, step='enveloped')

                    if save_plots:
                        t0 = float(df_walk['Time_s'].iloc[0])
                        plot_all_channels_normalized(
                            result,
                            save_path=plot_dir / f"all_channels_Ph{ph}_{speed}.png",
                            t_range=(t0, t0 + 10.0),
                        )
                        for ch_name in ['L_GM', 'R_SOL']:
                            if ch_name in result.channel_names:
                                idx = result.channel_names.index(ch_name)
                                plot_preprocessing_steps(
                                    result, channel_idx=idx,
                                    save_path=plot_dir / f"steps_{ch_name}_Ph{ph}_{speed}.png",
                                    t_range=(t0, t0 + 3.0),
                                )

                    summaries.append({
                        'status'      : 'ok',
                        'subject'     : subject,
                        'task'        : tk,
                        'phase'       : ph,
                        'speed'       : speed,
                        'n_channels'  : result.N_ch,
                        'n_samples'   : result.T,
                        'duration_s'  : result.T / result.fs,
                        'peak_mean_uV': float(result.peak_values.mean()),
                        'out_dir'     : str(out_dir),
                    })

                except Exception as e:
                    print(f"  [エラー] Phase {ph}: {e}")
                    summaries.append({
                        'status': 'error', 'subject': subject,
                        'task': tk, 'phase': ph, 'error': str(e),
                    })

    # ── サマリー CSV を保存 ───────────────────────────────────────
    df_sum   = pd.DataFrame(summaries)
    sum_path = Path(result_dir) / "2026" / "emg_preprocess_summary.csv"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(sum_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*55}")
    print(f"[バッチ完了]")
    print(f"  サマリー: {sum_path}")
    show_cols = [c for c in ['subject', 'task', 'phase', 'speed', 'status', 'n_samples']
                 if c in df_sum.columns]
    print(df_sum[show_cols].to_string(index=False))

    ok_count   = (df_sum['status'] == 'ok').sum()
    skip_count = (df_sum['status'] == 'skip').sum()
    err_count  = (df_sum['status'] == 'error').sum()
    print(f"\n  OK={ok_count}  スキップ={skip_count}  エラー={err_count}")

    return summaries


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMG 前処理パイプライン（複数被験者対応）")
    p.add_argument('--subjects',   nargs='+', default=None,
                   help="被験者名リスト（複数指定可: --subjects Ide Tanaka）")
    p.add_argument('--task',       default=None,
                   choices=['task01', 'task02', 'task03'],
                   help="処理するタスク")
    p.add_argument('--phase',      type=int, default=None,
                   help="フェーズ番号 1〜5")
    p.add_argument('--all_phases', action='store_true',
                   help="全フェーズ（1〜5）を処理")
    p.add_argument('--all_tasks',  action='store_true',
                   help="全タスク（task01/02/03）を処理")
    p.add_argument('--result_dir', default=RESULT_DIR)
    p.add_argument('--base_dir',   default=BASE_DIR)
    p.add_argument('--no_plots',   action='store_true', help="プロット生成をスキップ")
    p.add_argument('--all_steps',  action='store_true', help="全ステップCSVを保存")
    return p.parse_args()


def main() -> None:
    args       = _parse_args()
    all_phases = list(PHASES.keys())          # [1, 2, 3, 4, 5]
    all_tasks  = list(TASKS)                  # ['task01', 'task02', 'task03']
    save_plots = not args.no_plots

    # ── 被験者リストの決定 ────────────────────────────────────────
    if args.subjects:
        subjects = args.subjects
    else:
        # 対話モード
        print("=========================================")
        print("  EMG 前処理パイプライン（複数被験者対応）")
        print("=========================================")
        print(f"  登録被験者: {SUBJECTS}")
        inp = input("被験者名を入力（複数はスペース区切り、Enterで全員）: ").strip()
        subjects = inp.split() if inp else SUBJECTS

    # ── タスクの決定 ─────────────────────────────────────────────
    if args.all_tasks:
        task_keys = all_tasks
    elif args.task:
        task_keys = [args.task]
    else:
        if not args.subjects:   # 対話モード
            inp = input("タスク [task01/task02/task03] (Enterで全タスク): ").strip()
            task_keys = [inp] if inp in all_tasks else all_tasks
        else:
            task_keys = all_tasks

    # ── フェーズの決定 ───────────────────────────────────────────
    if args.all_phases:
        phases = all_phases
    elif args.phase:
        phases = [args.phase]
    else:
        if not args.subjects:   # 対話モード
            inp = input("フェーズ番号 [1-5] (Enterで全フェーズ): ").strip()
            if inp.isdigit() and int(inp) in all_phases:
                phases = [int(inp)]
            else:
                phases = all_phases
        else:
            phases = all_phases

    # ── 実行 ─────────────────────────────────────────────────────
    print(f"\n  実行設定:")
    print(f"    被験者  : {subjects}")
    print(f"    タスク  : {task_keys}")
    print(f"    フェーズ: {phases}")
    print(f"    出力先  : {args.result_dir}\\2026\\{{被験者}}\\{{タスク}}\\{{速度}}\\\n")

    if len(subjects) == 1 and len(task_keys) == 1 and len(phases) == 1:
        run_single(
            task_key   = task_keys[0],
            phase      = phases[0],
            subject    = subjects[0],
            result_dir = args.result_dir,
            base_dir   = args.base_dir,
            save_plots = save_plots,
            save_all_steps = args.all_steps,
        )
    else:
        run_batch(
            task_keys  = task_keys,
            phases     = phases,
            subjects   = subjects,
            result_dir = args.result_dir,
            base_dir   = args.base_dir,
            save_plots = save_plots,
        )


if __name__ == '__main__':
    main()
