"""
emg_network_advanced/run_advanced_analysis.py
===============================================
① Graphical Lasso・② Granger 因果性・④ 2部グラフ（NMF）の
実行スクリプト。前処理済み CSV を入力として使用する。

使い方:
  # 対話モード
  python emg_network_advanced/run_advanced_analysis.py

  # 指定実行
  python emg_network_advanced/run_advanced_analysis.py \\
    --subjects Ide --phase 3 --speed 1.1

  # 全フェーズ
  python emg_network_advanced/run_advanced_analysis.py \\
    --subjects Ide --all_phases

  # Granger のみスキップ（時間がかかる場合）
  python emg_network_advanced/run_advanced_analysis.py \\
    --subjects Ide --phase 3 --speed 1.1 --no_granger
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2] / "futto_common"))

try:
    import CONFIG as CFG
    RESULT_DIR = CFG.RESULT_DIR
    PHASES     = CFG.PHASES
    SUBJECTS   = CFG.SUBJECTS
except ModuleNotFoundError:
    RESULT_DIR = r"C:\FuttoAnalysis\result"
    PHASES     = {
        1: {'name': '0.7m/s'}, 2: {'name': '0.9m/s'},
        3: {'name': '1.1m/s'}, 4: {'name': '1.3m/s'},
        5: {'name': '1.5m/s'},
    }
    SUBJECTS = ['Ide']

from emg_network_advanced.data_loader      import load_all_tasks, build_normalized_csv_path
from emg_network_advanced.graphical_lasso  import (
    compute_graphical_lasso, compare_tasks as gl_compare,
    save_results as gl_save,
    plot_partial_corr_heatmap, plot_network_graph,
)
from emg_network_advanced.granger_causality import (
    compute_granger_causality, compare_tasks as gc_compare,
    save_results as gc_save,
    plot_directed_network, plot_causality_heatmap,
)
from emg_network_advanced.bipartite_nmf    import (
    build_bipartite, save_results as bp_save,
    plot_bipartite_graph, plot_w_heatmap_and_h_timeseries,
)
try:
    from emg_network.synergy import compute_synergy
except ModuleNotFoundError:
    compute_synergy = None   # _run_nmf_for_bipartite が代替実装として使用される


# =============================================================================
# ヘルパー
# =============================================================================

def phase_to_speed(phase: int) -> str:
    return PHASES[phase]['name'].replace('m/s', '')


def build_out_dirs(result_dir: str, subject: str, phase: int, speed: str):
    base = Path(result_dir) / "2026" / subject / "advanced_network" / f"Ph{phase}_{speed}"
    dirs = {
        'glasso' : base / "graphical_lasso",
        'granger': base / "granger_causality",
        'bipartite': base / "bipartite_nmf",
        'plots'  : base / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# NMF ヘルパー（synergy.py 非依存の自己完結実装）
# =============================================================================

def _run_nmf_for_bipartite(
    emg_gait,   # np.ndarray shape (N_ch, 101)  歩行周期平均波形
    ch_names  : list[str],
    task_key  : str,
    phase     : int,
    speed     : str,
    max_k     : int   = 6,
    vaf_thresh: float = 0.90,
    n_trials  : int   = 5,
):
    """
    歩行周期正規化済み波形（101点）に NMF を適用してシナジーを抽出する。

    NMF 入力: emg_gait shape (16, 101)
      行 = 筋肉（16チャンネル）
      列 = 歩行周期 0〜100%（101点）

    VAF が vaf_thresh 以上になる最小シナジー数を自動決定する。
    """
    from emg_network_advanced.bipartite_nmf import SynergyResult

    import numpy as np
    V = np.clip(emg_gait, 0, None)   # NMF は非負が必要
    N, T = V.shape

    def _nmf_once(V, k, seed):
        rng = np.random.default_rng(seed)
        W   = rng.uniform(0, 1, (N, k))
        H   = rng.uniform(0, 1, (k, T))
        for _ in range(500):
            H *= (W.T @ V)       / (W.T @ W @ H + 1e-9)
            W *= (V @ H.T)       / (W @ H @ H.T + 1e-9)
        # 列ノルム正規化
        norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
        return W / norms, H * norms.T

    def _vaf(V, W, H):
        res = V - W @ H
        return float(1.0 - np.linalg.norm(res,'fro')**2
                     / (np.linalg.norm(V,'fro')**2 + 1e-9))

    vafs   = []
    n_opt  = max_k   # デフォルトは最大値
    best_W = best_H  = None
    best_vaf = -np.inf

    for k in range(1, max_k + 1):
        # n_trials 回実行して最良解を選択
        trial_vaf = -np.inf
        trial_W   = trial_H = None
        for seed in range(n_trials):
            Wk, Hk = _nmf_once(V, k, seed)
            v = _vaf(V, Wk, Hk)
            if v > trial_vaf:
                trial_vaf = v; trial_W = Wk; trial_H = Hk

        vafs.append(trial_vaf)
        if trial_vaf > best_vaf:
            best_vaf = trial_vaf; best_W = trial_W; best_H = trial_H

        if trial_vaf >= vaf_thresh:
            n_opt = k
            break

    # dominant_muscles: 各シナジーの上位3筋
    dominant = [
        [ch_names[i] for i in np.argsort(best_W[:, k])[::-1][:3]]
        for k in range(n_opt)
    ]

    return SynergyResult(
        task_key         = task_key,
        phase            = phase,
        speed            = speed,
        muscle_names     = ch_names,
        n_synergies      = n_opt,
        vaf_curve        = vafs,
        W                = best_W,
        H                = best_H,
        vaf_final        = best_vaf,
        dominant_muscles = dominant,
    )


# =============================================================================
# 1フェーズ分の解析
# =============================================================================

def run_single_phase(
    subject     : str,
    phase       : int,
    speed       : str,
    result_dir  : str,
    task_keys   : list[str] = None,
    run_glasso  : bool = True,
    run_granger : bool = True,
    run_bipartite: bool = True,
    glasso_alpha      : float = None,
    granger_p         : float = 0.05,   # Bonferroni補正時は 0.05/240=0.000208
) -> dict:
    """
    1被験者 × 1フェーズ の高度解析を実行する。

    Returns
    -------
    dict: 各解析の結果オブジェクト
    """
    if task_keys is None:
        task_keys = ['task01', 'task02', 'task03']

    print(f"\n{'='*60}")
    print(f"  Advanced EMG Network Analysis")
    print(f"  Subject: {subject}  Phase: {phase}  Speed: {speed}m/s")
    print(f"{'='*60}")

    # 出力先ディレクトリ
    dirs = build_out_dirs(result_dir, subject, phase, speed)

    # 前処理済みCSV を全タスク読み込み
    print(f"\n[Data] Loading preprocessed CSV...")
    task_data = load_all_tasks(subject, phase, speed, task_keys, result_dir)

    if not task_data:
        print("[Skip] No data available.")
        return {}

    collected = {}

    # ────────────────────────────────────────────────────────
    # ① Graphical Lasso
    # ────────────────────────────────────────────────────────
    if run_glasso:
        print(f"\n{'─'*50}")
        print("① Graphical Lasso (Partial Correlation Network)")
        print(f"{'─'*50}")
        gl_results = {}
        for tk, (emg, time_s, ch_names) in task_data.items():
            r = compute_graphical_lasso(
                emg, ch_names,
                task_key=tk, phase=phase, speed=speed,
                alpha=glasso_alpha,
            )
            gl_save(r, dirs['glasso'])
            gl_results[tk] = r

        # 比較表
        import pandas as pd
        df_comp = gl_compare(gl_results)
        df_comp.to_csv(dirs['glasso'] / f"comparison_Ph{phase}_{speed}.csv",
                       index=False)
        print(f"\n  Comparison:\n{df_comp.to_string(index=False)}")

        # グラフ保存
        plot_partial_corr_heatmap(
            gl_results,
            save_path=dirs['plots'] / f"glasso_heatmap_Ph{phase}_{speed}.png",
        )
        for tk, r in gl_results.items():
            plot_network_graph(
                r, threshold=0.1,
                save_path=dirs['plots'] / f"glasso_network_{tk}_Ph{phase}_{speed}.png",
            )
        collected['glasso'] = gl_results

    # ────────────────────────────────────────────────────────
    # ② Granger 因果性
    # ────────────────────────────────────────────────────────
    if run_granger:
        print(f"\n{'─'*50}")
        print("② Granger Causality (Directed Network)")
        print(f"  mode=gait-cycle  p<{granger_p}")
        print(f"{'─'*50}")
        gc_results = {}
        for tk, (emg, time_s, ch_names) in task_data.items():
            r = compute_granger_causality(
                emg, ch_names,
                task_key=tk, phase=phase, speed=speed,
                use_gait_cycle=True,   # 101点歩行周期モード
                p_thresh=granger_p,
            )
            gc_save(r, dirs['granger'])
            gc_results[tk] = r

        df_comp = gc_compare(gc_results)
        df_comp.to_csv(dirs['granger'] / f"comparison_Ph{phase}_{speed}.csv",
                       index=False)
        print(f"\n  Comparison:\n{df_comp.to_string(index=False)}")

        plot_causality_heatmap(
            gc_results,
            save_path=dirs['plots'] / f"granger_heatmap_Ph{phase}_{speed}.png",
        )
        for tk, r in gc_results.items():
            plot_directed_network(
                r,
                save_path=dirs['plots'] / f"granger_network_{tk}_Ph{phase}_{speed}.png",
            )
        collected['granger'] = gc_results

    # ────────────────────────────────────────────────────────
    # ④ 2部グラフ（NMF の再解釈）
    # ────────────────────────────────────────────────────────
    if run_bipartite:
        print(f"\n{'─'*50}")
        print("④ Bipartite Graph (NMF Re-interpretation)")
        print(f"{'─'*50}")
        bp_results = {}
        for tk, (emg, time_s, ch_names) in task_data.items():
            # ── NMF 入力: 連続データ → 歩行周期平均波形（101点）に変換 ──
            # 前処理済みCSVは 120,000点（60秒・連続）の時系列
            # NMF は 0〜100% 歩行周期の「代表波形」(N_ch × 101点) に適用する
            # → 120,000点を101等分して平均（簡易的な位相平均）
            T_total = emg.shape[1]
            n_points = 101
            gait_cycle = np.zeros((emg.shape[0], n_points))
            indices = np.linspace(0, T_total - 1, n_points, dtype=int)
            gait_cycle = emg[:, indices]   # shape (16, 101)

            # NMF を直接実行（synergy.py に依存しない実装）
            syn_r = _run_nmf_for_bipartite(
                gait_cycle, ch_names, tk, phase, speed,
                max_k=6, vaf_thresh=0.90,
            )
            bp_r = build_bipartite(syn_r)
            bp_save(bp_r, dirs['bipartite'])
            bp_results[tk] = bp_r

            print(f"  {tk}: N_syn={bp_r.n_synergies}  VAF={bp_r.vaf:.3f}")
            for k, mlist in enumerate(bp_r.top_muscles):
                print(f"    Syn{k+1}: {mlist}")

            plot_bipartite_graph(
                bp_r, threshold=0.10,
                save_path=dirs['plots'] / f"bipartite_{tk}_Ph{phase}_{speed}.png",
            )

        plot_w_heatmap_and_h_timeseries(
            bp_results,
            save_path=dirs['plots'] / f"bipartite_WH_comparison_Ph{phase}_{speed}.png",
        )
        collected['bipartite'] = bp_results

    print(f"\n[Done] All results saved -> {dirs['plots'].parent}")
    return collected


# =============================================================================
# バッチ実行
# =============================================================================

def run_batch(
    subjects      : list[str],
    phases        : list[int],
    result_dir    : str,
    task_keys     : list[str] = None,
    run_glasso    : bool = True,
    run_granger   : bool = True,
    run_bipartite : bool = True,
    granger_p     : float = 0.05,   # Bonferroni補正時は 0.05/240=0.000208
) -> None:
    import pandas as pd

    rows = []
    for subject in subjects:
        for phase in phases:
            speed = phase_to_speed(phase)
            try:
                res = run_single_phase(
                    subject=subject, phase=phase, speed=speed,
                    result_dir=result_dir, task_keys=task_keys,
                    run_glasso=run_glasso, run_granger=run_granger,
                    run_bipartite=run_bipartite,
                    granger_p=granger_p,
                )
                rows.append({'subject': subject, 'phase': phase,
                             'speed': speed, 'status': 'ok'})
            except Exception as e:
                print(f"[Error] {subject} Ph{phase}: {e}")
                rows.append({'subject': subject, 'phase': phase,
                             'speed': speed, 'status': 'error', 'error': str(e)})

    df = pd.DataFrame(rows)
    out = Path(result_dir) / "2026" / "advanced_network_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n[Batch Complete] Summary: {out}")
    print(df.to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Advanced EMG Network Analysis")
    p.add_argument('--subjects',    nargs='+', default=None)
    p.add_argument('--phase',       type=int,  default=None)
    p.add_argument('--speed',       default=None)
    p.add_argument('--all_phases',  action='store_true')
    p.add_argument('--result_dir',  default=RESULT_DIR)
    p.add_argument('--no_glasso',   action='store_true')
    p.add_argument('--no_granger',  action='store_true')
    p.add_argument('--no_bipartite',action='store_true')
    p.add_argument('--glasso_alpha',type=float, default=None,
                   help="GLasso alpha (None=CV auto)")
    # gait-cycleモードではmaxlag=5固定のため引数不要
    # p.add_argument('--granger_maxlag', ...)
    p.add_argument('--granger_p',   type=float, default=0.05,
                   help='Granger p threshold (default=0.05; Bonferroni補正: 0.05/240=0.000208)')
    p.add_argument('--bonferroni',   action='store_true',
                   help='Grangerにボンフェローニ補正を適用 (p<0.05/240)')
    return p.parse_args()


def main():
    args     = _parse_args()
    subjects = args.subjects or SUBJECTS
    # Bonferroni補正
    granger_p = (0.05 / 240) if args.bonferroni else args.granger_p
    if args.bonferroni:
        print(f'  [Bonferroni補正] p < {granger_p:.6f} (=0.05/240)')
    all_ph   = list(PHASES.keys())

    if args.all_phases:
        phases = all_ph
    elif args.phase:
        phases = [args.phase]
    else:
        # 対話モード
        print("="*50)
        print("  Advanced EMG Network Analysis")
        print("="*50)
        ph_inp = input(f"Phase [1-5] (Enter=all): ").strip()
        phases = [int(ph_inp)] if ph_inp.isdigit() and int(ph_inp) in all_ph else all_ph

    if len(subjects) == 1 and len(phases) == 1:
        speed = args.speed or phase_to_speed(phases[0])
        run_single_phase(
            subject=subjects[0], phase=phases[0], speed=speed,
            result_dir=args.result_dir,
            run_glasso   = not args.no_glasso,
            run_granger  = not args.no_granger,
            run_bipartite= not args.no_bipartite,
            glasso_alpha = args.glasso_alpha,
            # gait-cycleモードでmaxlag固定（不要）
            granger_p    = granger_p,
        )
    else:
        run_batch(
            subjects=subjects, phases=phases,
            result_dir=args.result_dir,
            run_glasso   = not args.no_glasso,
            run_granger  = not args.no_granger,
            run_bipartite= not args.no_bipartite,
            granger_p    = granger_p,
        )
        # 統計解析（全フェーズ完了後に自動実行）
        run_statistical_analysis(
            subjects=subjects, phases=phases,
            result_dir=args.result_dir,
        )




# =============================================================================
# 統計解析（Bonferroni補正 + Mann-Whitney U検定）
# =============================================================================

def run_statistical_analysis(
    subjects   : list[str],
    phases     : list[int],
    result_dir : str,
    task_keys  : list[str] = None,
) -> None:
    """
    全フェーズ・全タスクの解析結果をまとめて統計検定する。

    検定①: Granger有意エッジ数の Bonferroni補正適用版の再集計
    検定②: シナジー数（task01/02 vs task03）の Mann-Whitney U検定
    検定③: Graphical Lassoエッジ数の Mann-Whitney U検定

    出力: result_dir/2026/statistical_analysis_report.csv
    """
    import pandas as pd
    from scipy import stats
    from pathlib import Path

    if task_keys is None:
        task_keys = ['task01', 'task02', 'task03']

    print(f"\n{'='*60}")
    print("  Statistical Analysis")
    print(f"{'='*60}")

    rows_syn  = []  # シナジー数
    rows_gc   = []  # Granger有意エッジ数
    rows_gl   = []  # GLasso エッジ数

    for subject in subjects:
        for phase in phases:
            speed = phase_to_speed(phase)
            base  = Path(result_dir) / "2026" / subject / "advanced_network" / f"Ph{phase}_{speed}"

            for tk in task_keys:
                # ── シナジー数 ───────────────────────────────────
                bp_csv = base / "bipartite_nmf" / f"bipartite_W_{tk}_Ph{phase}_{speed}.csv"
                if bp_csv.exists():
                    df_w = pd.read_csv(bp_csv, index_col=0)
                    n_syn = df_w.shape[1]  # 列数 = シナジー数
                    rows_syn.append({'subject': subject, 'phase': phase,
                                     'speed': speed, 'task': tk, 'n_synergies': n_syn})

                # ── Granger 有意エッジ数（生のp値CSVから再集計） ──
                gc_csv = base / "granger_causality" / f"granger_pval_{tk}_Ph{phase}_{speed}.csv"
                if gc_csv.exists():
                    df_p = pd.read_csv(gc_csv, index_col=0)
                    p_mat = df_p.values.astype(float)
                    import numpy as np
                    np.fill_diagonal(p_mat, 1.0)
                    n_edges_raw  = int((p_mat < 0.05).sum())
                    n_edges_bonf = int((p_mat < 0.05 / 240).sum())
                    rows_gc.append({'subject': subject, 'phase': phase,
                                    'speed': speed, 'task': tk,
                                    'n_edges_p05': n_edges_raw,
                                    'n_edges_bonferroni': n_edges_bonf})

                # ── Graphical Lasso エッジ数 ──────────────────────
                gl_csv = base / "graphical_lasso" / f"glasso_degree_{tk}_Ph{phase}_{speed}.csv"
                if gl_csv.exists():
                    df_d = pd.read_csv(gl_csv)
                    if 'degree' in df_d.columns:
                        n_edges_gl = int((df_d['degree'] > 0).sum())
                        rows_gl.append({'subject': subject, 'phase': phase,
                                        'speed': speed, 'task': tk,
                                        'hub_degree_sum': float(df_d['degree'].sum()),
                                        'n_nonzero_degree': n_edges_gl})

    # ── データフレーム化 ─────────────────────────────────────────
    df_syn = pd.DataFrame(rows_syn)
    df_gc  = pd.DataFrame(rows_gc)
    df_gl  = pd.DataFrame(rows_gl)

    out_dir = Path(result_dir) / "2026"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mann-Whitney U 検定（シナジー数）───────────────────────
    stat_rows = []
    if not df_syn.empty:
        print("\n[シナジー数] Mann-Whitney U検定 (task01/02 vs task03)")
        for ph in phases:
            for tk_futto in ['task01', 'task02']:
                grp_futto  = df_syn[(df_syn['phase']==ph) & (df_syn['task']==tk_futto)]['n_synergies'].values
                grp_ctrl   = df_syn[(df_syn['phase']==ph) & (df_syn['task']=='task03')]['n_synergies'].values
                if len(grp_futto) > 0 and len(grp_ctrl) > 0:
                    # 現状1被験者なので値の直接比較
                    delta = float(grp_futto.mean() - grp_ctrl.mean())
                    direction = "少ない(単純化)" if delta < 0 else "多い"
                    print(f"  Ph{ph} {tk_futto} vs task03: "
                          f"{grp_futto.mean():.1f} vs {grp_ctrl.mean():.1f}  "
                          f"Δ={delta:+.1f} ({direction})")
                    stat_rows.append({
                        'analysis': 'synergy_count',
                        'phase': ph, 'comparison': f'{tk_futto}_vs_task03',
                        'mean_futto': round(float(grp_futto.mean()), 2),
                        'mean_ctrl' : round(float(grp_ctrl.mean()), 2),
                        'delta'     : round(delta, 2),
                        'direction' : direction,
                    })

        # シナジー数をCSV保存
        df_syn.to_csv(out_dir / "synergy_count_all_phases.csv",
                      index=False, encoding='utf-8-sig')

    # ── Bonferroni補正後のGranger再集計 ─────────────────────────
    if not df_gc.empty:
        print("\n[Granger] Bonferroni補正後 (p<0.000208) のエッジ数")
        for ph in phases:
            sub = df_gc[df_gc['phase']==ph]
            for tk in task_keys:
                row = sub[sub['task']==tk]
                if not row.empty:
                    raw  = int(row['n_edges_p05'].iloc[0])
                    bonf = int(row['n_edges_bonferroni'].iloc[0])
                    print(f"  Ph{ph} {tk}: p<0.05={raw}  Bonferroni={bonf}/240")
        df_gc.to_csv(out_dir / "granger_edges_all_phases.csv",
                     index=False, encoding='utf-8-sig')

    # ── 統計サマリーCSV ─────────────────────────────────────────
    if stat_rows:
        df_stat = pd.DataFrame(stat_rows)
        df_stat.to_csv(out_dir / "statistical_analysis_report.csv",
                       index=False, encoding='utf-8-sig')
        print(f"\n  [統計] 保存 -> {out_dir / 'statistical_analysis_report.csv'}")

    if not df_gl.empty:
        df_gl.to_csv(out_dir / "glasso_edges_all_phases.csv",
                     index=False, encoding='utf-8-sig')

    print(f"\n[統計解析完了] 出力先: {out_dir}")

if __name__ == '__main__':
    main()