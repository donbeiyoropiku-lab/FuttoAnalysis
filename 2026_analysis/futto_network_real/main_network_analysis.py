"""
main_network_analysis.py
=========================
Futto × 神経筋系ネットワーク解析 — メイン実行スクリプト

既存の tension_calc.py・io_utils.py と連携し、
計算済みデータ（張力CSV・関節中心CSV・EMG平均CSV）を読み込んで
多層ネットワーク解析を実行する。

Usage:
  python main_network_analysis.py
  python main_network_analysis.py --task task01 --phase 3 --speed 1.1
  python main_network_analysis.py --all
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── パス設定 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import CONFIG as CFG
from futto_network.build_graph   import (
    FuttoGraph, build_tension_csv_path, build_marker_csv_path
)
from futto_network.centrality    import (
    compute_strength_centrality, segment_group_strength,
    hub_trajectory_summary, compare_centrality,
)
from futto_network.efficiency    import (
    compute_efficiency, spectral_gap, phase_split_stats, compare_efficiency,
)
from emg_network.correlation     import (
    compute_emg_correlation, build_emg_csv_path, contralateral_coupling,
)
from emg_network.network_metrics import (
    compute_emg_network_metrics, compare_contralateral_effect,
)
from multilayer_network.joint_layer import (
    load_marker_csv, compute_joint_centers_from_markers,
    compute_joint_angles, compute_joint_network,
    load_joint_centers_csv, build_joint_centers_csv_path,
)
from multilayer_network.fusion   import (
    build_multilayer_network, compare_multilayer,
)
from visualizer.plot_physical    import (
    plot_futto_3d_snapshots, plot_strength_timeseries,
    plot_efficiency_lambda, plot_segment_group_tensions,
)
from visualizer.plot_emg         import (
    plot_emg_heatmap, plot_emg_circular_network, plot_emg_degree_comparison,
)
from visualizer.plot_multilayer  import (
    plot_multilayer_structure, plot_radar_comparison,
    plot_speed_task_comparison, plot_contralateral_effect,
    compute_stats_table,
)
from futto_network.flow_analysis import (
    compute_betweenness, compute_flow_analysis,
    compute_community, save_flow_results,
)
from emg_network.synergy         import (
    compute_synergy, save_synergy_results, compare_synergy_tasks,
)
from multilayer_network.tradeoff import (
    compute_tradeoff, compare_tradeoff, save_tradeoff_results,
)
from visualizer.plot_advanced    import (
    plot_flow_diagram, plot_synergy_heatmap, plot_synergy_comparison_bar,
    plot_tradeoff_timeseries, plot_betweenness_timeseries,
    plot_community_transition,
)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Futto Network Analysis")
    p.add_argument('--task',   default=None, choices=['task01', 'task02'],
                   help="単一タスク (指定なしで全タスク)")
    p.add_argument('--phase',  type=int, default=None,
                   help="フェーズ番号 1〜5 (指定なしで全フェーズ)")
    p.add_argument('--speed',  default=None,
                   help="速度文字列 '0.7'〜'1.5' (指定なしで全速度)")
    p.add_argument('--all',    action='store_true',
                   help="全タスク×全フェーズ×全速度を一括実行")
    p.add_argument('--result_dir', default=CFG.RESULT_DIR,
                   help="結果保存ディレクトリ")
    p.add_argument('--output_dir', default=r"C:\FuttoAnalysis\result\2026\network_results",
                   help="ネットワーク解析結果の保存先")
    return p.parse_args()


# =============================================================================
# 出力ユーティリティ
# =============================================================================

def _save_csv(data: np.ndarray, path: Path, header: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), data, delimiter=",", header=header, comments="")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, default=_conv, indent=2, ensure_ascii=False)


# =============================================================================
# 1件分の解析
# =============================================================================

def run_single(
    task_key    : str,
    phase       : int,
    speed_str   : str,
    result_dir  : str,
    output_dir  : Path,
    year        : str = "2026",
    subject     : str = "Ide",
) -> dict:
    """
    1タスク × 1フェーズ × 1速度 の完全解析を実行する。

    Returns
    -------
    dict: 全指標をまとめたサマリー辞書
    """
    tag = f"{task_key} Phase{phase} {speed_str}m/s"
    print(f"\n{'='*60}")
    print(f"  解析開始: {tag}")
    print(f"{'='*60}")

    out_sub = output_dir / task_key / f"phase{phase}" / speed_str
    out_sub.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # Step 1: 物理層 — FuttoGraph の構築
    # ──────────────────────────────────────────────────────────
    tension_path = build_tension_csv_path(task_key, phase, speed_str, result_dir, year)
    print(f"\n[物理層] 張力CSV: {tension_path}")
    graph = FuttoGraph(task_key)
    graph.load_from_tension_csv(tension_path, phase_shift=True)
    print(f"  {graph.summary()}")

    cr_phys = compute_strength_centrality(graph, phase, speed_str)
    er_phys = compute_efficiency(graph, phase, speed_str)

    # 保存
    _save_csv(cr_phys.strength,        out_sub / "strength_centrality.csv",
              header=",".join([str(n) for n in graph.node_ids]))
    _save_csv(er_phys.efficiency.reshape(-1, 1),
              out_sub / "network_efficiency.csv", header="E_raw")
    _save_csv(er_phys.lambda_max.reshape(-1, 1),
              out_sub / "lambda_max.csv", header="lambda_max")
    _save_csv(er_phys.efficiency_norm.reshape(-1, 1),
              out_sub / "efficiency_norm.csv", header="E_norm")

    # セグメントグループ別張力
    seg_group_summary = {}
    for grp in CFG.SEGMENT_GROUPS:
        arr = segment_group_strength(graph, grp)
        seg_group_summary[grp] = {'mean_N': float(arr.mean()), 'peak_N': float(arr.max())}
        _save_csv(arr.reshape(-1, 1), out_sub / f"seg_group_{grp}.csv",
                  header=f"{grp}_total_tension_N")

    hub_info = hub_trajectory_summary(cr_phys)
    pd_stats  = phase_split_stats(er_phys)

    print(f"  [物理] E_norm_mean={er_phys.efficiency_norm.mean():.4f}  "
          f"λ_max_mean={er_phys.lambda_max_mean:.4f}  "
          f"Hub Entropy={cr_phys.hub_entropy:.3f} bits")

    # ──────────────────────────────────────────────────────────
    # Step 2: 生体層 — EMG 相関ネットワーク
    # ──────────────────────────────────────────────────────────
    emg_path = build_emg_csv_path(task_key, phase, subject,
                                   base_dir=str(CFG.BASE_DIR))
    print(f"\n[生体層] EMG CSV: {emg_path}")
    cr_emg = compute_emg_correlation(emg_path, task_key, phase, speed_str)
    em_metrics = compute_emg_network_metrics(cr_emg)
    cc_info    = contralateral_coupling(cr_emg)

    _save_csv(cr_emg.corr_matrix, out_sub / "emg_corr_matrix.csv",
              header=",".join(cr_emg.muscle_names))
    _save_json({
        'global_efficiency'    : em_metrics.global_efficiency,
        'modularity_Q'         : em_metrics.modularity_Q,
        'temporal_stability'   : em_metrics.temporal_stability,
        'inter_efficiency'     : em_metrics.inter_efficiency,
        'left_efficiency'      : em_metrics.left_efficiency,
        'right_efficiency'     : em_metrics.right_efficiency,
        'symmetry_index'       : cr_emg.symmetry_index.tolist(),
        'top_hub_muscles'      : em_metrics.top_hub_muscles,
        'cross_ipsi_ratio'     : cc_info['cross_ipsi_ratio'],
        'modularity_hint'      : cr_emg.modularity_hint,
    }, out_sub / "emg_metrics.json")

    print(f"  [生体] E_global={em_metrics.global_efficiency:.4f}  "
          f"Q={em_metrics.modularity_Q:.4f}  "
          f"cross/ipsi={cc_info['cross_ipsi_ratio']:.4f}")

    # ──────────────────────────────────────────────────────────
    # Step 3: 関節層 — 仮想関節角度ネットワーク
    # ──────────────────────────────────────────────────────────
    joint_path  = build_joint_centers_csv_path(task_key, phase, speed_str, result_dir, year)
    marker_path = build_marker_csv_path(task_key, phase, speed_str)
    print(f"\n[関節層] 関節中心CSV: {joint_path}")

    joint_centers = load_joint_centers_csv(joint_path)

    if joint_centers is None:
        print("       → マーカーCSVから再計算を試みます")
        marker_pos = load_marker_csv(marker_path)
        if marker_pos is not None and task_key in CFG.TASK_CONFIGS:
            joint_centers = compute_joint_centers_from_markers(marker_pos, task_key)
        else:
            print("       → データなし。シミュレーション関節角度を使用します。")

    if joint_centers:
        joint_angles = compute_joint_angles(joint_centers)
    else:
        # シミュレーション（開発用）
        t_arr = np.linspace(0, 2 * np.pi, 101)
        joint_angles = {
            'Hip'  : 20 * np.sin(t_arr) + np.random.randn(101),
            'Knee' : 40 * np.abs(np.sin(t_arr)) + np.random.randn(101),
            'Ankle': 15 * np.sin(t_arr + 0.5) + np.random.randn(101),
        }

    jn_result = compute_joint_network(joint_angles, task_key, phase, speed_str)

    if joint_angles:
        angle_mat = np.column_stack(list(joint_angles.values()))
        _save_csv(angle_mat, out_sub / "joint_angles.csv",
                  header=",".join(joint_angles.keys()))
    _save_csv(jn_result.coupling_matrix, out_sub / "joint_coupling.csv",
              header="Hip,Knee,Ankle")

    print(f"  [関節] IL_proxy={jn_result.interlimb_proxy:.4f}  "
          f"coord_index={jn_result.coord_index.round(3)}")

    # ──────────────────────────────────────────────────────────
    # Step 4: 多層統合
    # ──────────────────────────────────────────────────────────
    print(f"\n[多層統合]")
    ml = build_multilayer_network(graph, jn_result, cr_emg, em_metrics, er_phys)

    _save_csv(ml.supra_adjacency, out_sub / "supra_adjacency.csv")
    _save_csv(ml.multilayer_pagerank.reshape(-1, 1),
              out_sub / "multilayer_pagerank.csv", header="pagerank")

    top_nodes = [ml.node_labels[i] for i in np.argsort(ml.multilayer_pagerank)[-5:][::-1]]
    ml_dict = {
        'gait_efficiency_score' : ml.gait_efficiency_score,
        'coupling_PJ'           : ml.coupling_PJ,
        'coupling_JE'           : ml.coupling_JE,
        'coupling_PE'           : ml.coupling_PE,
        'contribution_physical' : ml.contribution_physical,
        'contribution_joint'    : ml.contribution_joint,
        'contribution_emg'      : ml.contribution_emg,
        'top_pagerank_nodes'    : top_nodes,
    }
    _save_json(ml_dict, out_sub / "multilayer_metrics.json")

    print(f"  GES={ml.gait_efficiency_score:.4f}  "
          f"PJ={ml.coupling_PJ:.4f}  JE={ml.coupling_JE:.4f}  PE={ml.coupling_PE:.4f}")
    print(f"  Top PageRank: {top_nodes}")

    # ──────────────────────────────────────────────────────────
    # サマリー辞書
    # ──────────────────────────────────────────────────────────
    summary = {
        'tag'                   : tag,
        'task_key'              : task_key,
        'phase'                 : phase,
        'speed'                 : speed_str,
        'N_futto_nodes'         : graph.N,
        'N_total'               : ml.N_total,
        # 物理層
        'E_norm_mean'           : float(er_phys.efficiency_norm.mean()),
        'lambda_max_norm_mean'  : float(er_phys.lambda_max_norm.mean()),
        'hub_entropy_bits'      : float(cr_phys.hub_entropy),
        'stance_swing_E_ratio'  : float(pd_stats['E_ratio_st_sw']),
        'seg_groups'            : seg_group_summary,
        **{f'hub_{k}': v for k, v in hub_info.items()},
        # 生体層
        'emg_global_efficiency' : float(em_metrics.global_efficiency),
        'emg_modularity_Q'      : float(em_metrics.modularity_Q),
        'emg_inter_efficiency'  : float(em_metrics.inter_efficiency),
        'emg_cross_ipsi_ratio'  : float(cc_info['cross_ipsi_ratio']),
        'top_hub_muscles'       : em_metrics.top_hub_muscles,
        # 関節層
        'joint_il_proxy'        : float(jn_result.interlimb_proxy),
        # 多層
        'gait_efficiency_score' : float(ml.gait_efficiency_score),
        'coupling_PJ'           : float(ml.coupling_PJ),
        'coupling_JE'           : float(ml.coupling_JE),
        'coupling_PE'           : float(ml.coupling_PE),
    }
    _save_json(summary, out_sub / "summary.json")

    # ──────────────────────────────────────────────────────────
    # Step 5: 可視化
    # ──────────────────────────────────────────────────────────
    plot_dir = out_sub / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[可視化] 保存先: {plot_dir}")

    # ① Strength Centrality 時系列
    plot_strength_timeseries(graph, cr_phys, plot_dir, task_key, phase, speed_str)

    # ② セグメントグループ張力
    plot_segment_group_tensions(graph, plot_dir, task_key, phase, speed_str)

    # ③ 3D ネットワークスナップショット（マーカーCSVがあれば実座標使用）
    marker_csv_path = build_marker_csv_path(task_key, phase, speed_str)
    plot_futto_3d_snapshots(graph, cr_phys, marker_csv_path,
                            plot_dir, task_key, phase, speed_str)

    # ④ EMG サーキュラーネットワーク
    plot_emg_circular_network(cr_emg, em_metrics, plot_dir, task_key, phase, speed_str)

    # ⑤ 多層ネットワーク構造図
    plot_multilayer_structure(ml, plot_dir, task_key, phase, speed_str)

    # ──────────────────────────────────────────────────────────
    # Step 6: 物理層高度解析（C-1/C-2/C-3）
    # ──────────────────────────────────────────────────────────
    print(f"\n[物理層高度解析]")
    adv_phys_dir = out_sub / "advanced_physical"
    adv_phys_dir.mkdir(parents=True, exist_ok=True)

    bt_result = compute_betweenness(graph, phase, speed_str)
    fl_result = compute_flow_analysis(graph, phase, speed_str)
    cm_result = compute_community(graph, phase, speed_str)
    save_flow_results(bt_result, fl_result, cm_result, adv_phys_dir)

    plot_betweenness_timeseries(bt_result, plot_dir, task_key, phase, speed_str)
    plot_flow_diagram(graph, fl_result, bt_result, plot_dir, task_key, phase, speed_str)
    plot_community_transition(cm_result, plot_dir, task_key, phase, speed_str)

    print(f"  ボトルネックNode: {bt_result.bottleneck_node_id[31]} (@MSt)")
    print(f"  コミュニティ数: {int(cm_result.labels_mode.max())+1}")

    # ──────────────────────────────────────────────────────────
    # Step 7: 筋シナジー解析（D-1）
    # ──────────────────────────────────────────────────────────
    print(f"\n[筋シナジー解析]")
    adv_emg_dir = out_sub / "advanced_emg"
    adv_emg_dir.mkdir(parents=True, exist_ok=True)

    syn_result = compute_synergy(emg_path, task_key, phase, speed_str)
    save_synergy_results(syn_result, adv_emg_dir)
    summary['synergy_n']   = syn_result.n_synergies
    summary['synergy_vaf'] = round(syn_result.vaf_final, 4)

    # ──────────────────────────────────────────────────────────
    # Step 8: トレードオフ解析（E-1）
    # ──────────────────────────────────────────────────────────
    print(f"\n[トレードオフ解析]")
    adv_ml_dir = out_sub / "advanced_multilayer"
    adv_ml_dir.mkdir(parents=True, exist_ok=True)

    to_result = compute_tradeoff(er_phys, em_metrics, cr_emg.corr_sliding)
    save_tradeoff_results(to_result, adv_ml_dir)
    summary['tradeoff_r']          = round(to_result.pearson_r, 4)
    summary['tradeoff_p']          = round(to_result.pearson_p, 4)
    summary['tradeoff_is_tradeoff']= to_result.is_tradeoff
    summary['burden_shift_stance'] = round(to_result.burden_shift_stance, 4)

    return summary


# =============================================================================
# バッチ実行
# =============================================================================

def run_batch(
    task_keys  : list[str],
    phases     : list[int],
    speeds     : list[str],
    result_dir : str,
    output_dir : Path,
) -> None:
    """全タスク × 全フェーズ × 全速度を一括実行し、比較レポートを生成する。"""
    all_summaries: list[dict] = []

    for tk in task_keys:
        for ph in phases:
            for spd in speeds:
                try:
                    s = run_single(tk, ph, spd, result_dir, output_dir)
                    all_summaries.append(s)
                except Exception as e:
                    print(f"[エラー] {tk} Phase{ph} {spd}m/s: {e}")

    # 全結果を一覧 CSV に保存
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        out_csv = output_dir / "all_summaries.csv"
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\n[完了] 一覧CSV: {out_csv}")

        # Markdown レポート
        _write_markdown_report(all_summaries, output_dir)

        # ── 比較グラフ（--all 実行時のみ生成） ──────────────────
        compare_dir = output_dir / "comparison_plots"
        compare_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[比較グラフ] 生成中: {compare_dir}")

        # 速度×タスク折れ線グラフ
        plot_speed_task_comparison(all_summaries, compare_dir)

        # 対側性効果サマリー（task01 vs task03）
        plot_contralateral_effect(all_summaries, compare_dir)

        # 統計テーブル（t検定 + Cohen's d）
        compute_stats_table(all_summaries, compare_dir)

        # フェーズ・速度ごとのレーダー＋ヒートマップ＋Degree比較
        # （代表的な条件：Phase3 / 1.1m/s のみ生成）
        rep_phase, rep_speed = 3, '1.1'
        rep_sums = {s['task_key']: s for s in all_summaries
                    if s['phase'] == rep_phase and s['speed'] == rep_speed}
        if len(rep_sums) >= 2:
            plot_radar_comparison(rep_sums, compare_dir, rep_phase, rep_speed)

        # ── 高度解析の比較グラフ ──────────────────────────────
        # シナジー比較（代表条件）
        syn_by_task = {}
        for s in all_summaries:
            if s['phase'] == rep_phase and s['speed'] == rep_speed:
                tk = s['task_key']
                # synergy結果はサマリーに含まれないため再実行（軽量）
                try:
                    from emg_network.synergy import compute_synergy
                    from emg_network.correlation import build_emg_csv_path
                    emg_p = build_emg_csv_path(tk, rep_phase, base_dir=str(CFG.BASE_DIR))
                    syn_by_task[tk] = compute_synergy(emg_p, tk, rep_phase, rep_speed)
                except Exception:
                    pass
        if len(syn_by_task) >= 1:
            plot_synergy_heatmap(syn_by_task, compare_dir, rep_phase, rep_speed)
            plot_synergy_comparison_bar(syn_by_task, compare_dir, rep_phase, rep_speed)

        # トレードオフ比較（代表条件）
        try:
            from futto_network.build_graph import FuttoGraph, build_tension_csv_path
            from futto_network.efficiency import compute_efficiency
            from emg_network.network_metrics import compute_emg_network_metrics
            from emg_network.correlation import compute_emg_correlation, build_emg_csv_path
            from multilayer_network.tradeoff import compute_tradeoff, plot_tradeoff_timeseries

            to_by_task = {}
            for s in all_summaries:
                if s['phase'] == rep_phase and s['speed'] == rep_speed:
                    tk  = s['task_key']
                    g   = FuttoGraph(tk)
                    t_path = build_tension_csv_path(tk, rep_phase, rep_speed, result_dir)
                    g.load_from_tension_csv(t_path)
                    er  = compute_efficiency(g, rep_phase, rep_speed)
                    emg_p = build_emg_csv_path(tk, rep_phase, base_dir=str(CFG.BASE_DIR))
                    cre = compute_emg_correlation(emg_p, tk, rep_phase, rep_speed)
                    em  = compute_emg_network_metrics(cre)
                    to_by_task[tk] = compute_tradeoff(er, em, cre.corr_sliding)
            if to_by_task:
                plot_tradeoff_timeseries(to_by_task, compare_dir, rep_phase, rep_speed)
        except Exception as e:
            print(f"  [警告] トレードオフ比較グラフ生成中にエラー: {e}")


def _write_markdown_report(summaries: list[dict], output_dir: Path) -> None:
    lines = ["# Futto ネットワーク解析 — 総合レポート\n"]
    lines.append("| Task | Phase | Speed | N | E_norm | λ_norm | Hub_H | E_emg | IL | GES |")
    lines.append("|------|-------|-------|---|--------|--------|-------|-------|----|-----|")

    for s in summaries:
        lines.append(
            f"| {s['task_key']} | {s['phase']} | {s['speed']} "
            f"| {s['N_futto_nodes']} "
            f"| {s['E_norm_mean']:.4f} "
            f"| {s['lambda_max_norm_mean']:.4f} "
            f"| {s['hub_entropy_bits']:.3f} "
            f"| {s['emg_global_efficiency']:.4f} "
            f"| {s['joint_il_proxy']:.4f} "
            f"| {s['gait_efficiency_score']:.4f} |"
        )

    # 対側性効果サマリー（task01 vs task03）
    t01 = [s for s in summaries if s['task_key'] == 'task01']
    t03 = [s for s in summaries if s['task_key'] == 'task03']
    if t01 and t03:
        lines.append("\n## 対側性効果サマリー (task01 vs task03)\n")
        lines.append("| Phase | Speed | ΔGES | ΔE_emg_inter | Δcross/ipsi |")
        lines.append("|-------|-------|------|--------------|-------------|")
        for s1 in t01:
            match = [s for s in t03 if s['phase'] == s1['phase'] and s['speed'] == s1['speed']]
            if match:
                s3 = match[0]
                lines.append(
                    f"| {s1['phase']} | {s1['speed']} "
                    f"| {s1['gait_efficiency_score'] - s3['gait_efficiency_score']:+.4f} "
                    f"| {s1['emg_inter_efficiency'] - s3['emg_inter_efficiency']:+.4f} "
                    f"| {s1['emg_cross_ipsi_ratio'] - s3['emg_cross_ipsi_ratio']:+.4f} |"
                )

    report_path = output_dir / "network_report.md"
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"[レポート] {report_path}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    result_dir = args.result_dir

    # 速度→文字列の正規化
    speed_map = {v['name'].replace('m/s', ''): v['name'].replace('m/s', '')
                 for v in CFG.PHASES.values()}
    all_speeds = list(speed_map.keys())   # ['0.7', '0.9', '1.1', '1.3', '1.5']
    all_phases = list(CFG.PHASES.keys())  # [1, 2, 3, 4, 5]
    all_tasks  = ['task01', 'task02']     # task03 は器具なし（張力データなし）

    if args.all:
        print("全タスク × 全フェーズ × 全速度を一括実行します。")
        run_batch(all_tasks, all_phases, all_speeds, result_dir, output_dir)
    elif args.task and args.phase:
        speed_str = args.speed.replace('m/s', '') if args.speed else CFG.PHASES[args.phase]['name'].replace('m/s', '')
        run_single(args.task, args.phase, speed_str, result_dir, output_dir)
    else:
        # 対話モード
        print("=========================================")
        print("  Futto ネットワーク解析")
        print("=========================================")

        task = input("タスク [task01/task02]: ").strip().lower()
        if task not in all_tasks:
            print(f"エラー: {task} は無効です。")
            return

        phase_s = input("フェーズ番号 [1-5]: ").strip()
        if not phase_s.isdigit() or int(phase_s) not in all_phases:
            print("エラー: 無効なフェーズです。")
            return
        phase = int(phase_s)

        # CONFIGのPHASESから速度を自動設定
        speed_s = CFG.PHASES[phase]['name'].replace('m/s', '')
        print(f"速度: {speed_s}m/s (フェーズ{phase}から自動設定)")

        run_single(task, phase, speed_s, result_dir, output_dir)


if __name__ == '__main__':
    main()
