# =============================================================================
# phase_analysis/cli/main.py
#
# 役割:
#   Futto ゴムの位相遅れ解析の対話メニュー。
#
# 実行方法:
#   cd C:\FuttoAnalysis\2026_analysis  (または任意のプロジェクトルート)
#   python -m phase_analysis.cli.main
#
# 解析モード:
#   1: 関節角度 vs ゴム張力      (股関節を中心とした Mechanical phase lag)
#   2: EMG vs ゴム張力           (神経適応の指標)
#   3: タスク間比較              (task01/02/03 の位相遅れをまとめて比較)
#   4: 床反力 vs ゴム張力         (LabChart 実測データとの位相関係)

#クロスコリレーション
#横軸が「ゴム張力を何%ずらしたか（ラグ）」、縦軸が「そのときの2信号の一致度（相関係数、-1〜1）」です。
#見る場所   読み取れること

# 山の頂点の位置（横軸）
# 位相遅れの大きさ。
# この例では+11.9%＝154ms。プラスなら「張力が角度より遅れる」、
# マイナスなら「張力が角度より先行する」
# 
# 山の頂点の高さ（縦軸）
# 連動の強さ。
# 1に近いほど「同じ形の波形が時間差だけずれている」、
# 0に近いと無相関、
# -1に近いと「逆向きの波形」
# 
# 山の幅・尖り方
# 信号の周波数特性。
# 尖った山は特定の遅れ量でしか一致しない敏感な関係、
# なだらかな山は遅れがあいまいでも一致度がゆるやかに保たれる関係
# 
# 軸ラベルの [positive = ... lags ...]
# 符号の意味を毎回確認できるようにしている注記
# 
# この例では山が鋭く高さもほぼ1なので、
# 「ゴム張力は股関節角度とほぼ同じ波形を保ったまま、
# 約154ms（歩行周期の12%）遅れて追随している」と読めます
# 1つの数値で全体の遅れを要約したいときに向いています。
#
#CRP（3段構成）
#**上段（ラップ版, -180〜180°）**
# 瞬間ごとの位相差です。
# 0°線（緑）に近いほど同時に変化、
# ±180°線（赤）に近いほど逆向きに変化していることを示します。
# この例では歩行周期12%付近で鋭く跳ね上がっているのが見えます。

#**中段（アンラップ版）**
# その跳ね上がりが「見かけ上のジャンプ」か「本物の変化」かを判定するためのものです。
# この例ではアンラップ後も同じ場所で急激に立ち上がっており（-100°→300°付近まで一気に変化）、
# ジャンプ検出フラグは False（±180°境界を跨いでいない）ですが、
# 実際に位相関係が急変した瞬間があることが分かります。
# 歩行周期0-10%あたりは緩やかな関係だったのが、10-15%あたりで一気に変化し、以降は安定しています。
#
# **下段左右（位相平面）**
# 各信号がどう周期運動しているかを表します。
# 左の股関節角度はきれいな円軌道＝滑らかな単振動です。
# 右のゴム張力は歪んだギザギザの軌道＝ノイズの影響が大きいことを示します。
# この歪みが大きい箇所（原点付近を頻繁に横切る場所）ほど、上段のCRPが不安定になりやすい場所です。
# =============================================================================

import sys
from pathlib import Path
 
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = 'phase_analysis.cli'
 
import numpy as np
import matplotlib.pyplot as plt
 
from ..config.phase_config import (
    SUBJECTS, TASKS, TASK_TITLES, PHASES,
    HIP_SEGMENTS, KNEE_SEGMENTS, ANKLE_SEGMENTS, ALL_SEGMENTS,
    get_phase_result_dir, DEFAULT_T_CYCLE_SEC,
)
from ..dataio.tension_loader import (
    get_tension_series, get_hip_tension_sum, list_available_segments,
)
from ..dataio.angle_loader import get_angle_series
from ..dataio.emg_loader import get_emg_series, MUSCLE_NAMES, DEFAULT_HIP_MUSCLE
from ..dataio.labchart_loader import (
    get_grf_phase_average_series,
)
from ..viz.phase_plot import (
    plot_signals_overlay, plot_cross_correlation,
    plot_crp_analysis, plot_task_lag_comparison,
    plot_grf_phase_average,
)
from ..core.cross_correlation import find_phase_lag
 
 
# ---------------------------------------------------------------------------
# 共通ヘルパー
# ---------------------------------------------------------------------------
 
def _select_from_list(prompt: str, options: list, default_idx: int = 0) -> str:
    """番号選択式の入力ヘルパー。"""
    print(f"\n{prompt}")
    for i, opt in enumerate(options):
        marker = ' (推奨)' if i == default_idx else ''
        print(f"  {i+1}: {opt}{marker}")
    while True:
        raw = input(f"番号を選択 (Enter で {default_idx+1}): ").strip()
        if raw == '':
            return options[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("有効な番号を入力してください。")
 
 
def _select_phase() -> int:
    print("\n--- フェーズ (歩行速度) を選択 ---")
    for num, info in PHASES.items():
        print(f"  {num}: {info['speed']} m/s")
    while True:
        try:
            n = int(input("フェーズ番号 (1-5): "))
            if n in PHASES:
                return n
        except ValueError:
            pass
        print("1〜5 を入力してください。")
 
 
def _select_task() -> str:
    return _select_from_list(
        "--- タスクを選択 ---",
        [f"{t} ({TASK_TITLES[t]})" for t in TASKS],
        default_idx=0
    ).split(' ')[0]
 
 
def _get_hip_angle(subject, task, phase_num, speed):
    """股関節角度を取得する共通ヘルパー。"""
    result = get_angle_series(task, phase_num, speed, joint='hip')
    if result is None:
        print("  -> 股関節角度データの取得に失敗しました。")
        print("     mechanics_analysis でメニュー5 (関節角度) を実行してください。")
    return result
 
 
def _resample_to_common_grid(cyc_a, sig_a, cyc_b, sig_b, n_points=101):
    """2つの時系列を共通の gait_cycle_% グリッドに揃える。"""
    common_cycles = np.linspace(0, 100, n_points)
    a = np.interp(common_cycles, cyc_a, sig_a)
    b = np.interp(common_cycles, cyc_b, sig_b)
    return common_cycles, a, b


def get_tension_sum_for_segments(task: str, phase_num: int, speed: str, segments_to_sum: list[str]):
    """指定されたセグメントリストの張力を合計して返す。"""
    sum_tension = None
    n_summed = 0
    available_segs = list_available_segments(task, phase_num, speed)
    if not available_segs:
        print(f"  -> [{task}] 張力データが利用できません。")
        return None, None

    target_segs = [s for s in segments_to_sum if s in available_segs]

    for seg in target_segs:
        result = get_tension_series(task, phase_num, speed, seg)
        if result:
            cyc, tension = result
            if sum_tension is None:
                sum_tension = np.zeros_like(tension)
                common_cycles = cyc
            # 念のためリサンプリング
            sum_tension += np.interp(common_cycles, cyc, tension)
            n_summed += 1

    if n_summed > 0:
        return common_cycles, sum_tension
    return None, None
 
 
# ---------------------------------------------------------------------------
# モード1: 関節角度 vs ゴム張力
# ---------------------------------------------------------------------------
 
def _run_angle_vs_tension(subject: str):
    task = _select_task()
    phase_num = _select_phase()
    speed = PHASES[phase_num]['speed']
 
    print(f"\n=== {subject} / {task} / Phase{phase_num} ({speed}m/s) ===")
    print("=== 関節角度 vs ゴム張力 ===\n")
 
    joint = _select_from_list(
        "--- 関節を選択 ---", ['hip', 'knee', 'ankle'], default_idx=0
    )
 
    angle_result = get_angle_series(task, phase_num, speed, joint=joint)
    if angle_result is None:
        return
    cyc_angle, angle = angle_result
 
    # 張力: 股関節を選んだ場合は HIP_SEGMENTS の合計、それ以外は個別セグメント選択
    if joint == 'hip':
        use_sum = input(
            "\n股関節周りのゴム合計張力を使いますか？ "
            "(y: 合計 / n: セグメント個別選択) [y]: "
        ).strip().lower()
        if use_sum != 'n':
            tension_result = get_hip_tension_sum(task, phase_num, speed)
            tension_label = 'Hip Rubber Tension (sum)'
        else:
            segs = list_available_segments(task, phase_num, speed)
            seg = _select_from_list("--- セグメントを選択 ---", segs, default_idx=0)
            tension_result = get_tension_series(task, phase_num, speed, seg)
            tension_label = f'{seg} Tension'
    else:
        segs = list_available_segments(task, phase_num, speed)
        default_group = KNEE_SEGMENTS if joint == 'knee' else ANKLE_SEGMENTS
        default_seg = next((s for s in default_group if s in segs), segs[0] if segs else None)
        default_idx = segs.index(default_seg) if default_seg in segs else 0
        seg = _select_from_list("--- セグメントを選択 ---", segs, default_idx=default_idx)
        tension_result = get_tension_series(task, phase_num, speed, seg)
        tension_label = f'{seg} Tension'
 
    if tension_result is None:
        return
    cyc_tension, tension = tension_result
 
    # 共通グリッドに揃える
    cycles, angle_i, tension_i = _resample_to_common_grid(
        cyc_angle, angle, cyc_tension, tension
    )
 
    label = f'{task} Phase{phase_num} ({speed}m/s)'
    angle_label = f'{joint.capitalize()} Angle'
 
    _run_phase_analysis_menu(
        cycles, angle_i, tension_i,
        x_label=angle_label, y_label=tension_label,
        label=label, save_dir=get_phase_result_dir(subject, task, speed),
    )
 
 
# ---------------------------------------------------------------------------
# モード2: EMG vs ゴム張力
# ---------------------------------------------------------------------------
 
def _run_emg_vs_tension(subject: str):
    task = _select_task()
    phase_num = _select_phase()
    speed = PHASES[phase_num]['speed']
 
    print(f"\n=== {subject} / {task} / Phase{phase_num} ({speed}m/s) ===")
    print("=== EMG vs ゴム張力 (神経適応) ===\n")
 
    muscle = _select_from_list(
        "--- 筋を選択 ---", MUSCLE_NAMES,
        default_idx=MUSCLE_NAMES.index(DEFAULT_HIP_MUSCLE)
    )
    leg = _select_from_list("--- 脚を選択 ---", ['L', 'R'], default_idx=0)
 
    emg_result = get_emg_series(subject, task, phase_num, muscle=muscle, leg=leg)
    if emg_result is None:
        return
    cyc_emg, emg = emg_result
 
    use_sum = input(
        "\n股関節周りのゴム合計張力を使いますか？ "
        "(y: 合計 / n: セグメント個別選択) [y]: "
    ).strip().lower()
    if use_sum != 'n':
        tension_result = get_hip_tension_sum(task, phase_num, speed)
        tension_label = 'Hip Rubber Tension (sum)'
    else:
        segs = list_available_segments(task, phase_num, speed)
        seg = _select_from_list("--- セグメントを選択 ---", segs, default_idx=0)
        tension_result = get_tension_series(task, phase_num, speed, seg)
        tension_label = f'{seg} Tension'
 
    if tension_result is None:
        return
    cyc_tension, tension = tension_result
 
    cycles, emg_i, tension_i = _resample_to_common_grid(
        cyc_emg, emg, cyc_tension, tension
    )
 
    label = f'{task} Phase{phase_num} ({speed}m/s) {leg}-leg'
    emg_label = f'{leg}_{muscle} EMG'
 
    _run_phase_analysis_menu(
        cycles, emg_i, tension_i,
        x_label=emg_label, y_label=tension_label,
        label=label, save_dir=get_phase_result_dir(subject, task, speed),
    )
 
 
# ---------------------------------------------------------------------------
# 共通: クロスコリレーション / CRP のグラフメニュー
# ---------------------------------------------------------------------------
 
def _run_phase_analysis_menu(cycles, x, y, x_label, y_label, label, save_dir):
    """
    2信号が揃った後の、クロスコリレーション/CRP解析メニュー。
    """
    while True:
        print("\n--- 位相解析グラフメニュー ---")
        print("  1: 信号重ね描き (正規化)")
        print("  2: クロスコリレーション")
        print("  3: Continuous Relative Phase (CRP)")
        print("  a: 全グラフを一括表示")
        print("  s: 全グラフを一括保存")
        print("  q: 終了")
        ch = input("選択 >> ").strip().lower()
 
        if ch == 'q':
            break
        elif ch == '1':
            plot_signals_overlay(cycles, x, y, x_label=x_label, y_label=y_label,
                                 label=label)
        elif ch == '2':
            fig, result = plot_cross_correlation(
                x, y, x_label=x_label, y_label=y_label, label=label,
                base_cycle_s=DEFAULT_T_CYCLE_SEC
            )
            print(f"\n  位相遅れ: {result['lag_pct']:.2f}%"
                  f" ({result['lag_ms']:.1f} ms)"
                  f"  相関係数: {result['peak_corr']:.3f}")
        elif ch == '3':
            fig, result = plot_crp_analysis(
                cycles, x, y, x_label=x_label, y_label=y_label, label=label
            )
            print(f"\n  平均|CRP|: {result['mean_abs_crp']:.1f} deg"
                  f"  (0=同位相, 180=逆位相)")
        elif ch == 'a':
            plot_signals_overlay(cycles, x, y, x_label=x_label, y_label=y_label,
                                 label=label)
            plot_cross_correlation(x, y, x_label=x_label, y_label=y_label,
                                   label=label, base_cycle_s=DEFAULT_T_CYCLE_SEC)
            plot_crp_analysis(cycles, x, y, x_label=x_label, y_label=y_label,
                              label=label)
        elif ch == 's':
            save_dir.mkdir(parents=True, exist_ok=True)
            base = f"{x_label}_vs_{y_label}".replace(' ', '_').replace('/', '-')
            plot_signals_overlay(
                cycles, x, y, x_label=x_label, y_label=y_label, label=label,
                save_path=str(save_dir / f"{base}_overlay.png"), show_plot=False
            )
            plot_cross_correlation(
                x, y, x_label=x_label, y_label=y_label, label=label,
                base_cycle_s=DEFAULT_T_CYCLE_SEC,
                save_path=str(save_dir / f"{base}_crosscorr.png"), show_plot=False
            )
            plot_crp_analysis(
                cycles, x, y, x_label=x_label, y_label=y_label, label=label,
                save_path=str(save_dir / f"{base}_crp.png"), show_plot=False
            )
            print(f"✓ グラフを保存しました: {save_dir}")
        else:
            print("無効な入力です。")
 
 
# ---------------------------------------------------------------------------
# モード3: タスク間比較
# ---------------------------------------------------------------------------
 
def _run_task_comparison(subject: str):
    phase_num = _select_phase()
    speed = PHASES[phase_num]['speed']
 
    print(f"\n=== {subject} / Phase{phase_num} ({speed}m/s) タスク間比較 ===\n")
 
    joint = _select_from_list(
        "--- 関節を選択 ---", ['hip', 'knee', 'ankle'], default_idx=0
    )
 
    lag_results = {}
    for task in TASKS:
        angle_result = get_angle_series(task, phase_num, speed, joint=joint)
        if angle_result is None:
            print(f"  -> [{task}] 関節角度データなし。スキップします。")
            continue
        cyc_angle, angle = angle_result
 
        if joint == 'hip':
            tension_result = get_hip_tension_sum(task, phase_num, speed)
        else:
            segs = list_available_segments(task, phase_num, speed)
            default_group = KNEE_SEGMENTS if joint == 'knee' else ANKLE_SEGMENTS
            seg = next((s for s in default_group if s in segs), None)
            if seg is None:
                print(f"  -> [{task}] 対応するセグメントなし。スキップします。")
                continue
            tension_result = get_tension_series(task, phase_num, speed, seg)
 
        if tension_result is None:
            print(f"  -> [{task}] 張力データなし。スキップします。")
            continue
        cyc_tension, tension = tension_result
 
        cycles, angle_i, tension_i = _resample_to_common_grid(
            cyc_angle, angle, cyc_tension, tension
        )
        lag_results[task] = find_phase_lag(
            angle_i, tension_i, base_cycle_s=DEFAULT_T_CYCLE_SEC
        )
        print(f"  [{task}] lag = {lag_results[task]['lag_pct']:.2f}%"
              f" ({lag_results[task]['lag_ms']:.1f} ms)"
              f"  r = {lag_results[task]['peak_corr']:.3f}")
 
    if not lag_results:
        print("\n比較可能なデータがありませんでした。")
        return
 
    x_label = f'{joint.capitalize()} Angle'
    y_label = 'Hip Rubber Tension' if joint == 'hip' else f'{joint.capitalize()} Rubber Tension'
 
    ans = input("\nグラフを表示しますか？ (y/n): ").strip().lower()
    if ans == 'y':
        plot_task_lag_comparison(lag_results, x_label=x_label, y_label=y_label)
 
    ans2 = input("グラフを保存しますか？ (y/n): ").strip().lower()
    if ans2 == 'y':
        save_dir = get_phase_result_dir(subject, 'comparison', speed)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f'Phase{phase_num}_{joint}_task_comparison.png')
        plot_task_lag_comparison(lag_results, x_label=x_label, y_label=y_label,
                                 save_path=save_path, show_plot=False)
        print(f"✓ 保存しました: {save_path}")
 
 
# ---------------------------------------------------------------------------
# モード4: 床反力 vs ゴム張力
# ---------------------------------------------------------------------------
 
def _run_grf_vs_tension(subject: str):
    date = input("\n計測日を入力してください (例: 20260217): ").strip()
    task = _select_task()
    phase_num = _select_phase()
    speed = PHASES[phase_num]['speed']
 
    print(f"\n=== {subject} / {task} / Phase{phase_num} ({speed}m/s) ===")
    print("=== 床反力 (GRF) フェーズ平均 vs ゴム張力 ===\n")
 
    leg = _select_from_list("--- 床反力の脚を選択 ---", ['L', 'R'], default_idx=0)
    component = _select_from_list(
        "--- 床反力の成分を選択 ---", ['Fz', 'Fy', 'Fx'], default_idx=0
    )
 
    # タスク内で検出された全歩行周期を正規化・平均し、フェーズ代表波形を得る
    grf_avg_result = get_grf_phase_average_series(
        date, task, leg=leg, component=component, normalize_bw=True
    )
    if grf_avg_result is None:
        return
 
    cyc_grf = grf_avg_result['cycles']
    grf = grf_avg_result['mean']
    grf_label = f'{leg} GRF {component} (n={grf_avg_result["n_cycles"]} avg)'
 
    ans = input("\n平均波形 (±SD) のグラフを確認しますか？ (y/n) [n]: ").strip().lower()
    if ans == 'y':
        show_all = input(
            "  各周期を薄く重ねて表示しますか？ (y/n) [n]: "
        ).strip().lower() == 'y'
        plot_grf_phase_average(
            grf_avg_result, leg=leg, component=component,
            label=f'{task} Phase{phase_num}', show_all_cycles=show_all
        )
 
    use_sum = input(
        "\n股関節周りのゴム合計張力を使いますか？ "
        "(y: 合計 / n: セグメント個別選択) [y]: "
    ).strip().lower()
    if use_sum != 'n':
        tension_result = get_hip_tension_sum(task, phase_num, speed)
        tension_label = 'Hip Rubber Tension (sum)'
    else:
        segs = list_available_segments(task, phase_num, speed)
        seg = _select_from_list("--- セグメントを選択 ---", segs, default_idx=0)
        tension_result = get_tension_series(task, phase_num, speed, seg)
        tension_label = f'{seg} Tension'
 
    if tension_result is None:
        return
    cyc_tension, tension = tension_result
 
    cycles, grf_i, tension_i = _resample_to_common_grid(
        cyc_grf, grf, cyc_tension, tension
    )
 
    label = (f'{task} Phase{phase_num} ({speed}m/s)'
             f' n={grf_avg_result["n_cycles"]}cycles avg')
 
    _run_phase_analysis_menu(
        cycles, grf_i, tension_i,
        x_label=grf_label, y_label=tension_label,
        label=label, save_dir=get_phase_result_dir(subject, task, speed),
    )
 
 
# ---------------------------------------------------------------------------
# モード5: 全自動実行モード
# ---------------------------------------------------------------------------

def _run_fully_automatic_mode(subject: str):
    print("\n=== 全自動実行モード ===")
    print("指定された関節と、それに対応するゴム張力の関係を、全タスク・全フェーズで自動解析し、結果を保存します。")

    joint = _select_from_list(
        "--- 解析対象の関節を選択してください ---", ['hip', 'knee', 'ankle'], default_idx=0
    )

    # 関節に応じたセグメントグループを定義
    if joint == 'hip':
        segments_to_sum = HIP_SEGMENTS
        tension_label_base = 'Hip Rubber Tension (sum)'
    elif joint == 'knee':
        segments_to_sum = KNEE_SEGMENTS + ['Back_Thigh_Out', 'Back_Thigh_In']
        tension_label_base = 'Knee Rubber Tension (sum)'
    else:  # ankle
        segments_to_sum = ANKLE_SEGMENTS
        tension_label_base = 'Ankle Rubber Tension (sum)'

    print(f"\n--- 解析開始: {joint.capitalize()} Angle vs {tension_label_base} ---")
    print(f"対象セグメント: {', '.join(segments_to_sum)}")

    for task in TASKS:
        for phase_num, phase_info in PHASES.items():
            speed = phase_info['speed']
            print(f"\n>>> Processing: {subject} / {task} / Phase{phase_num} ({speed}m/s)")

            # 1. 関節角度の取得
            angle_result = get_angle_series(task, phase_num, speed, joint=joint)
            if angle_result is None:
                print(f"  -> 関節角度データなし。スキップします。")
                continue
            cyc_angle, angle = angle_result

            # 2. 合計張力の取得
            cyc_tension, tension = get_tension_sum_for_segments(task, phase_num, speed, segments_to_sum)
            if tension is None:
                print(f"  -> 張力データなし。スキップします。")
                continue

            # 3. 共通グリッドにリサンプリング
            cycles, angle_i, tension_i = _resample_to_common_grid(
                cyc_angle, angle, cyc_tension, tension
            )

            # 4. 解析と保存
            label = f'{task} Phase{phase_num} ({speed}m/s)'
            angle_label = f'{joint.capitalize()} Angle'
            save_dir = get_phase_result_dir(subject, task, speed)
            save_dir.mkdir(parents=True, exist_ok=True)

            base_filename = f"{angle_label}_vs_{tension_label_base}".replace(' ', '_').replace('/', '-')

            try:
                # 信号重ね描き
                plot_signals_overlay(
                    cycles, angle_i, tension_i, x_label=angle_label, y_label=tension_label_base, label=label,
                    save_path=str(save_dir / f"{base_filename}_overlay.png"), show_plot=False
                )
                # クロスコリレーション
                plot_cross_correlation(
                    angle_i, tension_i, x_label=angle_label, y_label=tension_label_base, label=label,
                    base_cycle_s=DEFAULT_T_CYCLE_SEC,
                    save_path=str(save_dir / f"{base_filename}_crosscorr.png"), show_plot=False
                )
                # CRP
                plot_crp_analysis(
                    cycles, angle_i, tension_i, x_label=angle_label, y_label=tension_label_base, label=label,
                    save_path=str(save_dir / f"{base_filename}_crp.png"), show_plot=False
                )
                print(f"  ✓ グラフを保存しました: {save_dir}")
            except Exception as e:
                print(f"  -> グラフ生成・保存中にエラーが発生しました: {e}")

    print("\n--- 全自動実行が完了しました ---")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
 
def main():
    print("=" * 60)
    print(" Futto ゴム位相遅れ解析 (Phase Lag Analysis)")
    print(" クロスコリレーション + Continuous Relative Phase (CRP)")
    print("=" * 60)
 
    subject = _select_from_list("--- 被験者を選択 ---", SUBJECTS, default_idx=0)
 
    print("\n--- 解析モードを選択してください ---")
    print("  1: 関節角度 vs ゴム張力       (Mechanical phase lag)")
    print("  2: EMG vs ゴム張力            (神経適応)")
    print("  3: タスク間比較               (task01/02/03 の位相遅れ比較)")
    print("  4: 床反力 vs ゴム張力          (LabChart 実測データとの位相関係)")
    print("  5: 全自動実行モード           (関節角度vs合計張力を全件解析・保存)")
    mode = input("モード (1-5): ").strip()
 
    if mode == '1':
        _run_angle_vs_tension(subject)
    elif mode == '2':
        _run_emg_vs_tension(subject)
    elif mode == '3':
        _run_task_comparison(subject)
    elif mode == '4':
        _run_grf_vs_tension(subject)
    elif mode == '5':
        _run_fully_automatic_mode(subject)
    else:
        print("無効なモードです。")
 
 
if __name__ == '__main__':
    main()
