# =============================================================================
# mechanics_analysis/main.py
#
# 役割:
#   対話メニューでタスク・フェーズ・解析モードを選び、
#   各解析モジュールを呼び出す。
#
# 実行方法:
#   cd C:\FuttoAnalysis\2026_analysis\mechanics_analysis
#   python main.py
#
# 解析メニュー:
#   a: 全解析を一括実行 (CSV + グラフをすべて自動保存)
#   1: 関節トルク解析 + スティック図
#   2: ゴム仕事量 / エネルギー解析
#   3: ワークループ (剛性・ヒステリシス)
#   4: ポーラーチャート (下腿合力ベクトル)
#   5: 関節角度時系列
#   6: 3D 力場アニメーション
#   f: 力場3Dマップ (等値面 + ベクトル場)
#   c: タスク間トルク比較 (別タスクを追加ロード)
#   q: 終了
#
# タスク・フェーズ選択時:
#   'all' を入力すると全タスク/全フェーズを一括実行
# =============================================================================

import sys
import pandas as pd
from pathlib import Path

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'mechanics_analysis'

from futto_common import CONFIG as config
from .io_loader    import build_analysis_paths, load_opti_and_tension, save_csv
from .physics_core import (
    unify_coordinate_system,
    calc_all_torques,
    calc_work_data,
    calc_net_shank_force,
    calc_joint_angles,
    calc_joint_power,
    calc_frame_physics,
)
from .force_field_map import show_force_field_map
from .visualizer import (
    plot_joint_torques,
    plot_work_time_series,
    plot_work_loops,
    plot_polar_force,
    plot_joint_angles,
    plot_joint_power,
    plot_task_comparison,
    animate_force_field_3d,
)

WORK_LOOP_TARGETS = [
    'Front_Upper_In', 'Back_Upper_In',
    'Front_Knee_Upper_In', 'Back_Thigh_In',
    'Front_Shin', 'Back_Shin_In',
]


# ---------------------------------------------------------------------------
# タスク・フェーズ共通の選択・ロード処理
# ---------------------------------------------------------------------------

def _select_task() -> tuple:
    """タスクを対話選択し (task_key, cfg) を返す。
    'all' 入力で全タスク一括実行モード → ('all', None) を返す。"""
    available = [t for t in config.TASKS if t in config.TASK_CONFIGS]
    print(f"利用可能なタスク: {', '.join(available)} (全タスク・全フェーズ一括実行は 'all' を入力)")
    while True:
        task_key = input("タスク名を入力してください: ").strip().lower()
        if task_key == 'all':
            return 'all', None
        if task_key in config.TASK_CONFIGS:
            return task_key, config.TASK_CONFIGS[task_key]
        print(f"  -> '{task_key}' は CONFIG に登録されていません。")


def _select_phase() -> tuple:
    """フェーズを対話選択し (phase, speed) を返す。
    'all' 入力で全フェーズ一括実行モード → ('all', None) を返す。"""
    phases = getattr(config, 'PHASES', {})
    print("\n利用可能なフェーズ:")
    for k, v in phases.items():
        print(f"  {k}: {v['name']}")
    print("  all: 全フェーズを一括実行")
    while True:
        p = input("フェーズ番号 (1-5) または 'all': ").strip().lower()
        if p == 'all':
            return 'all', None
        if p.isdigit() and int(p) in phases:
            phase = int(p)
            speed = phases[phase]['name'].replace('m/s', '')
            return phase, speed
        print("無効な入力です。")


def _load_data(task_key, cfg, phase, speed):
    """パス生成・データロード・座標統一をまとめて行う。"""
    paths = build_analysis_paths(
        cfg        = cfg,
        task_key   = task_key,
        phase      = phase,
        speed      = speed,
        result_dir = getattr(config, 'RESULT_DIR', r"C:\FuttoAnalysis\result"),
    )
    df_mean, df_tension = load_opti_and_tension(
        paths['opti_csv'], paths['tension_csv']
    )
    if df_mean is None:
        return None, None, None, paths
    if df_tension is None:
        df_tension = pd.DataFrame(columns=['gait_cycle_%', 'segment', 'tension_N'])

    df_unified = unify_coordinate_system(df_mean, task_key)
    return df_unified, df_tension, paths, paths


# ---------------------------------------------------------------------------
# 全解析一括実行
# ---------------------------------------------------------------------------

def run_all(df_unified, df_tension, cfg, task_key, paths, speed=''):
    """
    1〜6 + 力場マップの全解析を順番に実行し、CSV とグラフをすべて自動保存する。
    ひとつの解析が失敗しても残りを続ける。

    Parameters
    ----------
    speed : str
        フェーズの速度文字列 (例: '0.7')。力場マップのファイル名に使用。
    """
    seg_groups = getattr(config, 'SEGMENT_GROUPS', {})
    save_dir   = paths['graph_dir']
    frames     = None   # フレームデータ (6と力場マップで共用)

    print("\n" + "=" * 60)
    print(f" 全解析一括実行: {task_key} / {Path(paths['graph_dir']).parent.name}m/s")
    print(f" 保存先: {save_dir}")
    print("=" * 60)

    # ---- 1: 関節トルク ----
    print("\n[1/6] 関節トルク解析...")
    try:
        torque_df = calc_all_torques(df_unified, df_tension, cfg)
        if torque_df is not None and not torque_df.empty:
            save_csv(torque_df, paths['torque_out'], "トルクデータ")
            plot_joint_torques(torque_df, cfg, task_key, df_unified, save_dir)
            print("  -> 完了")
        else:
            print("  -> スキップ (データなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 2: 仕事量 ----
    print("\n[2/6] ゴム仕事量 / エネルギー解析...")
    try:
        df_summary, df_instant, df_cumulative, df_energy, T_cycle = calc_work_data(
            df_unified, df_tension, cfg
        )
        if df_summary is not None:
            save_csv(df_summary, paths['work_out'], "仕事量サマリ")
            print("\n--- ゴムの総仕事量 (1周期あたり) ---")
            print(df_summary.to_string(float_format="%.3f"))
            if df_instant is not None:
                plot_work_time_series(
                    df_instant, df_cumulative, task_key, seg_groups, T_cycle, save_dir,
                    ylim_power=(-5, 5),      # Y軸範囲を固定
                    ylim_work=(-0.65, 0.65)      # Y軸範囲を固定
                )
            # ③ 弾性エネルギーデータを力場マップ用に保存
            if df_energy is not None:
                energy_out = paths['work_out'].replace('_work.csv', '_elastic_energy.csv')
                save_csv(df_energy, energy_out, "弾性エネルギーデータ")
            print("  -> 完了")
        else:
            print("  -> スキップ (データなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 3: ワークループ ----
    print("\n[3/6] ワークループ (剛性・ヒステリシス)...")
    try:
        plot_work_loops(
            df_unified, df_tension, cfg, task_key,
            targets=WORK_LOOP_TARGETS, save_dir=save_dir
        )
        print("  -> 完了")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 4: ポーラーチャート ----
    print("\n[4/6] ポーラーチャート (下腿合力ベクトル)...")
    try:
        net_force, cycles = calc_net_shank_force(df_unified, df_tension, cfg)
        if net_force is not None:
            plot_polar_force(net_force, cycles, task_key, save_dir)
            print("  -> 完了")
        else:
            print("  -> スキップ (Shank関連ゴムなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 5: 関節角度 ----
    print("\n[5/6] 関節角度時系列...")
    try:
        df_angles = calc_joint_angles(df_unified, cfg)
        if df_angles is not None:
            angle_out = paths['torque_out'].replace('_torque.csv', '_joint_angles.csv')
            save_csv(df_angles, angle_out, "関節角度データ")
            plot_joint_angles(df_angles, task_key, save_dir)
            print("  -> 完了")
        else:
            print("  -> スキップ (データなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 6: 3D アニメーション ----
    print("\n[6/6] 3D 力場アニメーション (最も時間がかかります)...")
    try:
        frames = calc_frame_physics(df_unified, df_tension, cfg)
        if frames:
            animate_force_field_3d(frames, cfg, task_key, save_dir)
            print("  -> 完了")
        else:
            print("  -> スキップ (フレームデータなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 7: 関節パワー ----
    print("\n[7/+] 関節パワー解析...")
    try:
        torque_df_pw = calc_all_torques(df_unified, df_tension, cfg)
        df_angles_pw = calc_joint_angles(df_unified, cfg)
        df_power     = calc_joint_power(torque_df_pw, df_angles_pw, cfg, speed=speed)
        if df_power is not None:
            power_out = paths['torque_out'].replace('_torque.csv', '_joint_power.csv')
            save_csv(df_power, power_out, "関節パワーデータ")
            plot_joint_power(df_power, torque_df_pw, df_angles_pw,
                             task_key, speed=speed, save_dir=save_dir)
            print("  -> 完了")
        else:
            print("  -> スキップ")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 力場3Dマップ ----
    print("\n[追加] 力場3Dマップ...")
    try:
        if frames:
            show_force_field_map(
                frames, cfg, task_key,
                phase=0, speed=speed,
                target_pcts=[0, 25, 50, 75],
                save_dir=save_dir,
            )
            print("  -> 完了")
        else:
            print("  -> スキップ (フレームデータなし)")
    except Exception as e:
        print(f"  -> エラー: {e}")

    # ---- 完了サマリー ----
    print("\n" + "=" * 60)
    print(f" 全解析完了。出力先: {save_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Futto 力学特性解析プログラム (2026年実験対応版)")
    print("=" * 60)

    task_key, cfg = _select_task()

    # ---- 全タスク・全フェーズ 一括実行モード ----
    if task_key == 'all':
        available_tasks = [t for t in config.TASKS if t in config.TASK_CONFIGS]
        phases = getattr(config, 'PHASES', {})
        for tk in available_tasks:
            t_cfg = config.TASK_CONFIGS[tk]
            for ph, p_info in phases.items():
                speed = p_info['name'].replace('m/s', '')
                print(f"\n{'='*50}")
                print(f" 一括処理: {tk} / Phase{ph} ({speed}m/s)")
                print(f"{'='*50}")
                df_unified, df_tension, paths, _ = _load_data(tk, t_cfg, ph, speed)
                if df_unified is not None:
                    run_all(df_unified, df_tension, t_cfg, tk, paths, speed=speed)
                else:
                    print("  -> データが見つからないためスキップします。")
        print("\nすべてのタスク・フェーズの一括処理が完了しました。")
        return

    phase, speed = _select_phase()

    # ---- 指定タスク・全フェーズ 一括実行モード ----
    if phase == 'all':
        phases = getattr(config, 'PHASES', {})
        for ph, p_info in phases.items():
            speed_str = p_info['name'].replace('m/s', '')
            print(f"\n{'='*50}")
            print(f" 一括処理: {task_key} / Phase{ph} ({speed_str}m/s)")
            print(f"{'='*50}")
            df_unified, df_tension, paths, _ = _load_data(task_key, cfg, ph, speed_str)
            if df_unified is not None:
                run_all(df_unified, df_tension, cfg, task_key, paths, speed=speed_str)
            else:
                print("  -> データが見つからないためスキップします。")
        print(f"\n{task_key} の全フェーズの一括処理が完了しました。")
        return

    # ---- 通常の対話モード ----
    print(f"\n--- {task_key} / Phase{phase} ({speed}m/s) のデータを読み込み中 ---")
    df_unified, df_tension, paths, _ = _load_data(task_key, cfg, phase, speed)
    if df_unified is None:
        print("データ読み込みに失敗しました。終了します。")
        return

    seg_groups = getattr(config, 'SEGMENT_GROUPS', {})
    save_dir   = paths['graph_dir']
    frames     = None   # フレームデータキャッシュ (6 と f で共用)

    while True:
        print("\n--- 解析メニュー ---")
        menu = {
            'a': '全解析を一括実行 (CSV + グラフをすべて自動保存)',
            '1': '関節トルク解析 + スティック図',
            '2': 'ゴム仕事量 / エネルギー解析',
            '3': 'ワークループ (剛性・ヒステリシス)',
            '4': 'ポーラーチャート (下腿合力ベクトル)',
            '5': '関節角度時系列',
            '6': '3D 力場アニメーション',
            '7': '関節パワー解析 (P = tau_y x omega_y)',
            'f': '力場3Dマップ (等値面 + ベクトル場)',
            'c': 'タスク間トルク比較 (別タスクを追加ロード)',
            'q': '終了',
        }
        for k, v in menu.items():
            print(f"  {k}: {v}")
        mode = input("モードを選択 >> ").strip().lower()

        # ---- a: 全解析一括 ----
        if mode == 'a':
            run_all(df_unified, df_tension, cfg, task_key, paths, speed=speed)

        # ---- 1: 関節トルク ----
        elif mode == '1':
            print("\n関節トルクを計算中...")
            torque_df = calc_all_torques(df_unified, df_tension, cfg)
            if torque_df is not None and not torque_df.empty:
                save_csv(torque_df, paths['torque_out'], "トルクデータ")
                plot_joint_torques(torque_df, cfg, task_key, df_unified, save_dir)
            else:
                print("トルク計算結果が空でした。")

        # ---- 2: 仕事量 ----
        elif mode == '2':
            print("\n仕事量を計算中...")
            df_summary, df_instant, df_cumulative, df_energy, T_cycle = calc_work_data(
                df_unified, df_tension, cfg
            )
            if df_summary is not None:
                save_csv(df_summary, paths['work_out'], "仕事量サマリ")
                print("\n--- ゴムの総仕事量 (1周期あたり) ---")
                print(df_summary.to_string(float_format="%.3f"))
                if df_instant is not None:
                    plot_work_time_series(
                        df_instant, df_cumulative, task_key, seg_groups, T_cycle, save_dir,
                        ylim_power=(-5, 5),      # Y軸範囲を固定
                        ylim_work=(-0.65, 0.65)      # Y軸範囲を固定
                    )
                # ③ 弾性エネルギーデータを力場マップ用に保存
                if df_energy is not None:
                    energy_out = paths['work_out'].replace('_work.csv', '_elastic_energy.csv')
                    save_csv(df_energy, energy_out, "弾性エネルギーデータ")

        # ---- 3: ワークループ ----
        elif mode == '3':
            print("\nワークループを描画中...")
            plot_work_loops(df_unified, df_tension, cfg, task_key,
                            targets=WORK_LOOP_TARGETS, save_dir=save_dir)

        # ---- 4: ポーラーチャート ----
        elif mode == '4':
            print("\n下腿合力ベクトルを計算中...")
            net_force, cycles = calc_net_shank_force(df_unified, df_tension, cfg)
            if net_force is not None:
                plot_polar_force(net_force, cycles, task_key, save_dir)

        # ---- 5: 関節角度 ----
        elif mode == '5':
            print("\n関節角度を計算中...")
            df_angles = calc_joint_angles(df_unified, cfg)
            plot_joint_angles(df_angles, task_key, save_dir)

        # ---- 6: 3D アニメーション ----
        elif mode == '6':
            print("\n3D力場データを計算中 (時間がかかる場合があります)...")
            frames = calc_frame_physics(df_unified, df_tension, cfg)
            if frames:
                animate_force_field_3d(frames, cfg, task_key, save_dir)

        # ---- 7: 関節パワー ----
        elif mode == '7':
            print("\n関節パワーを計算中...")
            torque_df_pw = calc_all_torques(df_unified, df_tension, cfg)
            df_angles_pw = calc_joint_angles(df_unified, cfg)
            df_power     = calc_joint_power(torque_df_pw, df_angles_pw, cfg, speed=speed)
            if df_power is not None:
                power_out = paths['torque_out'].replace('_torque.csv', '_joint_power.csv')
                save_csv(df_power, power_out, "関節パワーデータ")
                plot_joint_power(df_power, torque_df_pw, df_angles_pw,
                                 task_key, speed=speed, save_dir=save_dir)
            else:
                print("パワーデータが計算できませんでした。")

        # ---- f: 力場3Dマップ ----
        elif mode == 'f':
            print("\n力場3Dマップを計算中...")
            if frames is None:
                print("  -> フレームデータを計算中... (初回のみ時間がかかります)")
                frames = calc_frame_physics(df_unified, df_tension, cfg)
            if frames:
                pct_input = input(
                    "表示する歩行周期 (%) をカンマ区切りで入力 [Enter で 0,25,50,75]: "
                ).strip()
                target_pcts = (
                    [float(p) for p in pct_input.split(',')]
                    if pct_input else [0, 25, 50, 75]
                )
                sigma_input = input(
                    "ガウス拡散の広がり σ [mm] (Enter で 60mm): "
                ).strip()
                sigma = float(sigma_input) if sigma_input else 60.0
                show_force_field_map(
                    frames, cfg, task_key,
                    phase=phase, speed=speed,
                    target_pcts=target_pcts,
                    sigma=sigma,
                    save_dir=save_dir,
                )

        # ---- c: タスク間比較 ----
        elif mode == 'c':
            print("\n比較用の追加タスクをロードします。")
            torque_data_dict = {}

            t0 = calc_all_torques(df_unified, df_tension, cfg)
            if t0 is not None and not t0.empty:
                torque_data_dict[task_key] = t0

            for _ in range(2):
                ans = input("追加タスクを選択しますか？ (y/n): ").strip().lower()
                if ans != 'y':
                    break
                extra_task, extra_cfg = _select_task()
                extra_phase, extra_speed = _select_phase()
                df_extra, df_ten_extra, paths_extra, _ = _load_data(
                    extra_task, extra_cfg, extra_phase, extra_speed
                )
                if df_extra is not None:
                    t_extra = calc_all_torques(df_extra, df_ten_extra, extra_cfg)
                    if t_extra is not None and not t_extra.empty:
                        torque_data_dict[extra_task] = t_extra

            if len(torque_data_dict) >= 2:
                joint_name = input("比較する関節 (Hip/Knee/Ankle): ").strip().capitalize()
                plot_task_comparison(torque_data_dict, joint_name,
                                     save_dir=save_dir)
            else:
                print("比較に必要なデータが揃いませんでした。")

        elif mode == 'q':
            print("終了します。")
            break
        else:
            print("無効な入力です。")


if __name__ == '__main__':
    main()