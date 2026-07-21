# =============================================================================
# calculate_rubber_work.py (v2.1 - SEGMENT_GROUPS 参照先修正)
#
# 目的:
#   平均化マーカーデータとゴム張力データから、
#   各ゴムが1歩行周期あたりに行う仕事(エネルギー)を計算・可視化する。
#
# 修正点 (v2.1):
# - plot_work_time_series 関数が、タスク固有の 'cfg' ではなく、
#   グローバルな 'config.SEGMENT_GROUPS' を参照するように修正。
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config # 設定ファイルをインポート

def compute_t_cycle(labchart_path: str) -> float | None:
    """LabChartの歩行周期CSVから平均歩行周期時間 T_cycle [s] を計算する"""
    if not labchart_path or not os.path.exists(labchart_path):
        print(f"警告: LabChartファイルが見つかりません: {labchart_path}")
        return None
    try:
        df_lc = pd.read_csv(labchart_path)
        t_cycle = (df_lc['next_hs_time'] - df_lc['hs_time']).mean()
        print(f"T_cycle = {t_cycle:.4f} s (LabChart: {labchart_path})")
        return float(t_cycle)
    except Exception as e:
        print(f"警告: T_cycle の計算エラー: {e}")
        return None


def load_data(cfg: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """平均化マーカーデータと張力データを読み込む"""
    mean_csv_path_ranged = cfg.get('MEAN_CYCLE_RANGED_OUTPUT_PATH')
    mean_csv_path_all = cfg.get('MEAN_CYCLE_OUTPUT_PATH')
    mean_csv_path = None
    
    if mean_csv_path_ranged and os.path.exists(mean_csv_path_ranged):
        mean_csv_path = mean_csv_path_ranged
    elif mean_csv_path_all and os.path.exists(mean_csv_path_all):
        mean_csv_path = mean_csv_path_all
    else:
        print(f"エラー: 平均化データファイルが見つかりません。")
        if mean_csv_path_ranged: print(f"  (試行1: {mean_csv_path_ranged})")
        if mean_csv_path_all: print(f"  (試行2: {mean_csv_path_all})")
        return None, None
        
    tension_csv_path = cfg.get('TENSION_DATA_OUTPUT_PATH')
    if not tension_csv_path or not os.path.exists(tension_csv_path):
        print(f"エラー: 張力データファイルが見つかりません: {tension_csv_path}")
        return None, None

    try:
        df_mean_cycle = pd.read_csv(mean_csv_path)
        print(f"平均化データを読み込みました: {mean_csv_path}")
    except Exception as e:
        print(f"平均化データの読み込みエラー: {e}"); return None, None
        
    try:
        df_tension = pd.read_csv(tension_csv_path)
        print(f"張力データを読み込みました: {tension_csv_path}")
    except Exception as e:
        print(f"張力データの読み込みエラー: {e}"); return None, None

    return df_mean_cycle, df_tension

def calculate_work_data(df_mean_cycle: pd.DataFrame, tension_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    各ゴムセグメントの「総仕事量」と「時系列の仕事量」を計算する。
    """
    print("ゴムの仕事量（エネルギー）の計算を開始...")
    
    lines_to_draw_def = cfg.get('LINES_TO_DRAW', {})
    if not lines_to_draw_def:
        print("エラー: config に 'LINES_TO_DRAW' が未定義です。")
        return None, None, None
        
    try:
        tension_pivot = tension_df.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
    except Exception as e:
         print(f"張力データのピボット失敗: {e}"); return None, None, None

    # マーカーデータをピボット (インデックス: %, カラム: [id_x, id_y, id_z], 値: 座標)
    marker_pivot = df_mean_cycle.pivot(index='gait_cycle_%', columns='id', values=['x', 'y', 'z'])
    
    work_summary_results = [] # {segment, positive_work_J, ...}
    
    cycle_perc_steps = sorted(marker_pivot.index.unique())
    if len(cycle_perc_steps) < 2:
        print("エラー: 平均化データのステップが少なすぎます。")
        return None, None, None
    time_index_n_minus_1 = cycle_perc_steps[:-1]
    
    df_instant_work = pd.DataFrame(index=time_index_n_minus_1)
    df_cumulative_work = pd.DataFrame(index=time_index_n_minus_1)

    for segment_name, (p1, p2) in lines_to_draw_def.items():
        try:
            # マーカー座標を取得 ( (N, 3) 配列)
            coords1 = marker_pivot[('x', p1)], marker_pivot[('y', p1)], marker_pivot[('z', p1)]
            coords2 = marker_pivot[('x', p2)], marker_pivot[('y', p2)], marker_pivot[('z', p2)]
            p1_coords = np.vstack(coords1).T
            p2_coords = np.vstack(coords2).T
            
            segment_tension = tension_pivot[segment_name].values
            current_lengths_m = np.linalg.norm(p1_coords - p2_coords, axis=1) / 1000.0
            dL = np.diff(current_lengths_m)
            F_avg = (segment_tension[:-1] + segment_tension[1:]) / 2.0
            dW = F_avg * dL
            
            # --- 1. 総仕事量の計算 (v1) ---
            positive_work_J = -np.sum(dW[dL < 0]) # ゴムが縮む(アシスト)
            negative_work_J = np.sum(dW[dL > 0]) # ゴムが伸びる(吸収)
            net_work_J = positive_work_J - negative_work_J
            
            work_summary_results.append({
                "segment": segment_name,
                "positive_work_J": positive_work_J, "negative_work_J": negative_work_J, "net_work_J": net_work_J
            })

            # --- 2. 時系列データの計算 (v2) ---
            df_instant_work[segment_name] = dW
            # dL < 0 (ポジティブワーク) の時に dW が負になるため、-dW を積算する
            df_cumulative_work[segment_name] = np.cumsum(-dW)

        except KeyError:
            print(f"警告: '{segment_name}' の計算スキップ。マーカーID {p1} または {p2} が .csv に見つかりません。")
            continue
        except Exception as e:
            print(f"警告: '{segment_name}' の計算中にエラー: {e}")
            continue

    if not work_summary_results:
        print("エラー: 仕事量の計算結果が0件です。")
        return None, None, None
        
    print("ゴムの仕事量（エネルギー）計算完了。")
    df_summary = pd.DataFrame(work_summary_results)
    
    return df_summary, df_instant_work, df_cumulative_work

# --- ▼▼▼【関数修正】▼▼▼ ---
def calculate_global_axis_bounds(margin_ratio: float = 0.05) -> dict:
    """
    全タスクのデータを読み込み、セグメントグループごとの軸範囲(最小・最大)を算出する。
    instantaneous はワット変換済みの値 [W]、cumulative は [J] で求める。
    タスク間でグラフの軸スケールを揃えて比較しやすくするために使用する。
    """
    seg_groups = config.SEGMENT_GROUPS
    raw_bounds = {name: {'instant': [np.inf, -np.inf], 'cumulative': [np.inf, -np.inf]} for name in seg_groups}

    for other_task_key, other_cfg in config.TASK_CONFIGS.items():
        df_mean_cycle, df_tension = load_data(other_cfg)
        if df_mean_cycle is None or df_tension is None:
            continue
        _, df_instant, df_cumulative = calculate_work_data(df_mean_cycle, df_tension, other_cfg)
        if df_instant is None or df_cumulative is None:
            continue

        # このタスクの T_cycle でワット変換
        t_cycle = compute_t_cycle(other_cfg.get('LABCHART_CYCLES_PATH', ''))
        if t_cycle and t_cycle > 0:
            df_instant_w = df_instant * (100.0 / t_cycle)
        else:
            df_instant_w = df_instant  # 変換できない場合はそのまま

        for group_name, segment_list in seg_groups.items():
            inst_cols = [s for s in segment_list if s in df_instant_w.columns]
            cum_cols = [s for s in segment_list if s in df_cumulative.columns]
            if inst_cols:
                raw_bounds[group_name]['instant'][0] = min(raw_bounds[group_name]['instant'][0], df_instant_w[inst_cols].min().min())
                raw_bounds[group_name]['instant'][1] = max(raw_bounds[group_name]['instant'][1], df_instant_w[inst_cols].max().max())
            if cum_cols:
                raw_bounds[group_name]['cumulative'][0] = min(raw_bounds[group_name]['cumulative'][0], df_cumulative[cum_cols].min().min())
                raw_bounds[group_name]['cumulative'][1] = max(raw_bounds[group_name]['cumulative'][1], df_cumulative[cum_cols].max().max())

    # マージンを付与し、データが無いグループは None にする
    axis_bounds = {}
    for group_name, vals in raw_bounds.items():
        axis_bounds[group_name] = {}
        for key in ('instant', 'cumulative'):
            lo, hi = vals[key]
            if np.isfinite(lo) and np.isfinite(hi):
                span = hi - lo
                margin = span * margin_ratio if span > 0 else (abs(hi) * margin_ratio + 1e-6)
                axis_bounds[group_name][key] = (lo - margin, hi + margin)
            else:
                axis_bounds[group_name][key] = None
    return axis_bounds


def plot_work_time_series(df_instant: pd.DataFrame, df_cumulative: pd.DataFrame, task_key: str,
                          axis_bounds: dict | None = None, t_cycle: float | None = None):
    """瞬時パワー(W変換済み)と累積仕事量の時系列グラフを描画する"""
    print("仕事量の時系列グラフを生成中...")

    # ★ cfg.get(...) から config.SEGMENT_GROUPS に修正
    seg_groups = config.SEGMENT_GROUPS
    if not seg_groups:
        print("警告: config.py に 'SEGMENT_GROUPS' が未定義。グラフ作成をスキップします。")
        return

    # dW [J] → P [W]: P = dW × (100 / T_cycle)
    if t_cycle and t_cycle > 0:
        df_power = df_instant * (100.0 / t_cycle)
    else:
        df_power = df_instant

    # タスクごとの保存先: C:\FuttoAnalysis\result\2025\{task}\work
    output_dir = os.path.join(config.RESULT_DIR, "2025", task_key, "work")

    save_choice = input("\n時系列グラフを画像として保存しますか？ (y/n): ").lower()
    save_plots = (save_choice == 'y')
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"画像は {output_dir} に保存されます。")

    for group_name, segment_list in seg_groups.items():
        segments_to_plot_inst = [s for s in segment_list if s in df_power.columns]
        segments_to_plot_cum = [s for s in segment_list if s in df_cumulative.columns]

        if not segments_to_plot_inst and not segments_to_plot_cum:
            continue

        # sharex=True は使わず、両軸それぞれに目盛りを表示する
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Work Analysis (Time Series) - Group: {group_name}\nTask: {task_key}', fontsize=16)

        # --- 1. 瞬時パワー (P = dW × 100/T_cycle) のグラフ ---
        if segments_to_plot_inst:
            df_inst_group = df_power[segments_to_plot_inst]
            df_inst_group.plot(ax=ax1, linewidth=1.5)
            x_axis = df_inst_group.index
            for segment in segments_to_plot_inst:
                y_values = df_inst_group[segment]
                # P > 0 (伸びる/ネガティブ) を赤
                ax1.fill_between(x_axis, y_values, 0, where=(y_values > 0),
                                 facecolor='red', alpha=0.2, interpolate=True,
                                 label='Negative (Absorb)' if segment == segments_to_plot_inst[0] else "")
                # P < 0 (縮む/ポジティブ) を緑
                ax1.fill_between(x_axis, y_values, 0, where=(y_values < 0),
                                 facecolor='green', alpha=0.2, interpolate=True,
                                 label='Positive (Release)' if segment == segments_to_plot_inst[0] else "")

        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_title('Instantaneous Power')
        ax1.set_ylabel('Power [W]')
        ax1.set_xlabel('Gait Cycle [%]')
        ax1.set_xlim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        # タスク間で比較しやすいよう、軸範囲を全タスク共通のものに固定
        if axis_bounds and axis_bounds.get(group_name, {}).get('instant'):
            ax1.set_ylim(*axis_bounds[group_name]['instant'])

        # --- 2. 累積仕事量 (cumsum(-dW)) のグラフ ---
        if segments_to_plot_cum:
            df_cum_group = df_cumulative[segments_to_plot_cum]
            df_cum_group.plot(ax=ax2, linewidth=2.0)

        ax2.set_title('Cumulative Work (Energy Release/Absorption)')
        ax2.set_ylabel('Cumulative Energy [J]\n(Positive = Net Release)')
        ax2.set_xlabel('Gait Cycle [%]')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.set_xlim(0, 100)
        # タスク間で比較しやすいよう、軸範囲を全タスク共通のものに固定
        if axis_bounds and axis_bounds.get(group_name, {}).get('cumulative'):
            ax2.set_ylim(*axis_bounds[group_name]['cumulative'])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plots:
            try:
                save_filename = f"{task_key}_{group_name}_work_timeseries.png"
                save_path = os.path.join(output_dir, save_filename)
                fig.savefig(save_path, dpi=150)
                print(f"  -> グラフ '{group_name}' を保存しました。")
            except Exception as e:
                print(f"  -> グラフ '{group_name}' の保存エラー: {e}")
        
        plt.show()

    print("時系列グラフの生成完了。")
# --- ▲▲▲ 関数修正ここまで ▲▲▲ ---

def main():
    """メイン実行関数"""
    while True:
        task_key = input("解析するタスク名を入力してください (task1, task2, or task3): ").lower()
        if task_key in config.TASK_CONFIGS:
            cfg = config.TASK_CONFIGS.get(task_key); break
        else: print(f"エラー: 設定ファイル (config.py) に '{task_key}' が見つかりません。")
    if cfg is None: print(f"エラー: {task_key} の設定読み込み失敗。"); return

    print(f"\n--- {task_key} のゴム仕事量（エネルギー）解析を開始します ---")
    
    df_mean_cycle, df_tension = load_data(cfg)
    if df_mean_cycle is None or df_tension is None:
        print("エラー: データ読み込みに失敗しました。処理を終了します。"); return

    df_summary, df_instant, df_cumulative = calculate_work_data(df_mean_cycle, df_tension, cfg)
    
    if df_summary is not None and not df_summary.empty:
        print("\n--- ゴムの総仕事量 (1周期あたり) ---")
        print(df_summary.to_string(float_format="%.3f"))
        
        total_positive = df_summary['positive_work_J'].sum()
        total_negative = df_summary['negative_work_J'].sum()
        total_net = df_summary['net_work_J'].sum()
        
        print("\n--- 合計仕事量 ---")
        print(f"  総ポジティブワーク (放出エネルギー): {total_positive:.3f} J")
        print(f"  総ネガティブワーク (吸収エネルギー): {total_negative:.3f} J")
        print(f"  正味仕事量 (ネットワーク):       {total_net:.3f} J")
        
        if total_net > 0: print("  -> (器具全体として、1周期あたり {:.3f} J のエネルギーを放出しています)".format(total_net))
        else: print("  -> (器具全体として、1周期あたり {:.3f} J のエネルギーを吸収しています)".format(abs(total_net)))

        try:
            output_path_base = cfg.get('TENSION_DATA_OUTPUT_PATH', f"{task_key}_work_data.csv")
            output_path = output_path_base.replace('_tension_data.csv', '_rubber_work_summary.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_summary.to_csv(output_path, index=False, float_format='%.6f')
            print(f"\nゴム「総」仕事量データを保存しました: {output_path}")
        except Exception as e:
            print(f"仕事量CSVの保存エラー: {e}")

        if df_instant is not None and df_cumulative is not None:
            t_cycle = compute_t_cycle(cfg.get('LABCHART_CYCLES_PATH', ''))
            print("\nタスク間でグラフの軸を統一するため、全タスクのデータから軸範囲を算出しています...")
            axis_bounds = calculate_global_axis_bounds()
            plot_work_time_series(df_instant, df_cumulative, task_key, axis_bounds, t_cycle)
        else:
            print("時系列仕事量データの計算に失敗しました。")
    else:
        print("仕事量の計算に失敗したか、結果が空でした。")

    print(f"\n--- {task_key} の解析終了 ---")

if __name__ == "__main__":
    main()