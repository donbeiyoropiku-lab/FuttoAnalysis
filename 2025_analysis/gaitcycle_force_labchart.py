#-----------------------------------------------
#gaitcycle_force_labchart.py
#labchart解析用
#床反力をグラデーションまたは平均値でプロットする
#data_processing.pyを同じディレクトリに置いたうえで実行
#左足のみ

#-----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data_processing # あなたのdata_processing.pyをインポート
import os

# --- 設定 ---
LABCHART_FILE_PATH = r"C:\FuttoAnalysis\labchart\20260217\task03.txt"
OUTPUT_CSV_PATH = r"C:\FuttoAnalysis\labchart\20260217\task03_gait_cycles.csv"
# --- 設定 ---
# ※ data_processing.py内の列番号定義に合わせる
#labchartにつないでいるコード、生データのtxtファイルを確認してdata_processing.py適宜変更
# Rfz, Lfz = 3, 10
#Lfx_col, Lfy_col, Lfz_col = 1, 2, 3 # LabChartファイルのチャンネル1,2,3に対応
Lfx_col, Lfy_col, Lfz_col = 8, 9, 10 #20260217

def normalize_cycle(data, cycle_start, cycle_end, num_points=101):
    """歩行周期を指定されたサンプル数に正規化する"""
    cycle_data = data[cycle_start:cycle_end]
    x_new = np.linspace(0, 100, num_points)
    x_old = np.linspace(0, 100, len(cycle_data))
    normalized_data = np.interp(x_new, x_old, cycle_data)
    return normalized_data

def main():
    """メインの実行関数"""
    print("--- LabChart 床反力正規化プログラム ---")
    
    # --- データの読み込みと処理 ---
    print(f"ファイルを処理中: {LABCHART_FILE_PATH}")
    # フィルターとベースライン補正済みのデータを取得
    Lfx = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfx_col)
    Lfy = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfy_col)
    Lfz = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfz_col)
    
    # ★★★ 体重の計算を追加 ★★★
    # 最初の10秒間の立位区間(安定している1秒後から9秒間)のFzデータから平均値を計算
    standing_force = np.mean(Lfz[1000:10000]) 
    if standing_force <= 0:
        print("警告: 立位荷重が0以下です。体重の計算をスキップします。")
        body_weight = 1 # ゼロ除算を避ける
    else:
        body_weight = standing_force * 2
    
    print(f"\n推定された片足荷重 (Fz): {standing_force:.3f} V")
    print(f"推定された体重 (Body Weight): {body_weight:.3f} V")

    # ★★★ 床反力データを体重で正規化（無次元化） ★★★
    Lfx_normalized = (Lfx / body_weight) * 100
    Lfy_normalized = (Lfy / body_weight) * 100
    Lfz_normalized = (Lfz / body_weight) * 100
    
    # --- 歩行周期の検出 ---
    gait_cycles_list = data_processing.calculate_gait_cycles(Lfz) # 正規化前のFzで周期を検出
    if not gait_cycles_list:
        print("歩行周期が検出されませんでした。"); return
    gait_cycles_df = pd.DataFrame(gait_cycles_list)
    
    # ★★★ ここから追加 ★★★
    # 歩行周期データをCSVファイルに保存
    if not gait_cycles_df.empty:
        try:
            gait_cycles_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\n歩行周期データを '{OUTPUT_CSV_PATH}' に保存しました。")
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}")
    else:
        print("\n歩行周期が検出されなかったため、CSVファイルは作成されませんでした。")
        return # グラフ表示せずに終了
    # ★★★ ここまで追加 ★★★

    # --- 歩行周期のデータを正規化 ---
    # ★正規化後のデータを使って各周期を切り出す
    normalized_fx_cycles, normalized_fy_cycles, normalized_fz_cycles = [], [], []

    # gait_cycles_listではなく、DataFrameの行をループする
    for index, cycle in gait_cycles_df.iterrows():
        # ★★★ int()で整数に変換する処理を追加 ★★★
        start, end = int(cycle['hs_frame']), int(cycle['to_frame'])
        normalized_fx_cycles.append(normalize_cycle(Lfx_normalized, start, end))
        normalized_fy_cycles.append(normalize_cycle(Lfy_normalized, start, end))
        normalized_fz_cycles.append(normalize_cycle(Lfz_normalized, start, end))
    
    # --- ユーザーに表示スタイルを選択させる ---
    while True:
        try:
            style = int(input("\nプロットスタイルを選択 (1: グラデーション, 2: 平均±標準偏差): "))
            if style in [1, 2]: break
            else: print("1か2を入力してください。")
        except ValueError: print("数値を入力してください。")

    # --- グラフの描画 ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Normalized Ground Reaction Force (Left Foot)', fontsize=16)
    
    force_data = {
        'Fz (Vertical)': (axes[0], normalized_fz_cycles, 'Blues'),
        'Fy (Anterior/Posterior)': (axes[1], normalized_fy_cycles, 'Greens'),
        #'Fx (Medial/Lateral)': (axes[2], normalized_fx_cycles, 'Reds')
    }
    
    x_axis = np.linspace(0, 100, 101)

    for label, (ax, data, cmap_name) in force_data.items():
        if not data:
            ax.text(0.5, 0.5, 'No cycles detected', ha='center', va='center'); continue

        if style == 1: # グラデーション表示
            cmap = cm.get_cmap(cmap_name, len(data))
            for i, cycle_data in enumerate(data):
                ax.plot(x_axis, cycle_data, color=cmap(i / len(data)), alpha=0.6)
            ax.set_title(f'{label} - All Gait Cycles (Gradient)')

        elif style == 2: # 平均±標準偏差 表示
            mean_curve = np.mean(data, axis=0)
            std_curve = np.std(data, axis=0)
            ax.plot(x_axis, mean_curve, color='black', linewidth=2, label='Mean')
            ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3, label='Std. Dev.')
            ax.set_title(f'{label} - Mean +/- Std. Dev.')
            ax.legend()
        
        ax.set_ylabel('Force (%BW)') # ★単位を%BWに変更
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5) # 0のライン
    
    axes[-1].set_xlabel('Gait Cycle (%)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()

'''
#-----------------------------------------------
# gaitcycle_force_labchart.py
# LabChart解析用
# 右足・左足それぞれの床反力から歩行周期を算出し、個別のCSVに出力する
#-----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import data_processing # data_processing.pyをインポート
import os

# --- 設定 ---
LABCHART_FILE_PATH = r"C:\FuttoAnalysis\labchart\20251027\task2.txt"
OUTPUT_DIR = r"C:\FuttoAnalysis\labchart\20251027" # CSV保存先のディレクトリ

file_name_right = "task2_gait_cycles_right.csv"
file_name_left = "task2_gait_cycles_left.csv"

# --- 列番号の定義 ---
# ご指定の通り、LabChartのチャンネル設定に合わせる
# Right Foot (右足)
Rfx_col, Rfy_col, Rfz_col = 1, 2, 3
# Left Foot (左足)
Lfx_col, Lfy_col, Lfz_col = 8, 9, 10

def normalize_cycle(data, cycle_start, cycle_end, num_points=101):
    """歩行周期を指定されたサンプル数に正規化する"""
    cycle_data = data[cycle_start:cycle_end]
    x_new = np.linspace(0, 100, num_points)
    x_old = np.linspace(0, 100, len(cycle_data))
    normalized_data = np.interp(x_new, x_old, cycle_data)
    return normalized_data

def process_leg(leg_name, fx_col, fy_col, fz_col, output_filename):
    """
    片足分のデータを処理し、CSV保存と正規化データを返す関数
    """
    print(f"\n========== {leg_name} Foot の処理を開始 ==========")
    
    # --- データの読み込み ---
    # data_processing.adjusted_data を使用してフィルタ・補正済みデータを取得
    try:
        Fx = data_processing.adjusted_data(LABCHART_FILE_PATH, fx_col)
        Fy = data_processing.adjusted_data(LABCHART_FILE_PATH, fy_col)
        Fz = data_processing.adjusted_data(LABCHART_FILE_PATH, fz_col)
    except Exception as e:
        print(f"エラー: データの読み込みに失敗しました ({leg_name}) - {e}")
        return None, None

    # --- 体重の推定 ---
    # 最初の10秒間の立位区間(安定している1秒後から9秒間)のFzデータから平均値を計算
    standing_force = np.mean(Fz[1000:10000]) 
    if standing_force <= 0:
        print(f"警告: {leg_name}の立位荷重が0以下です。体重の計算を1.0で代用します。")
        body_weight = 1.0
    else:
        body_weight = standing_force * 2 # 片足荷重 * 2
    
    print(f"推定された片足荷重 (Fz): {standing_force:.3f} V")
    print(f"推定された体重 (Body Weight): {body_weight:.3f} V")

    # --- 床反力データを体重で正規化（無次元化） ---
    Fx_norm = (Fx / body_weight) * 100
    Fy_norm = (Fy / body_weight) * 100
    Fz_norm = (Fz / body_weight) * 100
    
    # --- 歩行周期の検出 ---
    gait_cycles_list = data_processing.calculate_gait_cycles(Fz) # Fzで周期を検出
    
    if not gait_cycles_list:
        print(f"{leg_name}: 歩行周期が検出されませんでした。")
        return None, None

    # --- CSVファイルに保存 ---
    gait_cycles_df = pd.DataFrame(gait_cycles_list)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        gait_cycles_df.to_csv(output_path, index=False)
        print(f"★ 保存成功: '{output_path}'")
    except Exception as e:
        print(f"エラー: CSVファイルの保存に失敗しました - {e}")

    # --- グラフ用に正規化された周期データを抽出 ---
    normalized_cycles = {'Fx': [], 'Fy': [], 'Fz': []}

    for index, cycle in gait_cycles_df.iterrows():
        start, end = int(cycle['hs_frame']), int(cycle['to_frame'])
        normalized_cycles['Fx'].append(normalize_cycle(Fx_norm, start, end))
        normalized_cycles['Fy'].append(normalize_cycle(Fy_norm, start, end))
        normalized_cycles['Fz'].append(normalize_cycle(Fz_norm, start, end))
    
    return normalized_cycles, gait_cycles_df

def main():
    """メインの実行関数"""
    print("--- LabChart 床反力正規化プログラム (左右対応版) ---")
    print(f"ファイルを処理中: {LABCHART_FILE_PATH}")
    
    # --- 右足 (Right) の処理 ---
    right_cycles, right_df = process_leg(
        leg_name="Right",
        fx_col=Rfx_col, fy_col=Rfy_col, fz_col=Rfz_col,
        output_filename=
    )

    # --- 左足 (Left) の処理 ---
    left_cycles, left_df = process_leg(
        leg_name="Left",
        fx_col=Lfx_col, fy_col=Lfy_col, fz_col=Lfz_col,
        output_filename= file_name_left
    )

    # --- ユーザーに表示する足を選択させる ---
    plot_data = None
    leg_label = ""

    while True:
        print("\n------------------------------------------------")
        choice = input("グラフを表示する足を選択してください (r: 右足, l: 左足, q: 終了): ").lower()
        
        if choice == 'r':
            if right_cycles:
                plot_data = right_cycles
                leg_label = "Right Foot"
                break
            else:
                print("右足のデータがありません。")
        elif choice == 'l':
            if left_cycles:
                plot_data = left_cycles
                leg_label = "Left Foot"
                break
            else:
                print("左足のデータがありません。")
        elif choice == 'q':
            print("終了します。")
            return
        else:
            print("r, l, q のいずれかを入力してください。")

    # --- ユーザーに表示スタイルを選択させる ---
    while True:
        try:
            style = int(input("プロットスタイルを選択 (1: グラデーション, 2: 平均±標準偏差): "))
            if style in [1, 2]: break
            else: print("1か2を入力してください。")
        except ValueError: print("数値を入力してください。")

    # --- グラフの描画 ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Normalized Ground Reaction Force ({leg_label})', fontsize=16)
    
    # FzとFyを描画対象とする（Fxも必要なら追加可能）
    force_data = {
        'Fz (Vertical)': (axes[0], plot_data['Fz'], 'Blues'),
        'Fy (Anterior/Posterior)': (axes[1], plot_data['Fy'], 'Greens'),
        # 'Fx (Medial/Lateral)': (axes[2], plot_data['Fx'], 'Reds') 
    }
    
    x_axis = np.linspace(0, 100, 101)

    for label, (ax, data, cmap_name) in force_data.items():
        if not data:
            ax.text(0.5, 0.5, 'No cycles detected', ha='center', va='center'); continue

        if style == 1: # グラデーション表示
            cmap = cm.get_cmap(cmap_name, len(data))
            for i, cycle_data in enumerate(data):
                ax.plot(x_axis, cycle_data, color=cmap(i / len(data)), alpha=0.6)
            ax.set_title(f'{label} - All Gait Cycles (Gradient)')

        elif style == 2: # 平均±標準偏差 表示
            mean_curve = np.mean(data, axis=0)
            std_curve = np.std(data, axis=0)
            ax.plot(x_axis, mean_curve, color='black', linewidth=2, label='Mean')
            ax.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3, label='Std. Dev.')
            ax.set_title(f'{label} - Mean +/- Std. Dev.')
            ax.legend()
        
        ax.set_ylabel('Force (%BW)') 
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5) # 0のライン
    
    axes[-1].set_xlabel('Gait Cycle (%)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()
'''