#-----------------------------------------------
#labchart解析用
#指定した秒数の区間の床反力(Fx,Fy,Fz)を可視化するプログラム
#data_processing.pyを同じディレクトリに置いたうえで実行
#左足のみ
#-----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import data_processing # あなたのdata_processing.pyをインポート

# --- ▼▼▼ 設定 ▼▼▼ ---
# 解析したいLabChartのテキストファイルのパス
LABCHART_FILE_PATH = r"C:\FuttoAnalysis\labchart\20250731\task1.txt"

# 使用するデータ系列の列番号 (data_processing.pyに合わせる)
#labchartにつないでいるコード、生データのtxtファイルを確認してdata_processing.py適宜変更
# Rfz_col, Lfz_col = 3, 10
Lfx_col, Lfy_col, Lfz_col =  1, 2 ,3# Ch1, Ch2, Ch3

# --- ▲▲▲ 設定はここまで ▲▲▲ ---

def main():
    """メインの実行関数"""
    print("--- LabChart 区間データ可視化プログラム ---")

    # --- データの読み込みと処理 ---
    print(f"ファイルを処理中: {LABCHART_FILE_PATH}")
    Lfx = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfx_col)
    Lfy = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfy_col)
    Lfz = data_processing.adjusted_data(LABCHART_FILE_PATH, Lfz_col)
    
    # --- ユーザーに表示区間を入力させる ---
    while True:
        try:
            start_time = float(input("\n表示を開始する時刻（秒）を入力してください (例: 10): "))
            end_time = float(input("表示を終了する時刻（秒）を入力してください (例: 50): "))
            if start_time < end_time and start_time >= 0 and end_time * 1000 <= len(Lfz):
                break
            else:
                print("エラー: 無効な時間範囲です。データは0秒から約60秒の間にあります。")
        except ValueError:
            print("エラー: 数値を入力してください。")

    # --- グラフの描画 ---
    print("グラフを作成中...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(f'Ground Reaction Force (Left Foot) from {start_time}s to {end_time}s', fontsize=16)
    
    # サンプリング周波数1000Hzと仮定
    time_axis = np.arange(len(Lfz)) / 1000.0
    
    force_data = {
        'Fz (Vertical)': (axes[0], Lfz, 'blue'),
        'Fy (Anterior/Posterior)': (axes[1], Lfy, 'green'),
        'Fx (Medial/Lateral)': (axes[2], Lfx, 'red')
    }
    
    for label, (ax, data, color) in force_data.items():
        ax.plot(time_axis, data, color=color)
        ax.set_ylabel('Force (V)')
        ax.grid(True)
        ax.set_title(label)
        # 指定された時間範囲に表示を限定
        ax.set_xlim(start_time, end_time)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()