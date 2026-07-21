"""
mechanics_analysis/debug_torque_3d.py
======================================
特定の歩行周期（%）における特定のゴムのトルク計算を
詳細な数値と3Dベクトルで可視化してデバッグするためのツール。

実行方法:
  cd C:\FuttoAnalysis\2026_analysis\mechanics_analysis
  python debug_torque_3d.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from futto_common import CONFIG as config
from mechanics_analysis.io_loader import build_analysis_paths, load_opti_and_tension
from mechanics_analysis.physics_core import calc_joint_center, normalize_tension_df

def main():
    print("=" * 60)
    print(" Futto トルク計算 3D可視化デバッガ")
    print("=" * 60)

    # --- 1. 条件の対話入力（デフォルト値あり） ---
    task_key = input("タスク名 [task02]: ").strip() or "task02"
    phase_str = input("フェーズ番号 (1-5) [3]: ").strip() or "3"
    phase = int(phase_str)
    
    target_line = input("確認したいゴム名 [Front_Upper_In]: ").strip() or "Front_Upper_In"
    cycle_str = input("確認したい歩行周期(%) [10]: ").strip() or "10"
    cycle_pct = float(cycle_str)

    cfg = config.TASK_CONFIGS.get(task_key)
    if not cfg:
        print(f"エラー: {task_key} はCONFIGに存在しません。")
        return

    speed = config.PHASES[phase]['name'].replace('m/s', '')
    paths = build_analysis_paths(cfg, task_key, phase, speed, config.RESULT_DIR)
    
    print(f"\nデータを読み込んでいます... ({task_key} / Phase{phase})")
    df_mean, df_tension = load_opti_and_tension(paths['opti_csv'], paths['tension_csv'])
    if df_mean is None or df_tension is None:
        return

    df_tension = normalize_tension_df(df_tension)

    # --- 2. データの抽出と計算 ---
    grouped = df_mean.groupby(df_mean['gait_cycle_%'].apply(lambda x: round(x, 5)))
    
    # 指定された%に最も近いフレームを抽出
    available_pcts = np.array(list(grouped.groups.keys()))
    nearest_pct = available_pcts[np.argmin(np.abs(available_pcts - cycle_pct))]
    frame_df = grouped.get_group(nearest_pct)

    positions = {
        int(row['id']): np.array([row['x'], row['y'], row['z']], dtype=float)
        for _, row in frame_df.iterrows()
    }

    # 対象ゴムの端点ID
    lines_def = cfg.get('LINES_TO_DRAW', {})
    if target_line not in lines_def:
        print(f"エラー: ゴム '{target_line}' が LINES_TO_DRAW に定義されていません。")
        return
    p1_id, p2_id = lines_def[target_line]

    # 張力
    try:
        ten_pivot = df_tension.pivot(index='gait_cycle_%', columns='segment', values='tension_N')
        nearest_ten_pct = ten_pivot.index[np.argmin(np.abs(ten_pivot.index - cycle_pct))]
        tension_N = float(ten_pivot.loc[nearest_ten_pct, target_line])
    except KeyError:
        print(f"エラー: 張力データに '{target_line}' が見つかりません。")
        return

    # 関節中心
    torque_line_joints = cfg.get('TORQUE_LINE_JOINTS', {})
    assigned_joint = torque_line_joints.get(target_line, 'Hip')
    joint_def = cfg.get('JOINT_CENTER_DEFS', {}).get(assigned_joint)
    
    jc = calc_joint_center(joint_def, positions) if joint_def else np.zeros(3)

    # physics_core.py と完全に同じ力学計算
    p1_pos = positions.get(p1_id)
    p2_pos = positions.get(p2_id)
    
    if p1_pos[2] < p2_pos[2]:
        p_attach, p_origin = p1_pos, p2_pos
        attach_id, origin_id = p1_id, p2_id
    else:
        p_attach, p_origin = p2_pos, p1_pos
        attach_id, origin_id = p2_id, p1_id
        
    vec = p_origin - p_attach
    norm = np.linalg.norm(vec)
    F_vec = (vec / norm) * tension_N
    r_vec = p_attach - jc
    tau_Nmm = np.cross(r_vec, F_vec)
    tau_Nm = tau_Nmm / 1000.0

    # --- 3. デバッグ数値のコンソール出力 ---
    print(f"\n{'='*50}")
    print(f" [分析結果] 歩行周期: {nearest_pct}% | ゴム: {target_line}")
    print(f"{'='*50}")
    print(f"対象関節 (Joint)      : {assigned_joint}")
    print(f"関節中心座標 (Hip JC) : X={jc[0]:.1f}, Y={jc[1]:.1f}, Z={jc[2]:.1f}")
    print(f"着力点 (遠位:{attach_id})   : X={p_attach[0]:.1f}, Y={p_attach[1]:.1f}, Z={p_attach[2]:.1f}")
    print(f"引く方向 (近位:{origin_id}) : X={p_origin[0]:.1f}, Y={p_origin[1]:.1f}, Z={p_origin[2]:.1f}")
    print("-" * 50)
    print(f"張力 (Tension)        : {tension_N:.2f} N")
    print(f"張力ベクトル (F_vec)  : X={F_vec[0]:.2f}, Y={F_vec[1]:.2f}, Z={F_vec[2]:.2f}")
    print(f"ﾓｰﾒﾝﾄｱｰﾑ (r_vec)      : X={r_vec[0]:.1f}, Y={r_vec[1]:.1f}, Z={r_vec[2]:.1f}")
    print("-" * 50)
    print(f"算出トルク (3D)       : X={tau_Nm[0]:.3f}, Y={tau_Nm[1]:.3f}, Z={tau_Nm[2]:.3f} Nm")
    
    # Y軸成分の解釈
    if assigned_joint == 'Hip':
        is_flexion = tau_Nm[1] < 0
        direction_str = "屈曲方向 (Flexion / 前方へ上げる)" if is_flexion else "伸展方向 (Extension / 後方へ引く)"
    elif assigned_joint == 'Knee':
        is_flexion = tau_Nm[1] > 0
        direction_str = "屈曲方向 (Flexion / 後方へ曲げる)" if is_flexion else "伸展方向 (Extension / 前方へ伸ばす)"
    elif assigned_joint == 'Ankle':
        is_flexion = tau_Nm[1] < 0
        direction_str = "背屈方向 (Dorsiflexion / つま先上げ)" if is_flexion else "底屈方向 (Plantarflexion / つま先下げ)"
    else:
        direction_str = "正値なら+Y軸回転"
        
    print(f"★ 矢状面トルク(Y軸)  : {tau_Nm[1]:.3f} Nm  => 【 {direction_str} 】")
    print("=" * 50)

    # --- 4. 3Dプロット可視化 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 関節中心
    ax.scatter(*jc, color='blue', s=100, label=f'{assigned_joint} Center')
    # ゴムの端点
    ax.scatter(*p_attach, color='black', s=50, label=f'Distal ({attach_id})')
    ax.scatter(*p_origin, color='gray', s=50, label=f'Proximal ({origin_id})')
    
    # ゴムのライン
    ax.plot([p_attach[0], p_origin[0]], [p_attach[1], p_origin[1]], [p_attach[2], p_origin[2]], 
            color='black', linestyle='--', label='Rubber Line')

    # モーメントアーム (r) 緑色
    ax.quiver(*jc, *r_vec, color='green', linewidth=2, arrow_length_ratio=0.1, label='Moment Arm (r)')
    
    # 力ベクトル (F) 赤色 (視認性のため10倍にスケール)
    F_scale = 10.0
    ax.quiver(*p_attach, *(F_vec * F_scale), color='red', linewidth=3, arrow_length_ratio=0.2, 
              label=f'Tension Vector (F x{F_scale})')
              
    # トルクベクトル (Tau) 金色 (視認性のため20000倍にスケール)
    T_scale = 20
    ax.quiver(*jc, *(tau_Nm * T_scale), color='gold', linewidth=4, arrow_length_ratio=0.2, 
              label=f'Torque Vector (Tau x{T_scale})')

    ax.set_title(f"Torque Debug 3D View\n{target_line} @ {nearest_pct}% (Task: {task_key})", fontsize=14)
    ax.set_xlabel('X (Progression: +fwd)')
    ax.set_ylabel('Y (Lateral: R->L+)')
    ax.set_zlabel('Z (Vertical: +up)')
    
    # 軸のスケールを1:1:1に揃える
    pts = np.array([jc, p_attach, p_origin, p_attach + F_vec*F_scale, jc + tau_Nm*T_scale])
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), 
                          pts[:,1].max()-pts[:,1].min(), 
                          pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) * 0.5
    mid_y = (pts[:,1].max()+pts[:,1].min()) * 0.5
    mid_z = (pts[:,2].max()+pts[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()