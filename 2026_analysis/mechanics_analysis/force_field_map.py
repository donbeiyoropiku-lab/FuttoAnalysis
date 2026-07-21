# =============================================================================
# mechanics_analysis/force_field_map.py
#
# 役割:
#   Futtoゴムが下肢周囲の空間に作る「力場」を3Dマップとして可視化する。
#
# 等値面の定義 (弾性エネルギー場):
#   各セグメントの弾性エネルギー u_i をマーカー位置を中心に
#   ガウス関数で空間に拡散させたスカラー場 U(P) を定義する。
#
#       U(P) = Σ_i  u_i × exp(-|P - p_i|² / (2σ²))
#
#   u_i = T_i × ε_i / 2  [J]   (T=張力[N], ε=ひずみ[m])
#   σ   : 力場の広がりパラメータ [mm] (デフォルト 60mm)
#
#   → 重力場のポテンシャル等値面に対応する表現。
#     u_i は常に正なので力の相殺が起きず、ゴムが存在する場所が
#     必ず等値面として現れる。
#
# ベクトル場の定義 (張力ベクトル場):
#   合力ベクトル F(P) = Σ_i  F_i × exp(-|P - p_i|² / (2σ²)) [N]
#   → 各点での力の向きを矢印で表示する。
#
# 表示内容:
#   ① 弾性エネルギー U(P) の等値面 (isosurface)  ← 今回変更
#      外側: 薄青 → 中層: 橙 → 内側: 赤 の4層
#      「この面より内側 = このエネルギー以上が蓄積されている領域」
#   ② 張力ベクトル場の矢印 (quiver)
#      各点での合力の向きと大きさ (plasma カラーマップ)
#   ③ マーカー・骨格・ゴム線 (参照用)
#      ゴム線: 張力の大きさで黄→赤着色
#
# 背景色の切り替え:
#   DARK_BACKGROUND = True  で黒背景 (デフォルト)
#   DARK_BACKGROUND = False で白背景
#   (下記 ★ の箇所をコメントアウトで切り替える)
#
# 実行方法:
#   mechanics_analysis/main.py のメニュー 'f' から呼び出す。
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("警告: scikit-image がインストールされていません。等値面は表示されません。")
    print("  pip install scikit-image でインストールしてください。")

from .physics_core import calc_joint_center, normalize_tension_df


# ---------------------------------------------------------------------------
# ★ 背景色設定: True=黒背景 / False=白背景 (どちらかをコメントアウト)
# ---------------------------------------------------------------------------
DARK_BACKGROUND = False    # 黒背景
# DARK_BACKGROUND = False   # 白背景


# ---------------------------------------------------------------------------
# 背景色に応じたカラー設定
# ---------------------------------------------------------------------------

def _theme():
    """DARK_BACKGROUND に応じた描画カラー設定を返す。"""
    if DARK_BACKGROUND:
        return {
            'bg':          '#0D0D1A',
            'pane_edge':   '#333355',
            'tick':        '#AAAACC',
            'label':       '#AAAACC',
            'title':       'white',
            'cbar_label':  'white',
            'cbar_tick':   'white',
            # マーカー
            'marker_c':    'white',       # ★ 黒背景用
            # 'marker_c':  'black',       # ★ 白背景用
            'marker_edge': 'gray',
            # 骨格線
            'bone_c':      'white',       # ★ 黒背景用
            # 'bone_c':    'black',       # ★ 白背景用
            'bone_edge':   'gray',
        }
    else:
        return {
            'bg':          'white',
            'pane_edge':   '#CCCCCC',
            'tick':        '#333333',
            'label':       '#333333',
            'title':       'black',
            'cbar_label':  'black',
            'cbar_tick':   'black',
            # マーカー
            # 'marker_c':  'white',       # ★ 黒背景用
            'marker_c':    'black',       # ★ 白背景用
            'marker_edge': 'gray',
            # 骨格線
            # 'bone_c':    'white',       # ★ 黒背景用
            'bone_c':      'black',       # ★ 白背景用
            'bone_edge':   'gray',
        }


# ---------------------------------------------------------------------------
# 等値面・ベクトル場の定数
# ---------------------------------------------------------------------------

SIGMA_DEFAULT = 60.0   # ガウス拡散の広がり [mm]

# エネルギー等値面のレベル（最大値に対する割合）と色・透明度
# 外側=薄い → 内側=濃い
ISOSURFACE_LEVELS = [
    (0.05, '#4A90D9', 0.08),   # 最外層: 薄青
    (0.15, '#2E75B6', 0.13),   # 外層:   青
    (0.30, '#F5A623', 0.20),   # 中層:   橙
    (0.55, '#D0021B', 0.30),   # 最内層: 赤
]

VECTOR_DENSITY = 3   # quiver の間引き (整数: 大きいほど疎)


# ---------------------------------------------------------------------------
# エネルギー場・力場の計算
# ---------------------------------------------------------------------------

def _calc_segment_energies(frame_data: dict, cfg: dict) -> dict:
    """
    1フレームの各セグメントの弾性エネルギー u_i [J] を計算する。

    u_i = T_i × ε_i / 2
    ε_i = max(0, length_i - natural_length_i) / 1000  [m]

    NATURAL_LENGTHS が未定義のセグメントは ε_i ≈ length_i / 1000 で近似。

    Returns
    -------
    dict: seg_name -> {'u': float[J], 'p1': array, 'p2': array}
    """
    pos_map       = frame_data['pos_map']
    rubber_states = frame_data.get('rubber_states', {})
    nat_lengths   = cfg.get('NATURAL_LENGTHS', {})   # mm単位
    lines_def     = cfg.get('LINES_TO_DRAW', {})

    result = {}
    for seg_name, (p1_id, p2_id) in lines_def.items():
        if seg_name not in rubber_states:
            continue
        rs = rubber_states[seg_name]
        tension = rs.get('ten', 0.0)
        if tension <= 0:
            continue
        p1 = pos_map.get(p1_id)
        p2 = pos_map.get(p2_id)
        if p1 is None or p2 is None:
            continue

        length_mm = np.linalg.norm(p1 - p2)
        nat_mm    = nat_lengths.get(seg_name, 0.0)
        strain_m  = max(0.0, (length_mm - nat_mm) / 1000.0)
        u_i       = tension * strain_m / 2.0   # [J]

        result[seg_name] = {'u': u_i, 'p1': p1, 'p2': p2,
                             'ten': tension, 'len': length_mm}
    return result


def _calc_marker_forces(frame_data: dict, cfg: dict) -> dict:
    """
    1フレームの各マーカーへの合力ベクトルを計算する。

    Returns
    -------
    dict: marker_id -> np.array([Fx, Fy, Fz]) [N]
    """
    pos_map       = frame_data['pos_map']
    lines_def     = cfg.get('LINES_TO_DRAW', {})
    rubber_states = frame_data.get('rubber_states', {})

    marker_forces = {mid: np.zeros(3) for mid in pos_map}

    for seg_name, (p1, p2) in lines_def.items():
        if seg_name not in rubber_states:
            continue
        tension = rubber_states[seg_name].get('ten', 0.0)
        if tension <= 0 or p1 not in pos_map or p2 not in pos_map:
            continue
        vec  = pos_map[p2] - pos_map[p1]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        unit = vec / norm
        marker_forces[p1] += unit * tension
        marker_forces[p2] += -unit * tension

    return marker_forces


def calc_energy_field_volume(frame_data: dict, cfg: dict,
                              grid_n: int = 24,
                              sigma: float = SIGMA_DEFAULT,
                              margin: float = 80.0):
    """
    3D格子上で弾性エネルギー場 U(P) と力場ベクトル F(P) を計算する。

    Returns
    -------
    tuple:
        (X, Y, Z)        : meshgrid 座標 [mm]
        U_field          : 弾性エネルギー場 [J]  ← 等値面に使用
        (Fx, Fy, Fz)     : 力場ベクトル [N]      ← 矢印に使用
        seg_energies     : セグメント別エネルギー辞書
        (xr, yr, zr)     : 軸の範囲
    """
    pos_map = frame_data['pos_map']
    if not pos_map:
        return None

    all_pos = np.array(list(pos_map.values()))
    x_min, y_min, z_min = all_pos.min(axis=0) - margin
    x_max, y_max, z_max = all_pos.max(axis=0) + margin

    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    zs = np.linspace(z_min, z_max, grid_n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    # ① 弾性エネルギー場 (スカラー, 常に正 → 等値面に適する)
    U_field = np.zeros_like(X)
    seg_energies = _calc_segment_energies(frame_data, cfg)

    for seg_name, data in seg_energies.items():
        u_i = data['u']
        if u_i < 1e-9:
            continue
        # エネルギーはセグメントの両端マーカーを中心点としてガウス拡散
        for p in [data['p1'], data['p2']]:
            r2 = (X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2
            w  = np.exp(-r2 / (2 * sigma**2))
            U_field += u_i * w

    # ② 張力ベクトル場 (矢印用)
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    Fz = np.zeros_like(Z)
    marker_forces = _calc_marker_forces(frame_data, cfg)

    for mid, F_vec in marker_forces.items():
        if np.linalg.norm(F_vec) < 0.01:
            continue
        p  = pos_map[mid]
        r2 = (X - p[0])**2 + (Y - p[1])**2 + (Z - p[2])**2
        w  = np.exp(-r2 / (2 * sigma**2))
        Fx += F_vec[0] * w
        Fy += F_vec[1] * w
        Fz += F_vec[2] * w

    return ((X, Y, Z), U_field, (Fx, Fy, Fz), seg_energies,
            (x_min, x_max), (y_min, y_max), (z_min, z_max))


# ---------------------------------------------------------------------------
# 描画ヘルパー
# ---------------------------------------------------------------------------

def _draw_isosurfaces(ax, X, Y, Z, U_field, levels_frac):
    """弾性エネルギー場 U_field の等値面を Marching Cubes で描画する。"""
    if not HAS_SKIMAGE:
        return

    u_max = U_field.max()
    if u_max < 1e-9:
        return

    for frac, color, alpha in levels_frac:
        level = frac * u_max
        try:
            verts, faces, _, _ = measure.marching_cubes(U_field, level=level)
        except (ValueError, RuntimeError):
            continue

        x_scale = (X[-1, 0, 0] - X[0, 0, 0]) / (U_field.shape[0] - 1)
        y_scale = (Y[0, -1, 0] - Y[0, 0, 0]) / (U_field.shape[1] - 1)
        z_scale = (Z[0, 0, -1] - Z[0, 0, 0]) / (U_field.shape[2] - 1)
        verts_real = (verts * np.array([x_scale, y_scale, z_scale])
                      + np.array([X[0, 0, 0], Y[0, 0, 0], Z[0, 0, 0]]))

        mesh = Poly3DCollection(verts_real[faces], alpha=alpha, linewidth=0)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)


def _draw_vectors(ax, X, Y, Z, Fx, Fy, Fz, density=VECTOR_DENSITY):
    """間引いた格子点に力ベクトル矢印を描画する (plasma カラーマップ)。"""
    d = density
    Xs, Ys, Zs = X[::d, ::d, ::d], Y[::d, ::d, ::d], Z[::d, ::d, ::d]
    Us, Vs, Ws = Fx[::d, ::d, ::d], Fy[::d, ::d, ::d], Fz[::d, ::d, ::d]

    F_mag_s = np.sqrt(Us**2 + Vs**2 + Ws**2)
    nz = F_mag_s > 0.01
    if not nz.any():
        return

    scale = F_mag_s.max()
    if scale < 1e-6:
        return

    cmap    = plt.get_cmap('plasma')
    colors  = cmap(F_mag_s[nz] / scale)
    arrow_len = (X[-1, 0, 0] - X[0, 0, 0]) / (X.shape[0] / density) * 0.35

    ax.quiver(
        Xs[nz], Ys[nz], Zs[nz],
        Us[nz] / (F_mag_s[nz] + 1e-9) * arrow_len,
        Vs[nz] / (F_mag_s[nz] + 1e-9) * arrow_len,
        Ws[nz] / (F_mag_s[nz] + 1e-9) * arrow_len,
        color=colors, length=1.0, normalize=False,
        linewidth=0.8, arrow_length_ratio=0.4, alpha=0.7
    )


def _draw_skeleton(ax, frame_data, cfg, th):
    """マーカー・ゴム線・骨格線を参照用に描画する。"""
    pos_map       = frame_data['pos_map']
    lines_def     = cfg.get('LINES_TO_DRAW', {})
    joint_centers = frame_data.get('joint_centers', {})
    rubber_states = frame_data.get('rubber_states', {})

    # マーカー
    if pos_map:
        pts = np.array(list(pos_map.values()))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=th['marker_c'],         # ★ 黒背景: 'white' / 白背景: 'black'
                   # c='white',              # ★ 黒背景用 (コメントアウトで切り替え)
                   # c='black',              # ★ 白背景用 (コメントアウトで切り替え)
                   s=18, zorder=5,
                   edgecolors=th['marker_edge'], linewidths=0.5)

    # ゴム線（張力の大きさで黄→赤着色）
    tensions = [rubber_states[n].get('ten', 0) for n in lines_def if n in rubber_states]
    t_max = max(tensions) if tensions else 1.0
    cmap_rubber = plt.get_cmap('YlOrRd')

    for seg_name, (p1, p2) in lines_def.items():
        if p1 not in pos_map or p2 not in pos_map:
            continue
        a, b = pos_map[p1], pos_map[p2]
        t = rubber_states.get(seg_name, {}).get('ten', 0)
        color = cmap_rubber(t / (t_max + 1e-9))
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, linewidth=2.5, alpha=0.9, zorder=6)

    # 骨格線
    bone_pts = []
    for jname in ['Hip', 'Knee', 'Ankle']:
        jc = joint_centers.get(jname)
        if jc is not None and not np.any(np.isnan(jc)):
            bone_pts.append(jc)
    if len(bone_pts) >= 2:
        bp = np.array(bone_pts)
        ax.plot(bp[:, 0], bp[:, 1], bp[:, 2],
                color=th['bone_c'],          # ★ 黒背景: 'white' / 白背景: 'black'
                # color='white',             # ★ 黒背景用 (コメントアウトで切り替え)
                # color='black',             # ★ 白背景用 (コメントアウトで切り替え)
                linewidth=3, marker='o', markersize=7,
                zorder=7, markeredgecolor=th['bone_edge'])


# ---------------------------------------------------------------------------
# メイン表示関数
# ---------------------------------------------------------------------------

def show_force_field_map(frames_data: list, cfg: dict,
                          task_key: str, phase: int, speed: str,
                          target_pcts: list = None,
                          sigma: float = SIGMA_DEFAULT,
                          grid_n: int = 24,
                          save_dir: str = None):
    """
    Futto力場の3Dマップをインタラクティブウィンドウで表示する。

    等値面: 弾性エネルギー場 U(P)  (重力場ポテンシャルに対応)
    矢印:   張力ベクトル場 F(P)    (力の向きと大きさ)
    ゴム線: 張力の大きさで黄→赤着色

    Parameters
    ----------
    frames_data   : calc_frame_physics() の出力リスト
    cfg           : TASK_CONFIGS[task_key]
    target_pcts   : 表示する歩行周期 [%] のリスト (デフォルト: [0,25,50,75])
    sigma         : ガウス拡散の広がり [mm]
    grid_n        : 格子点数 (推奨 20〜30)
    save_dir      : 指定時は PNG として保存
    """
    if not frames_data:
        print("フレームデータがありません。")
        return

    if target_pcts is None:
        target_pcts = [0, 10, 25, 40, 50, 60, 75, 90]

    th        = _theme()
    all_steps = [f['step'] for f in frames_data]

    print(f"\n力場マップを計算中... (σ={sigma}mm, grid={grid_n}³)")
    print(f"背景色: {'黒' if DARK_BACKGROUND else '白'}")
    print("等値面レベル (最大値に対する割合):",
          [f"{lv[0]*100:.0f}%" for lv in ISOSURFACE_LEVELS])

    for target_pct in target_pcts:
        idx        = int(np.argmin([abs(s - target_pct) for s in all_steps]))
        frame_data = frames_data[idx]
        actual_pct = frame_data['step']

        print(f"\n  [{actual_pct:.0f}%] 計算中...", end='', flush=True)

        result = calc_energy_field_volume(frame_data, cfg,
                                           grid_n=grid_n, sigma=sigma)
        if result is None:
            print(" スキップ (データなし)")
            continue

        (X, Y, Z), U_field, (Fx, Fy, Fz), seg_energies, xr, yr, zr = result
        u_max = U_field.max()
        u_total = sum(d['u'] for d in seg_energies.values())
        print(f" 完了  U_max={u_max:.4f}J  合計エネルギー={u_total:.4f}J")

        # --- 描画 ---
        fig = plt.figure(figsize=(12, 10), facecolor=th['bg'])
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(th['bg'])
        fig.patch.set_facecolor(th['bg'])

        ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
        ax.set_box_aspect([xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]])
        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.pane.fill = False
            spine.pane.set_edgecolor(th['pane_edge'])
        ax.tick_params(colors=th['tick'], labelsize=8)
        ax.set_xlabel('X mm', color=th['label'], fontsize=9)
        ax.set_ylabel('Y mm', color=th['label'], fontsize=9)
        ax.set_zlabel('Z mm', color=th['label'], fontsize=9)
        ax.view_init(elev=25, azim=55)

        ax.set_title(
            f"Futto Energy Field  |  {task_key}  Phase{phase} ({speed}m/s)\n"
            f"Gait Cycle: {actual_pct:.0f}%  "
            f"σ={sigma}mm  U_total={u_total:.4f}J",
            color=th['title'], fontsize=12, pad=10
        )

        # ① 弾性エネルギー等値面
        _draw_isosurfaces(ax, X, Y, Z, U_field, ISOSURFACE_LEVELS)

        # ② 張力ベクトル矢印
        _draw_vectors(ax, X, Y, Z, Fx, Fy, Fz, density=VECTOR_DENSITY)

        # ③ マーカー・ゴム線・骨格
        _draw_skeleton(ax, frame_data, cfg, th)

        # カラーバー (エネルギー等値面用)
        cmap_iso = mcolors.LinearSegmentedColormap.from_list(
            'energy', ['#4A90D9', '#F5A623', '#D0021B']
        )
        sm = plt.cm.ScalarMappable(
            cmap=cmap_iso,
            norm=mcolors.Normalize(vmin=0, vmax=u_max)
        )
        cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=15,
                             pad=0.02, location='left')
        cbar.set_label('Elastic Energy (J)', color=th['cbar_label'], fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=th['cbar_tick'],
                                       labelcolor=th['cbar_tick'])

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            bg_tag  = 'dark' if DARK_BACKGROUND else 'light'
            save_path = os.path.join(
                save_dir,
                f"{task_key}_Phase{phase}_{speed}ms"
                f"_energyfield_{int(actual_pct):03d}_{bg_tag}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                         facecolor=th['bg'])
            print(f"  -> 保存: {save_path}")

        print(f"  [{actual_pct:.0f}%] ウィンドウを表示中 (閉じると次へ)")
        plt.show()
        plt.close(fig)