"""
rla_phase_network.py
ランチョロスアミーゴ（RLA）8 フェーズ別 筋協調ネットワーク解析

【概要】
  gaitcycle_force_labchart.py で生成した gait_cycles CSV（hs_frame, to_frame, next_hs_frame）と
  emg_pipeline で生成した正規化済み EMG CSV を組み合わせて、
  RLA 8 フェーズそれぞれの Graphical Lasso・Granger 因果性ネットワークを計算する。

【RLA 8 フェーズ定義（歩行周期 % ベース）】
  1. Initial Contact     (IC)  :   0 –  2%
  2. Loading Response    (LR)  :   2 – 12%
  3. Mid Stance          (MSt) :  12 – 31%
  4. Terminal Stance     (TSt) :  31 – 50%
  5. Pre-Swing           (PSw) :  50 – 62%  ← hs_frame → to_frame が 0–62% に相当
  6. Initial Swing       (ISw) :  62 – 75%  ← to_frame → next_hs_frame の中間
  7. Mid Swing           (MSw) :  75 – 87%
  8. Terminal Swing      (TSw) :  87 – 100%

【入力ファイル】
  (A) EMG 正規化済み CSV:
    C:\\FuttoAnalysis\\result\\2026\\{subject}\\{task}\\{speed}\\
      {task}_Phase{N}_{speed}ms_emg_normalized.csv
    → 列: Time_s + 16 チャンネル名、サンプリング 2000 Hz

  (B) 歩行周期 CSV（gaitcycle_force_labchart.py の出力）:
    C:\\FuttoAnalysis\\labchart\\{date}\\{task}_gait_cycles.csv
    → 列: hs_time, to_time, next_hs_time, hs_frame, to_frame, next_hs_frame
    ※ GRF サンプリングは 1000 Hz → EMG は 2000 Hz なのでフレーム番号を 2 倍して対応

【出力】
  C:\\FuttoAnalysis\\result\\2026\\{subject}\\rla_network\\Ph{N}_{speed}\\
    glasso_rla_{task}_Ph{N}_{speed}.csv   ← 8×8フェーズ × タスク別偏相関行列
    granger_rla_{task}_Ph{N}_{speed}.csv  ← 8フェーズ別有向エッジ数
    rla_network_summary.csv               ← 全フェーズ × タスク × 指標の一覧

使い方:
  python rla_phase_network.py --subject Ide --phase 3 --date 20260217
  python rla_phase_network.py --all_phases --date 20260217
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLassoCV
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# パス設定
# ──────────────────────────────────────────────
RESULT_BASE  = Path(r"C:\FuttoAnalysis\result")
LABCHART_BASE= Path(r"C:\FuttoAnalysis\labchart")

PHASE_SPEED  = {1:'0.7', 2:'0.9', 3:'1.1', 4:'1.3', 5:'1.5'}
TASKS        = ['task01', 'task02', 'task03']

# GRF サンプリング周波数（data_processing.py のデフォルト）
GRF_FS = 1000   # Hz
EMG_FS = 2000   # Hz
FRAME_RATIO = EMG_FS / GRF_FS   # = 2.0（GRF の 1 フレーム = EMG の 2 フレーム）

# ──────────────────────────────────────────────
# RLA 8 フェーズ定義（歩行周期 % の範囲）
# ──────────────────────────────────────────────
RLA_PHASES = {
    'IC':  (0,   2),   # Initial Contact
    'LR':  (2,  12),   # Loading Response
    'MSt': (12, 31),   # Mid Stance
    'TSt': (31, 50),   # Terminal Stance
    'PSw': (50, 62),   # Pre-Swing
    'ISw': (62, 75),   # Initial Swing
    'MSw': (75, 87),   # Mid Swing
    'TSw': (87, 100),  # Terminal Swing
}
RLA_NAMES = list(RLA_PHASES.keys())


# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────
def load_emg(subject: str, task: str, phase: int, speed: str) -> tuple:
    """正規化済み EMG CSV を読み込む。(16, T) の numpy 配列とチャンネル名を返す。"""
    path = (RESULT_BASE / '2026' / subject / task / speed
            / f'{task}_Phase{phase}_{speed}ms_emg_normalized.csv')
    if not path.exists():
        raise FileNotFoundError(f'EMG CSV が見つかりません: {path}')
    df       = pd.read_csv(str(path))
    ch_names = [c for c in df.columns if c != 'Time_s']
    emg      = df[ch_names].values.T.astype(float)   # (16, T)
    return emg, ch_names


def load_gait_cycles(task: str, date: str) -> pd.DataFrame:
    """
    gaitcycle_force_labchart.py が保存した CSV を読み込む。
    列: hs_frame, to_frame, next_hs_frame（GRF 1000 Hz ベース）
    """
    path = LABCHART_BASE / date / f'{task}_gait_cycles.csv'
    if not path.exists():
        raise FileNotFoundError(f'GRF 歩行周期 CSV が見つかりません: {path}')
    df = pd.read_csv(str(path))
    required = {'hs_frame', 'to_frame', 'next_hs_frame'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV に必要な列がありません: {required - set(df.columns)}')
    return df


def filter_cycles_by_speed_phase(
    gait_df : pd.DataFrame,
    phase   : int,
    n_speed_phases: int = 5,
    trial_duration_sec: float = 60.0,
    grf_fs  : int = GRF_FS,
    margin_sec: float = 12.0,
) -> pd.DataFrame:
    """
    速度フェーズ（1 分ごと）に対応する歩行周期だけを抽出する。

    仮定: 試験は n_speed_phases × trial_duration_sec 秒の連続記録
    phase 1 → 0–60 秒、phase 2 → 60–120 秒、…

    margin_sec: 速度切り替え直後・直前を除外する余白（秒）
    """
    phase_start_sec = (phase - 1) * trial_duration_sec
    phase_end_sec   = phase       * trial_duration_sec

    # 余白を加えた有効区間（秒）
    valid_start_sec = phase_start_sec + margin_sec
    valid_end_sec   = phase_end_sec   - margin_sec

    # GRF フレーム → 秒
    hs_sec = gait_df['hs_frame'] / grf_fs
    filtered = gait_df[
        (hs_sec >= valid_start_sec) & (hs_sec <= valid_end_sec)
    ].copy()

    print(f'  [filter] Phase {phase}: {valid_start_sec:.0f}–{valid_end_sec:.0f}s → {len(filtered)} 歩行周期')
    return filtered.reset_index(drop=True)


# ──────────────────────────────────────────────
# RLA フェーズ分割
# ──────────────────────────────────────────────
def extract_rla_emg_segments(
    emg       : np.ndarray,   # (16, T_emg)
    gait_df   : pd.DataFrame,
    frame_ratio: float = FRAME_RATIO,
) -> dict:
    """
    各歩行周期を RLA 8 フェーズに分割し、
    フェーズごとに EMG セグメントを積み重ねた配列を返す。

    Returns
    -------
    dict: {phase_name: np.ndarray shape (N_cycles * segment_len, 16)}
        N_cycles 分のセグメントを縦に連結した行列。
        Graphical Lasso は「大量の観測」として扱う。
    """
    T_emg = emg.shape[1]
    segments = {ph: [] for ph in RLA_NAMES}

    for _, row in gait_df.iterrows():
        # GRF フレーム → EMG フレームに変換
        hs_emg      = int(row['hs_frame']      * frame_ratio)
        next_hs_emg = int(row['next_hs_frame'] * frame_ratio)
        cycle_len   = next_hs_emg - hs_emg

        if cycle_len <= 0 or next_hs_emg > T_emg:
            continue   # 範囲外・異常周期はスキップ

        for ph_name, (pct_start, pct_end) in RLA_PHASES.items():
            seg_start = hs_emg + int(cycle_len * pct_start / 100)
            seg_end   = hs_emg + int(cycle_len * pct_end   / 100)
            seg_end   = min(seg_end, T_emg)

            if seg_end > seg_start:
                # shape: (seg_len, 16)
                segments[ph_name].append(emg[:, seg_start:seg_end].T)

    # 各フェーズの全周期を縦方向に連結
    concatenated = {}
    for ph_name, segs in segments.items():
        if segs:
            concatenated[ph_name] = np.vstack(segs)   # (N_total_samples, 16)
            print(f'    {ph_name}: {len(segs)} cycles  samples={concatenated[ph_name].shape[0]}')
        else:
            concatenated[ph_name] = None
            print(f'    {ph_name}: データなし')

    return concatenated


# ──────────────────────────────────────────────
# ① Graphical Lasso（RLA フェーズ別）
# ──────────────────────────────────────────────
def compute_glasso_rla(
    segments : dict,
    ch_names : list,
) -> pd.DataFrame:
    """
    RLA 8 フェーズそれぞれに Graphical Lasso を適用する。

    Returns
    -------
    DataFrame: index=RLA フェーズ名、列=各チャンネルの次数（偏相関の強さの合計）
               + 'n_edges' 列
    """
    rows = []
    for ph_name in RLA_NAMES:
        seg = segments.get(ph_name)
        if seg is None or seg.shape[0] < 20:
            rows.append({'phase': ph_name, 'n_edges': 0,
                         **{ch: 0.0 for ch in ch_names}})
            continue

        try:
            model = GraphicalLassoCV(cv=3, max_iter=500)
            model.fit(seg)
            pcor  = -model.precision_.copy()
            d     = np.sqrt(np.diag(model.precision_))
            pcor /= d[:, None]; pcor /= d[None, :]
            np.fill_diagonal(pcor, 0)
            pcor  = np.abs(pcor)
            n_edges = int((pcor > 0).sum() / 2)
            degree  = pcor.sum(axis=1)
            rows.append({'phase': ph_name, 'n_edges': n_edges,
                         **{ch: round(float(degree[i]), 4) for i, ch in enumerate(ch_names)}})
        except Exception as e:
            print(f'    [GLasso エラー] {ph_name}: {e}')
            rows.append({'phase': ph_name, 'n_edges': 0,
                         **{ch: 0.0 for ch in ch_names}})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# ② Granger 因果性（RLA フェーズ別）
# ──────────────────────────────────────────────
def _granger_pair(seg, i, j, maxlag, p_thresh):
    """1 ペアの Granger 検定（joblib 並列用）"""
    try:
        xy = np.column_stack([seg[:, j], seg[:, i]])
        res = grangercausalitytests(xy, maxlag=maxlag, verbose=False)
        p_min = min(res[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1))
        return (i, j, p_min)
    except Exception:
        return (i, j, 1.0)


def compute_granger_rla(
    segments  : dict,
    ch_names  : list,
    maxlag    : int   = 3,
    p_thresh  : float = 0.05 / 240,   # Bonferroni 補正
    n_jobs    : int   = -1,
) -> pd.DataFrame:
    """
    RLA 8 フェーズそれぞれに Granger 因果性を適用する。

    RLA フェーズのセグメントは短いため maxlag=3 を推奨
    （歩行周期の 1–2 フェーズ分のサンプル数が限られるため）

    Returns
    -------
    DataFrame: index=RLA フェーズ名、列=n_edges + hub_muscle
    """
    N = len(ch_names)
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    rows  = []

    for ph_name in RLA_NAMES:
        seg = segments.get(ph_name)
        if seg is None or seg.shape[0] < maxlag * 5:
            rows.append({'phase': ph_name, 'n_edges': 0, 'hub_muscle': 'N/A', 'max_out_degree': 0})
            continue

        results = Parallel(n_jobs=n_jobs)(
            delayed(_granger_pair)(seg, i, j, maxlag, p_thresh)
            for i, j in pairs
        )
        sig_edges = [(i, j) for i, j, p in results if p < p_thresh]
        n_edges   = len(sig_edges)

        # ハブ筋（送信側出次数が最大の筋）
        out_degree = {ch: 0 for ch in ch_names}
        for i, j in sig_edges:
            out_degree[ch_names[i]] += 1
        hub   = max(out_degree, key=out_degree.get)
        max_d = out_degree[hub]

        rows.append({'phase': ph_name, 'n_edges': n_edges,
                     'hub_muscle': hub, 'max_out_degree': max_d})
        print(f'    {ph_name}: {n_edges}/{len(pairs)} edges  hub={hub}({max_d})')

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
def run_rla_analysis(
    subject : str,
    phase   : int,
    date    : str,
    tasks   : list = None,
    margin_sec: float = 12.0,
    granger_maxlag: int = 3,
):
    if tasks is None:
        tasks = TASKS

    speed   = PHASE_SPEED[phase]
    out_dir = RESULT_BASE / '2026' / subject / 'rla_network' / f'Ph{phase}_{speed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  RLA フェーズ別ネットワーク解析')
    print(f'  {subject}  Phase {phase} ({speed} m/s)')
    print(f'{"="*60}')

    summary_rows = []

    for task in tasks:
        print(f'\n─── {task} ───')

        # EMG 読み込み
        try:
            emg, ch_names = load_emg(subject, task, phase, speed)
            print(f'  EMG: {emg.shape}')
        except FileNotFoundError as e:
            print(f'  [スキップ] {e}')
            continue

        # 歩行周期 CSV 読み込み
        try:
            gait_df = load_gait_cycles(task, date)
        except FileNotFoundError as e:
            print(f'  [スキップ] {e}')
            continue

        # 速度フェーズに対応する周期だけ抽出
        gait_df_ph = filter_cycles_by_speed_phase(gait_df, phase,
                                                   margin_sec=margin_sec)
        if len(gait_df_ph) == 0:
            print(f'  [スキップ] 有効な歩行周期が 0 件')
            continue

        # RLA フェーズ分割
        print('  RLA フェーズ分割:')
        segments = extract_rla_emg_segments(emg, gait_df_ph)

        # ① Graphical Lasso
        print('  ① Graphical Lasso (RLA 8 フェーズ):')
        df_gl = compute_glasso_rla(segments, ch_names)
        df_gl.insert(0, 'task', task)
        gl_path = out_dir / f'glasso_rla_{task}_Ph{phase}_{speed}.csv'
        df_gl.to_csv(str(gl_path), index=False, encoding='utf-8-sig')
        print(f'    → {gl_path}')

        # ② Granger 因果性
        print('  ② Granger 因果性 (RLA 8 フェーズ, Bonferroni補正):')
        df_gc = compute_granger_rla(segments, ch_names,
                                     maxlag=granger_maxlag,
                                     p_thresh=0.05 / 240)
        df_gc.insert(0, 'task', task)
        gc_path = out_dir / f'granger_rla_{task}_Ph{phase}_{speed}.csv'
        df_gc.to_csv(str(gc_path), index=False, encoding='utf-8-sig')
        print(f'    → {gc_path}')

        # サマリー
        for _, row in df_gl.iterrows():
            summary_rows.append({
                'subject': subject, 'task': task, 'phase': phase,
                'speed': speed, 'rla_phase': row['phase'],
                'glasso_n_edges': row['n_edges'],
            })

    # サマリー CSV
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        s_path = out_dir / f'rla_network_summary_Ph{phase}_{speed}.csv'
        df_summary.to_csv(str(s_path), index=False, encoding='utf-8-sig')
        print(f'\n[完了] サマリー → {s_path}')

    return out_dir


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='RLA フェーズ別 筋協調ネットワーク解析'
    )
    p.add_argument('--subject',    default='Ide')
    p.add_argument('--phase',      type=int, default=None,
                   help='速度フェーズ番号 1–5（省略時: 全フェーズ）')
    p.add_argument('--date',       required=True,
                   help='LabChart データの日付フォルダ名（例: 20260217）')
    p.add_argument('--tasks',      nargs='+', default=TASKS)
    p.add_argument('--margin_sec', type=float, default=12.0,
                   help='速度切り替え後の除外秒数（デフォルト: 12秒）')
    p.add_argument('--granger_maxlag', type=int, default=3,
                   help='Granger の最大ラグ（RLA フェーズは短いので 3 推奨）')
    args = p.parse_args()

    phases = [args.phase] if args.phase else list(PHASE_SPEED.keys())

    print('=' * 60)
    print('  RLA Phase Network Analysis')
    print('=' * 60)
    print(f'  subject    : {args.subject}')
    print(f'  phases     : {phases}')
    print(f'  date       : {args.date}')
    print(f'  margin_sec : {args.margin_sec}')
    print(f'  maxlag     : {args.granger_maxlag}')

    for phase in phases:
        run_rla_analysis(
            subject       = args.subject,
            phase         = phase,
            date          = args.date,
            tasks         = args.tasks,
            margin_sec    = args.margin_sec,
            granger_maxlag= args.granger_maxlag,
        )


if __name__ == '__main__':
    main()