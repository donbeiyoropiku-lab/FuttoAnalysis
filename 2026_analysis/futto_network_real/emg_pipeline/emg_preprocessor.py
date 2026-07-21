"""
emg_pipeline/emg_preprocessor.py
===================================
EMG 生データの前処理パイプライン

処理ステップ（この順番で実行）:
  Step 1: ベースライン補正（直流オフセット除去）
           → 静止前区間（0〜40s）の平均値を各チャンネルから差し引く
  Step 2: バンドパスフィルタ（20〜450 Hz）
           → 動作アーティファクト（<20Hz）と高周波ノイズ（>450Hz）を除去
  Step 3: ノッチフィルタ（59〜61 Hz）
           → 電源ノイズを除去
  Step 4: 全波整流（Rectification）
           → 信号の絶対値を取る（|EMG|）
  Step 5: ローパスフィルタ（10 Hz）
           → 包絡線（エンベロープ）を抽出
  Step 6: 正規化（%Peak）
           → 各チャンネルの最大値で除算し 0〜1 に統一

出力:
  - 前処理済み EMG（shape: N_ch × T_samples）
  - 各ステップの中間出力（デバッグ・品質確認用）
  - 前処理済みデータの CSV 保存
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scipy.signal import butter, sosfiltfilt, iirnotch


# =============================================================================
# フィルタ設計
# =============================================================================

def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """バンドパスフィルタ（Butterworth）の SOS 係数を返す。"""
    nyq  = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')


def _butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """ローパスフィルタ（Butterworth）の SOS 係数を返す。"""
    nyq = fs / 2.0
    return butter(order, cutoff / nyq, btype='low', output='sos')


def _notch_filter(freq: float, fs: float, Q: float = 30.0):
    """
    ノッチフィルタ（IIR）の SOS 係数を返す。
    Q が高いほど帯域が狭くなる。
    """
    b, a = iirnotch(freq / (fs / 2.0), Q)
    # sosfiltfilt で使えるよう SOS に変換
    from scipy.signal import tf2sos
    return tf2sos(b, a)


# =============================================================================
# 結果コンテナ
# =============================================================================

@dataclass
class PreprocessResult:
    """前処理パイプラインの出力コンテナ。"""
    task_key      : str
    phase         : int
    speed         : str
    channel_names : list[str]
    fs            : float

    # shape (N_ch, T_samples)
    raw           : np.ndarray   # Step 0: 生データ [μV]
    baseline_corr : np.ndarray   # Step 1: ベースライン補正後
    bandpassed    : np.ndarray   # Step 2+3: バンドパス+ノッチ後
    rectified     : np.ndarray   # Step 4: 整流後
    enveloped     : np.ndarray   # Step 5: ローパス（包絡線）後
    normalized    : np.ndarray   # Step 6: 正規化後（%Peak, 0〜1）

    # 各チャンネルの最大値（正規化の基準値）
    peak_values   : np.ndarray   # shape (N_ch,)

    # 時刻配列
    time_s        : np.ndarray   # shape (T_samples,)

    @property
    def N_ch(self) -> int:
        return len(self.channel_names)

    @property
    def T(self) -> int:
        return self.normalized.shape[1]

    def summary(self) -> str:
        lines = [
            f"PreprocessResult [{self.task_key} Phase{self.phase} {self.speed}m/s]",
            f"  Channels : {self.N_ch}",
            f"  Samples  : {self.T}  ({self.T / self.fs:.1f} s @ {self.fs:.0f} Hz)",
            f"  Peak vals: min={self.peak_values.min():.2f}  "
            f"max={self.peak_values.max():.2f}  mean={self.peak_values.mean():.2f} μV",
        ]
        return "\n".join(lines)


# =============================================================================
# メイン前処理クラス
# =============================================================================

class EMGPreprocessor:
    """
    Cometa Pico 生データを前処理するクラス。

    使用例:
    --------
    >>> from emg_pipeline.emg_loader import load_cometa_txt, extract_phase, get_emg_array
    >>> df_all  = load_cometa_txt(r"C:\\...\\task01.txt")
    >>> df_base = extract_phase(df_all, phase=0)   # 静止前区間（ベースライン）
    >>> df_walk = extract_phase(df_all, phase=3)   # 1.1 m/s 区間
    >>> proc    = EMGPreprocessor(fs=2000)
    >>> result  = proc.run(df_walk, df_base, task_key='task01', phase=3)
    """

    def __init__(
        self,
        fs              : float = 2000.0,
        bandpass_low    : float = 20.0,
        bandpass_high   : float = 450.0,
        notch_freqs     : list[float] = [59.0, 61.0],
        lowpass_cutoff  : float = 10.0,
        filter_order    : int   = 4,
    ):
        """
        Parameters
        ----------
        fs             : サンプリング周波数 [Hz]
        bandpass_low   : バンドパス下限 [Hz]
        bandpass_high  : バンドパス上限 [Hz]
        notch_freqs    : ノッチ周波数リスト [Hz]
        lowpass_cutoff : 包絡線抽出のローパス上限 [Hz]
        filter_order   : Butterworth フィルタ次数
        """
        self.fs             = fs
        self.bandpass_low   = bandpass_low
        self.bandpass_high  = bandpass_high
        self.notch_freqs    = notch_freqs
        self.lowpass_cutoff = lowpass_cutoff
        self.filter_order   = filter_order

        # フィルタ係数を事前計算
        self._sos_bp = _butter_bandpass(
            bandpass_low, bandpass_high, fs, filter_order
        )
        self._sos_notch = [
            _notch_filter(f, fs) for f in notch_freqs
        ]
        self._sos_lp = _butter_lowpass(lowpass_cutoff, fs, filter_order)

    # ------------------------------------------------------------------
    # メインパイプライン
    # ------------------------------------------------------------------

    def run(
        self,
        df_walk      : pd.DataFrame,
        df_baseline  : Optional[pd.DataFrame] = None,
        task_key     : str = "",
        phase        : int = 0,
        speed        : str = "",
        channel_names: Optional[list[str]] = None,
    ) -> PreprocessResult:
        """
        前処理パイプラインを実行する。

        Parameters
        ----------
        df_walk      : 歩行区間の DataFrame（load_cometa_txt → extract_phase の出力）
        df_baseline  : ベースライン区間（静止前 Phase=0）の DataFrame
                       None の場合は df_walk の先頭2秒を使用
        task_key     : タスク名（メタデータ）
        phase        : フェーズ番号（メタデータ）
        speed        : 速度文字列（メタデータ）
        channel_names: 処理する列名リスト。None なら全 EMG 列

        Returns
        -------
        PreprocessResult
        """
        from emg_pipeline.emg_loader import get_emg_array

        # チャンネル選択
        if channel_names is None:
            channel_names = [c for c in df_walk.columns
                             if c != 'Time_s' and not c.startswith('Acc')]

        emg_raw, ch = get_emg_array(df_walk, channel_names)   # (N_ch, T)
        time_s      = df_walk['Time_s'].values

        N_ch, T = emg_raw.shape
        print(f"[前処理] {task_key} Phase{phase}  {N_ch}ch × {T}samples ({T/self.fs:.1f}s)")

        # ── Step 1: ベースライン補正 ─────────────────────────────────
        print("  Step 1: ベースライン補正")
        baseline_mean = self._compute_baseline(df_baseline, channel_names, emg_raw)
        emg_bc = emg_raw - baseline_mean[:, np.newaxis]   # (N_ch, T)

        # ── Step 2: バンドパスフィルタ ──────────────────────────────
        print(f"  Step 2: バンドパスフィルタ ({self.bandpass_low}〜{self.bandpass_high} Hz)")
        emg_bp = np.zeros_like(emg_bc)
        for i in range(N_ch):
            emg_bp[i] = sosfiltfilt(self._sos_bp, emg_bc[i])

        # ── Step 3: ノッチフィルタ ─────────────────────────────────
        print(f"  Step 3: ノッチフィルタ {self.notch_freqs} Hz")
        emg_notch = emg_bp.copy()
        for sos_n in self._sos_notch:
            for i in range(N_ch):
                emg_notch[i] = sosfiltfilt(sos_n, emg_notch[i])

        # ── Step 4: 全波整流 ────────────────────────────────────────
        print("  Step 4: 全波整流 (|EMG|)")
        emg_rect = np.abs(emg_notch)

        # ── Step 5: ローパスフィルタ（包絡線） ──────────────────────
        print(f"  Step 5: ローパスフィルタ ({self.lowpass_cutoff} Hz) → 包絡線")
        emg_env = np.zeros_like(emg_rect)
        for i in range(N_ch):
            emg_env[i] = sosfiltfilt(self._sos_lp, emg_rect[i])
        emg_env = np.maximum(emg_env, 0)   # 負値を 0 にクリップ

        # ── Step 6: 正規化（%Peak） ──────────────────────────────────
        print("  Step 6: 正規化 (%Peak)")
        peak_values = emg_env.max(axis=1)   # (N_ch,)
        emg_norm    = np.zeros_like(emg_env)
        for i in range(N_ch):
            if peak_values[i] > 1e-9:
                emg_norm[i] = emg_env[i] / peak_values[i]
            else:
                emg_norm[i] = np.zeros(T)
                print(f"    [警告] {ch[i]} のピーク値がゼロです。")

        print(f"  完了。正規化後: [{emg_norm.min():.3f}, {emg_norm.max():.3f}]")

        return PreprocessResult(
            task_key      = task_key,
            phase         = phase,
            speed         = speed,
            channel_names = ch,
            fs            = self.fs,
            raw           = emg_raw,
            baseline_corr = emg_bc,
            bandpassed    = emg_notch,
            rectified     = emg_rect,
            enveloped     = emg_env,
            normalized    = emg_norm,
            peak_values   = peak_values,
            time_s        = time_s,
        )

    # ------------------------------------------------------------------
    # ベースライン計算
    # ------------------------------------------------------------------

    def _compute_baseline(
        self,
        df_baseline   : Optional[pd.DataFrame],
        channel_names : list[str],
        emg_raw       : np.ndarray,
    ) -> np.ndarray:
        """
        各チャンネルのベースライン（直流オフセット）を計算する。

        df_baseline が与えられた場合: その区間の平均値を使用
        None の場合: emg_raw の先頭 2 秒の平均値を使用

        Returns
        -------
        baseline : np.ndarray shape (N_ch,)
        """
        if df_baseline is not None:
            from emg_pipeline.emg_loader import get_emg_array
            bl_emg, _ = get_emg_array(df_baseline, channel_names)
            return bl_emg.mean(axis=1)
        else:
            n_baseline = int(2.0 * self.fs)   # 先頭 2 秒
            return emg_raw[:, :n_baseline].mean(axis=1)

    # ------------------------------------------------------------------
    # CSV 保存
    # ------------------------------------------------------------------

    def save_csv(
        self,
        result   : PreprocessResult,
        out_dir  : str | Path,
        step     : str = 'normalized',
    ) -> Path:
        """
        前処理済みデータを CSV に保存する。

        Parameters
        ----------
        result  : PreprocessResult
        out_dir : 保存先ディレクトリ
        step    : 保存するステップ
                  'raw' / 'baseline_corr' / 'bandpassed' /
                  'rectified' / 'enveloped' / 'normalized'

        Returns
        -------
        保存先ファイルパス
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data  = getattr(result, step)   # (N_ch, T)
        df_out= pd.DataFrame(
            data.T,
            columns=result.channel_names,
        )
        df_out.insert(0, 'Time_s', result.time_s)

        fname = (
            out_dir
            / f"{result.task_key}_Phase{result.phase}"
              f"_{result.speed}ms_emg_{step}.csv"
        )
        df_out.to_csv(fname, index=False, float_format='%.6f')
        print(f"  [保存] {fname.name}")
        return fname

    def save_all_steps(
        self,
        result  : PreprocessResult,
        out_dir : str | Path,
    ) -> list[Path]:
        """全ステップの CSV を保存する。"""
        steps = ['raw', 'baseline_corr', 'bandpassed',
                 'rectified', 'enveloped', 'normalized']
        paths = []
        for step in steps:
            paths.append(self.save_csv(result, out_dir, step))
        return paths


# =============================================================================
# 品質確認プロット
# =============================================================================

def plot_preprocessing_steps(
    result      : PreprocessResult,
    channel_idx : int = 0,
    save_path   : Optional[str | Path] = None,
    t_range     : Optional[tuple[float, float]] = (0, 5),
) -> None:
    """
    指定チャンネルの各前処理ステップの波形を縦に並べてプロットする。

    Parameters
    ----------
    result      : PreprocessResult
    channel_idx : 表示するチャンネルのインデックス
    save_path   : 保存先パス（None の場合は画面表示）
    t_range     : 表示する時間範囲 [s]（None で全区間）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ch_name = result.channel_names[channel_idx]
    steps   = [
        ('raw',           f'1. Raw signal [uV]',         result.raw),
        ('baseline_corr', '2. Baseline corrected [uV]', result.baseline_corr),
        ('bandpassed',    '3. Bandpass + Notch filtered [uV]', result.bandpassed),
        ('rectified',     '4. Full-wave rectified [uV]',          result.rectified),
        ('enveloped',     '5. Envelope (Low-pass) [uV]', result.enveloped),
        ('normalized',    '6. Normalized (%Peak) [0-1]',  result.normalized),
    ]

    n     = len(steps)
    t     = result.time_s
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)

    for ax, (_, label, data) in zip(axes, steps):
        sig = data[channel_idx]
        if t_range is not None:
            t0, t1 = t_range
            mask = (t >= t_range[0]) & (t <= t_range[1])
            ax.plot(t[mask], sig[mask], linewidth=0.8, color='#2c3e50')
        else:
            ax.plot(t, sig, linewidth=0.6, color='#2c3e50')
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]', fontsize=10)
    fig.suptitle(
        f'EMG 前処理ステップ — {ch_name}\n'
        f'{result.task_key}  Phase{result.phase}  {result.speed}m/s',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [QC図] 保存 → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_all_channels_normalized(
    result    : PreprocessResult,
    save_path : Optional[str | Path] = None,
    t_range   : Optional[tuple[float, float]] = None,
) -> None:
    """
    全チャンネルの正規化済み包絡線を縦に並べてプロットする（品質確認用）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    N   = result.N_ch
    t   = result.time_s

    fig, axes = plt.subplots(N, 1, figsize=(14, 1.8 * N), sharex=True)
    if N == 1:
        axes = [axes]

    colors_R = '#3498db'
    colors_L = '#e74c3c'

    for idx, (ax, ch) in enumerate(zip(axes, result.channel_names)):
        sig   = result.normalized[idx]
        color = colors_R if ch.startswith('R_') else colors_L

        if t_range is not None:
            mask = (t >= t_range[0]) & (t <= t_range[1])
            ax.plot(t[mask], sig[mask], color=color, linewidth=1.0)
        else:
            ax.plot(t, sig, color=color, linewidth=0.7)

        ax.set_ylabel(ch, fontsize=8, rotation=0, ha='right', labelpad=50)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.2)
        ax.set_yticks([0, 0.5, 1.0])

    axes[-1].set_xlabel('Time [s]', fontsize=10)
    fig.suptitle(
        f'Normalized EMG Envelope - All {N} Channels\n'
        f'{result.task_key}  Phase{result.phase}  {result.speed}m/s',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [全ch図] 保存 → {save_path}")
    else:
        plt.show()
    plt.close(fig)
