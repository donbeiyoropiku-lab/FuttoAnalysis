"""
futto_network/build_graph.py
============================
実CONFIGの構造（TASK_CONFIGS / LINES_TO_DRAW / KEYFRAME_MAP）に基づき、
時変重み付き隣接行列 W(t) を生成する。

実CONFIG仕様:
  - LINES_TO_DRAW  : dict[str, tuple[int,int]]  セグメント名 → (marker_id1, marker_id2)
  - KEYFRAME_MAP   : dict[int, int]             marker_id → marker_id (identity map)
  - ノードIDは5桁整数 (例: 16000, 15970, 57960 ...)
  - task03 は TASK_CONFIGS に存在せず、器具なしの対照条件

データパス規則:
  張力CSV: C:\\FuttoAnalysis\\result\\2026\\{task}\\{speed}\\{task}_Phase{phase}_{speed}ms_tension.csv
  マーカーCSV: C:\\FuttoAnalysis\\opti\\20260217\\{task}_mean_cycle_Phase{phase}_{speed}ms.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import CONFIG as CFG


# =============================================================================
# ノードインデックス管理
# =============================================================================

def build_node_index(task_key: str) -> tuple[list[int], dict[int, int]]:
    """
    KEYFRAME_MAP からノードIDリストと id→行列インデックス辞書を生成する。

    Returns
    -------
    node_ids  : list[int]   ソート済みマーカーID
    id_to_idx : dict[int,int]  マーカーID → 行列インデックス
    """
    if task_key not in CFG.TASK_CONFIGS:
        return [], {}
    kfm = CFG.TASK_CONFIGS[task_key]['KEYFRAME_MAP']
    node_ids  = sorted(kfm.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    return node_ids, id_to_idx


# =============================================================================
# 位相シフト（右脚擬似生成）
# =============================================================================

PHASE_SHIFT_PCT = 50   # 歩行周期の50%シフト

def _phase_shift_50(arr: np.ndarray) -> np.ndarray:
    """
    歩行周期正規化配列（長さT）を50%位相シフトして返す。
    右脚の張力を左脚データから擬似生成するために使用。
    """
    T = len(arr)
    return np.roll(arr, T * PHASE_SHIFT_PCT // 100)


# =============================================================================
# FuttoGraph クラス
# =============================================================================

class FuttoGraph:
    """
    Futto の時変重み付き隣接行列 W(t) を保持するクラス。

    W は (T, N, N) の対称テンソル。
    W[t, i, j] = 時刻 t における マーカーi—j 間のゴム張力 [N]。

    右脚 Futto データが存在しない場合、
    左脚張力を 50% 位相シフトして疑似的な両脚統合行列を構築する。
    """

    def __init__(self, task_key: str):
        self.task_key = task_key
        self.cfg      = CFG.TASK_CONFIGS.get(task_key, {})
        self.node_ids, self.id_to_idx = build_node_index(task_key)
        self.N  = len(self.node_ids)
        self.T  = 101   # gait_cycle_% = 0, 1, ..., 100
        self.W  = np.zeros((self.T, self.N, self.N), dtype=float)
        self._segment_tensions: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # データ読み込み
    # ------------------------------------------------------------------

    def load_from_tension_csv(
        self,
        tension_csv_path: str | Path,
        phase_shift: bool = True,
    ) -> "FuttoGraph":
        """
        張力CSVファイル（実測データ）から W(t) を構築する。

        CSVフォーマット:
          列: gait_cycle_% | Front_Upper_In | Front_Upper_Out | ...
          行: gait_cycle_% = 0〜100 (101行)

        Parameters
        ----------
        tension_csv_path : str or Path
            C:\\FuttoAnalysis\\result\\2026\\{task}\\{speed}\\...tension.csv
        phase_shift : bool
            True の場合、左脚実測値を 50% シフトして右脚分を合成する。

        Returns
        -------
        self
        """
        path = Path(tension_csv_path)
        if not path.exists():
            print(f"[警告] 張力CSVが見つかりません: {path}")
            print("       シミュレーションデータで代替します。")
            return self._fill_simulated()

        df = pd.read_csv(path)

        # gait_cycle_% 列でソート・補間して 0〜100 の 101 点に揃える
        if 'gait_cycle_%' in df.columns:
            df = df.sort_values('gait_cycle_%').reset_index(drop=True)
            gc = df['gait_cycle_%'].values
            seg_cols = [c for c in df.columns if c != 'gait_cycle_%']
        else:
            gc = np.linspace(0, 100, len(df))
            seg_cols = list(df.columns)

        t_new = np.arange(0, 101, dtype=float)
        lines = self.cfg.get('LINES_TO_DRAW', {})

        for seg_name, (id1, id2) in lines.items():
            if seg_name not in seg_cols:
                continue
            if id1 not in self.id_to_idx or id2 not in self.id_to_idx:
                continue

            raw = df[seg_name].values.astype(float)
            # 長さが 101 でない場合は線形補間
            if len(gc) != 101:
                from scipy.interpolate import interp1d
                f   = interp1d(gc, raw, kind='linear',
                               fill_value=(raw[0], raw[-1]), bounds_error=False)
                raw = f(t_new)

            raw = np.clip(raw, 0, None)   # 負値除去
            self._segment_tensions[seg_name] = raw

            i = self.id_to_idx[id1]
            j = self.id_to_idx[id2]

            if phase_shift:
                # 左脚実測 + 右脚擬似（50%シフト）の平均で統合行列を構築
                shifted = _phase_shift_50(raw)
                combined = (raw + shifted) / 2.0
            else:
                combined = raw

            self.W[:, i, j] = combined
            self.W[:, j, i] = combined   # 無向対称

        return self

    def load_from_tension_dict(
        self,
        tension_data: dict[str, np.ndarray],
        phase_shift: bool = True,
    ) -> "FuttoGraph":
        """
        tension_calc.calculate_all_tensions() の出力辞書から W(t) を構築する。

        Parameters
        ----------
        tension_data : dict  セグメント名 → np.ndarray shape(T,)
        phase_shift  : bool  50% 位相シフトを適用するか
        """
        lines = self.cfg.get('LINES_TO_DRAW', {})

        for seg_name, (id1, id2) in lines.items():
            if seg_name not in tension_data:
                continue
            if id1 not in self.id_to_idx or id2 not in self.id_to_idx:
                continue

            raw = np.clip(tension_data[seg_name].astype(float), 0, None)

            # 長さを 101 に揃える
            if len(raw) != 101:
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 100, len(raw))
                x_new = np.arange(0, 101, dtype=float)
                raw   = interp1d(x_old, raw, kind='linear',
                                 fill_value=(raw[0], raw[-1]),
                                 bounds_error=False)(x_new)

            self._segment_tensions[seg_name] = raw
            i = self.id_to_idx[id1]
            j = self.id_to_idx[id2]

            if phase_shift:
                combined = (raw + _phase_shift_50(raw)) / 2.0
            else:
                combined = raw

            self.W[:, i, j] = combined
            self.W[:, j, i] = combined

        return self

    # ------------------------------------------------------------------
    # シミュレーション（実データ未取得時の開発用代替）
    # ------------------------------------------------------------------

    def _fill_simulated(self) -> "FuttoGraph":
        """
        実データがない場合の開発用擬似データで W(t) を埋める。
        典型的な歩行パターン（立脚期ピーク）を模した正弦波。
        """
        rng   = np.random.default_rng(hash(self.task_key) % 2**31)
        t     = np.linspace(0, 2 * np.pi, self.T)
        lines = self.cfg.get('LINES_TO_DRAW', {})

        for idx, (seg_name, (id1, id2)) in enumerate(lines.items()):
            if id1 not in self.id_to_idx or id2 not in self.id_to_idx:
                continue
            # 各セグメントに少しずつ異なる位相・振幅を割り当て
            amp    = rng.uniform(5, 25)
            phase  = rng.uniform(0, np.pi)
            base   = rng.uniform(8, 15)
            raw    = np.clip(base + amp * np.sin(t + phase)
                             + rng.normal(0, 1.5, self.T), 0, None)

            self._segment_tensions[seg_name] = raw
            i = self.id_to_idx[id1]
            j = self.id_to_idx[id2]
            combined = (raw + _phase_shift_50(raw)) / 2.0
            self.W[:, i, j] = combined
            self.W[:, j, i] = combined

        return self

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def adjacency_at(self, t: int) -> np.ndarray:
        """時刻 t の N×N 隣接行列を返す。"""
        return self.W[t]

    def segment_tension(self, seg_name: str) -> Optional[np.ndarray]:
        """指定セグメントの張力時系列（shape (T,)）を返す。"""
        return self._segment_tensions.get(seg_name)

    @property
    def segments(self) -> list[str]:
        return list(self._segment_tensions.keys())

    def summary(self) -> str:
        wmax  = self.W.max() if self.W.size > 0 else 0
        wmean = self.W.mean() if self.W.size > 0 else 0
        return (
            f"FuttoGraph [{self.task_key}]\n"
            f"  Nodes   : {self.N}  {self.node_ids}\n"
            f"  Segments: {len(self.segments)}\n"
            f"  W shape : {self.W.shape}\n"
            f"  W max   : {wmax:.3f} N\n"
            f"  W mean  : {wmean:.3f} N"
        )


# =============================================================================
# パス生成ヘルパー
# =============================================================================

def build_tension_csv_path(
    task_key: str,
    phase: int,
    speed: str,
    result_dir: str = r"C:\FuttoAnalysis\result",
    year: str = "2026",
) -> Path:
    """
    実験プロトコルに従った張力CSVパスを生成する。

    例: C:\\FuttoAnalysis\\result\\2026\\task01\\0.7\\task01_Phase1_0.7ms_tension.csv
    """
    return (
        Path(result_dir) / year / task_key / speed
        / f"{task_key}_Phase{phase}_{speed}ms_tension.csv"
    )


def build_marker_csv_path(
    task_key: str,
    phase: int,
    speed: str,
    opti_base: str = r"C:\FuttoAnalysis\opti\20260217",
) -> Path:
    """
    マーカーデータ（平均周期）CSVのパスを生成する。

    例: C:\\FuttoAnalysis\\opti\\20260217\\task01_mean_cycle_Phase1_0.7ms.csv
    """
    return Path(opti_base) / f"{task_key}_mean_cycle_Phase{phase}_{speed}ms.csv"


# =============================================================================
# バッチビルダー
# =============================================================================

def build_graph_from_path(
    task_key: str,
    phase: int,
    speed: str,
    result_dir: str = r"C:\FuttoAnalysis\result",
    year: str = "2026",
    phase_shift: bool = True,
) -> FuttoGraph:
    """
    パスを自動解決して FuttoGraph を構築するショートカット。

    Parameters
    ----------
    task_key : 'task01' or 'task02'
    phase    : 1〜5
    speed    : '0.7', '0.9', '1.1', '1.3', '1.5'
    """
    g    = FuttoGraph(task_key)
    path = build_tension_csv_path(task_key, phase, speed, result_dir, year)
    g.load_from_tension_csv(path, phase_shift=phase_shift)
    return g


# =============================================================================
# __main__ テスト
# =============================================================================

if __name__ == "__main__":
    print("=== build_graph.py テスト (シミュレーションデータ) ===\n")
    for tk in ['task01', 'task02']:
        g = FuttoGraph(tk)._fill_simulated()
        print(g.summary())
        print()
