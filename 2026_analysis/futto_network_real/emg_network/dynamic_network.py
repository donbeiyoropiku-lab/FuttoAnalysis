"""
emg_network/dynamic_network.py
================================
Phase 1 (CPU環境): 相互情報量（Mutual Information）に基づく
非線形・動的筋協調ネットワーク解析

【位置づけ】
従来の解析との差別化:
  ① NMF（線形成分分解）     → 線形な「筋シナジー」を抽出
  ② ピアソン相関（correlation.py）→ 線形な「筋間相関」を評価
  ③ 本モジュール（MI）       → 非線形な「情報論的依存関係（第三の協調）」を抽出

本モジュールは連続EMG生データ（5分間など）をスライディングウィンドウで分割し、
各ウィンドウにおける16筋間の相互情報量 MI_ij を計算して時変グラフ A(t) を構築する。

Phase 2 (GPU) への接続点:
  本クラスの出力 `adjacency_sequence` (T_windows × N × N) は、
  PyTorch Geometric の Graph AutoEncoder / Graph Attention Network の
  入力グラフ列としてそのまま使用できる設計になっている。

データフロー:
  連続EMG生データ (N_ch × T_total)
       ↓  スライディングウィンドウ
  ウィンドウ列 (T_windows × N_ch × W)
       ↓  相互情報量
  時変隣接行列列 A(t) (T_windows × N_ch × N_ch)
       ↓  スパース化（閾値 or MST）
  スパース隣接行列列 A_sparse(t)
       ↓  NetworkX
  時変グラフ指標 (Global Efficiency, Modularity, Degree Centrality)
       ↓  可視化
  時系列推移グラフ + 代表ネットワーク構造図
"""

from __future__ import annotations

import numpy as np
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import sys
_HERE = Path(__file__).resolve()
# futto_network_real/ をパスに追加
sys.path.insert(0, str(_HERE.parents[1]))
# futto_common/ (2026_analysis/futto_common/CONFIG.py) があればそちらを優先
_futto_common = _HERE.parents[2] / "futto_common"
if _futto_common.exists():
    sys.path.insert(0, str(_futto_common))

try:
    import CONFIG as CFG
except ModuleNotFoundError:
    # スタンドアロン実行時: フォールバック用ダミーCFG
    class _DummyCFG:
        FS_EMG       = 2000
        MUSCLE_NAMES_BASE = ['GM','ILIO','ST','RF','VL','BF','SOL','TA']
        MUSCLE_NAMES_L    = [f'L_{m}' for m in MUSCLE_NAMES_BASE]
        MUSCLE_NAMES_R    = [f'R_{m}' for m in MUSCLE_NAMES_BASE]
        MUSCLE_NAMES      = MUSCLE_NAMES_R + MUSCLE_NAMES_L
    CFG = _DummyCFG()


# =============================================================================
# 依存ライブラリの遅延インポート（未インストール時にエラーを出さない）
# =============================================================================

def _require(pkg_name: str):
    import importlib
    try:
        return importlib.import_module(pkg_name)
    except ImportError:
        raise ImportError(
            f"'{pkg_name}' が必要です。\n"
            f"  pip install {pkg_name.replace('.', '-').split('_')[0]}"
        )


# =============================================================================
# 定数
# =============================================================================

MUSCLE_NAMES = CFG.MUSCLE_NAMES   # 16筋

# 解剖学的な左右分類（描画用）
LEFT_MUSCLES  = CFG.MUSCLE_NAMES_L   # L_GM, ..., L_TA
RIGHT_MUSCLES = CFG.MUSCLE_NAMES_R   # R_GM, ..., R_TA


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class DynamicNetworkResult:
    """
    `DynamicMuscleNetworkAnalyzer.run()` の出力コンテナ。

    全フィールドは時間軸（ウィンドウインデックス t）を第1軸とする。
    """
    n_channels    : int
    n_windows     : int
    window_size   : int
    step_size     : int
    channel_names : list[str]

    # 時変隣接行列（スパース化後）  shape (T_windows, N, N)
    adjacency_sequence : np.ndarray

    # 時変ネットワーク指標  shape (T_windows,)
    global_efficiency  : np.ndarray
    modularity         : np.ndarray

    # 各ノードの時変次数中心性  shape (T_windows, N)
    degree_centrality  : np.ndarray

    # ウィンドウの開始サンプルインデックス  shape (T_windows,)
    window_start_indices : np.ndarray

    # スパース化パラメータ（記録用）
    sparsify_method : str
    sparsify_param  : float


# =============================================================================
# メインクラス
# =============================================================================

class DynamicMuscleNetworkAnalyzer:
    """
    連続EMGデータから非線形・動的な筋協調ネットワークを構築・解析するクラス。

    Phase 1 (CPU): 相互情報量（MI）による非線形エッジ抽出
    Phase 2 (GPU): PyTorch Geometric の GAE/GAT への接続（設計済み）

    使用例:
    --------
    >>> analyzer = DynamicMuscleNetworkAnalyzer(fs=2000, window_ms=200, step_ms=100)
    >>> result = analyzer.run(emg_data)         # emg_data: shape (16, T_total)
    >>> analyzer.plot_timeseries(result)
    >>> analyzer.plot_network(result, window_idx=50)
    """

    def __init__(
        self,
        fs          : int   = CFG.FS_EMG,       # サンプリング周波数 [Hz]
        window_ms   : float = 200.0,             # ウィンドウ幅 [ms]
        step_ms     : float = 100.0,             # スライドステップ [ms]
        n_neighbors : int   = 5,                 # MI計算時のk-NN数
        n_jobs      : int   = -1,                # 並列ジョブ数（-1=全コア）
        sparsify    : Literal['threshold', 'mst', 'percentile'] = 'percentile',
        sparsify_param : float = 0.70,           # 閾値法: 0〜1の分位点 / MST: 上位パーセンタイル
        channel_names : Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        fs           : EMGサンプリング周波数 [Hz]。CONFIG.FS_EMG をデフォルトに使用。
        window_ms    : スライディングウィンドウ幅 [ms]
        step_ms      : ウィンドウのスライド量 [ms]
        n_neighbors  : 相互情報量計算のk-NN数（大きいほど滑らか・遅い）
        n_jobs       : joblib 並列数。-1=全CPU使用、1=シングル（デバッグ向け）
        sparsify     : スパース化手法
                        'percentile' : 上位 (1-sparsify_param)*100% のエッジのみ残す
                        'threshold'  : sparsify_param 以上の MI 値のみ残す
                        'mst'        : 最小全域木 + 上位パーセンタイルエッジを残す
        sparsify_param: スパース化の強度パラメータ（手法に依存）
        channel_names: 筋肉チャンネル名のリスト。None なら CONFIG.MUSCLE_NAMES を使用
        """
        self.fs          = fs
        self.window_size = int(fs * window_ms / 1000)
        self.step_size   = int(fs * step_ms / 1000)
        self.n_neighbors = n_neighbors
        self.n_jobs      = n_jobs
        self.sparsify    = sparsify
        self.sparsify_param = sparsify_param
        self.channel_names  = channel_names or list(MUSCLE_NAMES)

    # ------------------------------------------------------------------
    # メインパイプライン
    # ------------------------------------------------------------------

    def run(
        self,
        emg_data : np.ndarray,   # shape (N_channels, T_total)
        verbose  : bool = True,
    ) -> DynamicNetworkResult:
        """
        EMG連続データからネットワーク解析を一括実行する。

        Parameters
        ----------
        emg_data : np.ndarray shape (N_channels, T_total)
            正規化済みEMGデータ（各行が1筋の時系列）。
            ゼロ以上の値を推奨（MIは非負データを想定）。

        Returns
        -------
        DynamicNetworkResult
        """
        N_ch, T_total = emg_data.shape
        if verbose:
            print(f"[DynamicMuscleNetwork] 入力: {N_ch}ch × {T_total}samples")
            print(f"  window={self.window_size}pts, step={self.step_size}pts")

        # ── Step 1: スライディングウィンドウ分割 ─────────────────────
        windows, start_indices = self._sliding_windows(emg_data)
        T_win = len(windows)
        if verbose:
            print(f"  ウィンドウ数: {T_win}")

        # ── Step 2: 相互情報量による隣接行列計算 ─────────────────────
        if verbose:
            print(f"  MI計算中 ({T_win}ウィンドウ × {N_ch}×{N_ch}対) ...")
        adj_raw = self._compute_mi_sequence(windows, N_ch, verbose)

        # ── Step 3: スパース化 ────────────────────────────────────────
        if verbose:
            print(f"  スパース化 ({self.sparsify}) ...")
        adj_sparse = self._sparsify_sequence(adj_raw)

        # ── Step 4: ネットワーク指標の計算 ───────────────────────────
        if verbose:
            print(f"  ネットワーク指標を計算中 ...")
        E_t, Q_t, DC_t = self._compute_network_metrics(adj_sparse, verbose)

        if verbose:
            print(f"  完了。E_mean={E_t.mean():.4f}  Q_mean={Q_t.mean():.4f}")

        return DynamicNetworkResult(
            n_channels           = N_ch,
            n_windows            = T_win,
            window_size          = self.window_size,
            step_size            = self.step_size,
            channel_names        = self.channel_names,
            adjacency_sequence   = adj_sparse,
            global_efficiency    = E_t,
            modularity           = Q_t,
            degree_centrality    = DC_t,
            window_start_indices = start_indices,
            sparsify_method      = self.sparsify,
            sparsify_param       = self.sparsify_param,
        )

    # ------------------------------------------------------------------
    # Step 1: スライディングウィンドウ
    # ------------------------------------------------------------------

    def _sliding_windows(
        self,
        emg_data : np.ndarray,   # (N_ch, T_total)
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        データをスライディングウィンドウで分割する。

        Returns
        -------
        windows       : list of np.ndarray  各要素は shape (N_ch, window_size)
        start_indices : np.ndarray  各ウィンドウの開始サンプルインデックス
        """
        T_total = emg_data.shape[1]
        windows = []
        starts  = []

        t = 0
        while t + self.window_size <= T_total:
            windows.append(emg_data[:, t : t + self.window_size])
            starts.append(t)
            t += self.step_size

        return windows, np.array(starts, dtype=int)

    # ------------------------------------------------------------------
    # Step 2: 相互情報量による隣接行列
    # ------------------------------------------------------------------

    def _compute_mi_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        2つの1次元信号間の相互情報量を計算する。

        実装: sklearn の mutual_info_regression を使用。
        k-NNに基づく推定器（Kraskov et al., 2004）であり、
        非線形な依存関係もキャプチャできる。

        MI(X;Y) = H(X) - H(X|Y)  ≥ 0
        ピアソン相関が線形依存のみを捉えるのに対し、
        MIは任意の統計的依存関係を捉える。
        """
        from sklearn.feature_selection import mutual_info_regression
        x_2d = x.reshape(-1, 1)   # (T,) → (T, 1)
        mi   = mutual_info_regression(x_2d, y,
                                       n_neighbors=self.n_neighbors,
                                       random_state=42)[0]
        return float(mi)

    def _compute_mi_matrix(self, window: np.ndarray) -> np.ndarray:
        """
        単一ウィンドウのデータから N×N の MI 隣接行列を計算する。

        対称性（MI(X;Y) = MI(Y;X)）を利用して上三角のみ計算し、
        下三角にコピーすることで計算量を 1/2 に削減する。

        Parameters
        ----------
        window : shape (N_ch, W)

        Returns
        -------
        A : shape (N_ch, N_ch)  対称 MI 行列
        """
        N = window.shape[0]
        A = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                mi = self._compute_mi_pair(window[i], window[j])
                A[i, j] = mi
                A[j, i] = mi   # 対称

        return A

    def _compute_mi_sequence(
        self,
        windows  : list[np.ndarray],
        N_ch     : int,
        verbose  : bool,
    ) -> np.ndarray:
        """
        全ウィンドウの MI 行列を計算する（joblib による並列化）。

        joblib が使えない場合はシリアル実行にフォールバックする。

        Returns
        -------
        adj_raw : shape (T_windows, N_ch, N_ch)
        """
        T = len(windows)

        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, prefer='threads', verbose=0)(
                delayed(self._compute_mi_matrix)(w) for w in windows
            )
            adj_raw = np.stack(results, axis=0)

        except Exception as e:
            if verbose:
                print(f"  [警告] joblib 並列処理失敗 ({e})。シリアル実行に切り替えます。")
            adj_raw = np.zeros((T, N_ch, N_ch))
            for t, w in enumerate(windows):
                adj_raw[t] = self._compute_mi_matrix(w)
                if verbose and (t + 1) % max(1, T // 5) == 0:
                    print(f"    {t+1}/{T} ウィンドウ完了")

        return adj_raw

    # ------------------------------------------------------------------
    # Step 3: スパース化
    # ------------------------------------------------------------------

    def _sparsify_sequence(self, adj_raw: np.ndarray) -> np.ndarray:
        """
        MI 隣接行列列をスパース化する。

        手法:
          'percentile': 各ウィンドウで上位 (1-p)*100% のエッジのみ残す
          'threshold' : MI値がsparsify_param以上のエッジのみ残す
          'mst'       : 最小全域木 + 上位パーセンタイルエッジを保持
        """
        T, N, _ = adj_raw.shape
        adj_out = np.zeros_like(adj_raw)

        for t in range(T):
            A = adj_raw[t].copy()
            np.fill_diagonal(A, 0)

            if self.sparsify == 'percentile':
                # 上位 (1-sparsify_param)*100 % のエッジを残す
                off_diag = A[np.triu_indices(N, k=1)]
                thresh   = np.quantile(off_diag, self.sparsify_param)
                A_sparse = np.where(A >= thresh, A, 0.0)

            elif self.sparsify == 'threshold':
                A_sparse = np.where(A >= self.sparsify_param, A, 0.0)

            elif self.sparsify == 'mst':
                # 最小全域木（距離 = 1/MI）で必須エッジを確保
                # + 上位パーセンタイルでさらにエッジを追加
                A_sparse = self._mst_sparsify(A, N)

            else:
                A_sparse = A

            np.fill_diagonal(A_sparse, 0)
            adj_out[t] = A_sparse

        return adj_out

    def _mst_sparsify(self, A: np.ndarray, N: int) -> np.ndarray:
        """
        最小全域木（MST）ベースのスパース化。

        距離行列 D = 1 / (MI + ε) に対して Prim 法で MST を構築し、
        MST エッジ + 上位 30% の高 MI エッジを保持する。
        """
        eps = 1e-9
        # 距離行列
        with np.errstate(divide='ignore'):
            D = np.where(A > eps, 1.0 / A, 1.0 / eps)
        np.fill_diagonal(D, 0)

        # Prim 法で MST を構築
        in_tree   = np.zeros(N, dtype=bool)
        min_edge  = np.full(N, np.inf)
        parent    = np.full(N, -1, dtype=int)
        min_edge[0] = 0

        mst_mask = np.zeros((N, N), dtype=bool)
        for _ in range(N):
            u = np.argmin(np.where(~in_tree, min_edge, np.inf))
            in_tree[u] = True
            if parent[u] >= 0:
                mst_mask[u, parent[u]] = True
                mst_mask[parent[u], u] = True

            for v in range(N):
                if not in_tree[v] and D[u, v] < min_edge[v]:
                    min_edge[v] = D[u, v]
                    parent[v]   = u

        # 上位パーセンタイルエッジも追加
        off_diag = A[np.triu_indices(N, k=1)]
        thresh   = np.quantile(off_diag, self.sparsify_param)
        hi_mask  = A >= thresh

        A_sparse = np.where(mst_mask | hi_mask, A, 0.0)
        return A_sparse

    # ------------------------------------------------------------------
    # Step 4: ネットワーク指標
    # ------------------------------------------------------------------

    def _compute_network_metrics(
        self,
        adj_sparse : np.ndarray,   # (T, N, N)
        verbose    : bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スパース化された隣接行列列から各時刻のネットワーク指標を算出する。

        使用指標:
          Global Efficiency: E = 1/(N(N-1)) * Σ_{i≠j} 1/d_ij
            d_ij = NetworkX の weighted_shortest_path (weight=1/MI)
          Modularity Q: Louvain アルゴリズムによるコミュニティ最適化
          Degree Centrality: 重み付き次数を最大値で正規化

        Returns
        -------
        E_t  : shape (T,)
        Q_t  : shape (T,)
        DC_t : shape (T, N)
        """
        import networkx as nx

        T, N, _ = adj_sparse.shape
        E_t  = np.zeros(T)
        Q_t  = np.zeros(T)
        DC_t = np.zeros((T, N))

        for t in range(T):
            A  = adj_sparse[t]
            G  = self._adj_to_networkx(A)

            # Global Efficiency
            E_t[t] = self._global_efficiency_nx(G)

            # Modularity（Louvain）
            Q_t[t] = self._louvain_modularity(G)

            # Degree Centrality（重み付き次数を正規化）
            wd_dict = dict(G.degree(weight='weight'))
            wd_arr  = np.array([wd_dict.get(i, 0.0) for i in range(N)])
            wd_max  = wd_arr.max()
            DC_t[t] = wd_arr / (wd_max + 1e-12)

            if verbose and (t + 1) % max(1, T // 5) == 0:
                print(f"    指標計算 {t+1}/{T}")

        return E_t, Q_t, DC_t

    def _adj_to_networkx(self, A: np.ndarray):
        """
        隣接行列 → NetworkX 重み付き無向グラフ に変換する。
        エッジ属性 'weight' = MI 値。
        """
        import networkx as nx
        N = A.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] > 0:
                    G.add_edge(i, j, weight=float(A[i, j]))
        return G

    def _global_efficiency_nx(self, G) -> float:
        """
        Global Efficiency を NetworkX で計算する。
        距離 = 1 / weight として最短経路を求める。

        E = 1/(N(N-1)) * Σ_{i≠j} 1/d_ij
        """
        import networkx as nx
        N = G.number_of_nodes()
        if N <= 1:
            return 0.0

        # エッジ属性 'distance' = 1/MI を追加
        G_dist = G.copy()
        for u, v, data in G_dist.edges(data=True):
            G_dist[u][v]['distance'] = 1.0 / (data['weight'] + 1e-12)

        total_inv = 0.0
        for node in G_dist.nodes():
            lengths = nx.single_source_dijkstra_path_length(
                G_dist, node, weight='distance'
            )
            for other, d in lengths.items():
                if other != node and d > 0:
                    total_inv += 1.0 / d

        return float(total_inv / (N * (N - 1)))

    def _louvain_modularity(self, G) -> float:
        """
        Louvain 法によるコミュニティ分割とモジュラリティ Q を計算する。

        networkx-community パッケージが利用できる場合はそちらを使用。
        なければスペクトル 2 分割で近似する。

        Q = (1/2m) * Σ_{ij} [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
        m: 総エッジ重み, k_i: 重み付き次数
        """
        import networkx as nx

        if G.number_of_edges() == 0:
            return 0.0

        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G, weight='weight')
            Q = nx.algorithms.community.quality.modularity(
                G, communities, weight='weight'
            )
            return float(max(Q, 0.0))
        except Exception:
            pass

        # フォールバック: スペクトル 2 分割
        N = G.number_of_nodes()
        if N < 4:
            return 0.0
        A = nx.to_numpy_array(G, weight='weight')
        k = A.sum(axis=1)
        m = k.sum() / 2.0
        if m == 0:
            return 0.0
        D = np.diag(k)
        L = D - A
        evals, evecs = np.linalg.eigh(L)
        fiedler = evecs[:, 1]
        labels  = (fiedler >= 0).astype(int)
        Q = sum(
            (A[i, j] - k[i] * k[j] / (2 * m))
            for i in range(N) for j in range(N)
            if labels[i] == labels[j]
        ) / (2 * m)
        return float(max(Q, 0.0))

    # ------------------------------------------------------------------
    # 可視化
    # ------------------------------------------------------------------

    def plot_timeseries(
        self,
        result   : DynamicNetworkResult,
        save_path: Optional[str | Path] = None,
        task_key : str = "",
        phase    : int = 0,
        speed    : str = "",
    ) -> None:
        """
        時系列推移プロット。

        上段: Global Efficiency（青）
        中段: Modularity Q（橙）
        下段: 上位3筋の Degree Centrality（ハブ筋の動的追跡）

        横軸: 時刻（ウィンドウインデックス / 実時間[s]）
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        T    = result.n_windows
        t_sec = result.window_start_indices / self.fs   # 実時間[s]

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        # ── 上段: Global Efficiency ──────────────────────────────────
        ax = axes[0]
        ax.plot(t_sec, result.global_efficiency,
                color='#1f77b4', linewidth=2, label='Global Efficiency')
        ax.fill_between(t_sec, result.global_efficiency,
                        alpha=0.2, color='#1f77b4')
        ax.set_ylabel('Global Efficiency $E$', fontsize=10)
        ax.set_title(
            f'Dynamic Muscle Network — MI-based\n'
            f'{task_key}  Phase{phase}  {speed}m/s'
            if task_key else 'Dynamic Muscle Network (MI-based)',
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── 中段: Modularity ─────────────────────────────────────────
        ax2 = axes[1]
        ax2.plot(t_sec, result.modularity,
                 color='#ff7f0e', linewidth=2, label='Modularity Q')
        ax2.fill_between(t_sec, result.modularity,
                         alpha=0.2, color='#ff7f0e')
        ax2.set_ylabel('Modularity $Q$', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ── 下段: ハブ筋 Degree Centrality ──────────────────────────
        ax3    = axes[2]
        N      = result.n_channels
        dc_mean= result.degree_centrality.mean(axis=0)     # 平均次数
        top3   = np.argsort(dc_mean)[::-1][:3]
        colors = ['#e74c3c', '#2ecc71', '#9b59b6']
        for ci, mi_idx in enumerate(top3):
            name = result.channel_names[mi_idx] if mi_idx < len(result.channel_names) else str(mi_idx)
            ax3.plot(t_sec, result.degree_centrality[:, mi_idx],
                     color=colors[ci], linewidth=1.8, label=f'{name}', alpha=0.85)

        ax3.set_xlabel('Time [s]', fontsize=10)
        ax3.set_ylabel('Degree Centrality', fontsize=10)
        ax3.set_title('Top-3 Hub Muscles (by mean Degree Centrality)')
        ax3.legend(fontsize=8, ncol=3)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Dynamic] 時系列グラフ保存 → {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_network(
        self,
        result      : DynamicNetworkResult,
        window_idx  : int = 0,
        save_path   : Optional[str | Path] = None,
        title_extra : str = "",
    ) -> None:
        """
        指定ウィンドウにおけるネットワーク構造図を描画する。

        ノード配置:
          右脚筋（R_GM〜R_TA）: 上半円
          左脚筋（L_GM〜L_TA）: 下半円
        ノードサイズ: Degree Centrality に比例
        エッジ幅    : MI値（重み）に比例
        エッジ色    : 同側=グレー、対側=紫（対側性の可視化）
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx

        A   = result.adjacency_sequence[window_idx]
        dc  = result.degree_centrality[window_idx]
        ch  = result.channel_names
        N   = len(ch)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.axis('off')

        # ── ノード配置（左右半円）───────────────────────────────────
        angles  = np.zeros(N)
        n_right = sum(1 for c in ch if c.startswith('R_'))
        n_left  = N - n_right

        right_idx = [i for i, c in enumerate(ch) if c.startswith('R_')]
        left_idx  = [i for i, c in enumerate(ch) if c.startswith('L_')]

        for rank, idx in enumerate(right_idx):
            angles[idx] = np.pi - rank * np.pi / max(len(right_idx) - 1, 1)
        for rank, idx in enumerate(left_idx):
            angles[idx] = -rank * np.pi / max(len(left_idx) - 1, 1)

        R = 1.0
        xs = R * np.cos(angles)
        ys = R * np.sin(angles)

        # ── エッジ描画 ────────────────────────────────────────────────
        w_max = A.max() + 1e-9
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] <= 0:
                    continue
                lw    = 0.5 + (A[i, j] / w_max) * 5.0
                alpha = 0.2 + (A[i, j] / w_max) * 0.6
                # 同側か対側か
                same_side = (ch[i][0] == ch[j][0])
                color = '#95a5a6' if same_side else '#9b59b6'
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        color=color, linewidth=lw, alpha=alpha, zorder=1)

        # ── ノード描画 ─────────────────────────────────────────────────
        dc_max = dc.max() + 1e-9
        for i in range(N):
            side  = 'R' if ch[i].startswith('R_') else 'L'
            color = '#3498db' if side == 'R' else '#e74c3c'
            size  = 80 + (dc[i] / dc_max) * 400
            ax.scatter(xs[i], ys[i], s=size, c=color, zorder=5,
                       edgecolors='white', linewidths=1.5, alpha=0.9)

            label_r = 1.18
            ha = 'left' if xs[i] >= 0 else 'right'
            ax.text(label_r * xs[i], label_r * ys[i],
                    ch[i].replace('_', '\n'),
                    ha=ha, va='center', fontsize=7, fontweight='bold', color=color)

        # ── 中央情報 ──────────────────────────────────────────────────
        t_sec = result.window_start_indices[window_idx] / self.fs
        ax.text(0, 0,
                f"t = {t_sec:.2f}s\n"
                f"E = {result.global_efficiency[window_idx]:.3f}\n"
                f"Q = {result.modularity[window_idx]:.3f}",
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

        # ── 凡例 ──────────────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(color='#3498db', label='Right leg'),
            mpatches.Patch(color='#e74c3c', label='Left leg'),
            plt.Line2D([0], [0], color='#9b59b6', linewidth=2,
                       label='Contralateral edge'),
            plt.Line2D([0], [0], color='#95a5a6', linewidth=2,
                       label='Ipsilateral edge'),
        ]
        ax.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)

        title = (f'Muscle Network (MI)  window={window_idx}  '
                 f'{title_extra}')
        ax.set_title(title, fontsize=11, pad=15)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Dynamic] ネットワーク図保存 → {save_path}")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Phase 2 接続インターフェース（GPU/GNN 用）
    # ------------------------------------------------------------------

    def export_for_gnn(
        self,
        result      : DynamicNetworkResult,
        out_path    : str | Path,
        emg_data    : np.ndarray = None,   # shape (N_ch, T_total) 連続EMGデータ
    ) -> Path:
        """
        Phase 2 (PyTorch Geometric) 用のデータをエクスポートする。

        【指摘反映版】ノード特徴量として使うべき「EMGウィンドウ」を
        .npz に同梱する。Phase 2 で adjacency_to_pyg() が正しい特徴量
        (shape: N, W) を使えるようになる。

        出力形式: NumPy .npz ファイル
          'adjacency'         : shape (T_windows, N, N)  スパース化済み MI 行列
          'emg_windows'       : shape (T_windows, N, W)  ★各ウィンドウのEMG波形
                                  W = window_size（ノード特徴量として使用）
          'global_efficiency' : shape (T_windows,)
          'modularity'        : shape (T_windows,)
          'degree_centrality' : shape (T_windows, N)
          'window_starts'     : shape (T_windows,)  開始サンプルインデックス
          'channel_names'     : shape (N,)

        PyTorch Geometric での読み込み例:
            data = np.load('for_gnn.npz', allow_pickle=True)
            adj         = data['adjacency']     # (T, N, N)
            emg_windows = data['emg_windows']   # (T, N, W) ← ノード特徴量
            # → adjacency_to_pyg(adj[t], emg_windows[t]) で PyG Data に変換

        Parameters
        ----------
        result   : DynamicNetworkResult
        out_path : 出力ファイルパス (.npz)
        emg_data : 元の連続EMGデータ shape (N_ch, T_total)。
                   指定するとウィンドウ切り出しを行い emg_windows を作成する。
                   None の場合は adjacency の行ベクトルで代替（精度低下）。
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # EMGウィンドウを切り出す
        if emg_data is not None:
            windows, _ = self._sliding_windows(emg_data)
            # windows: list of (N_ch, W) → stack → (T, N, W)
            emg_windows = np.stack(windows, axis=0)
        else:
            # 代替: 隣接行列の行ベクトルを特徴量として使用（精度低下注意）
            emg_windows = result.adjacency_sequence
            print("  [警告] emg_data が指定されていないため、隣接行列の行ベクトルを"
                  "ノード特徴量として代替します。")
            print("         精度向上のため emg_data=<連続EMGデータ> を渡してください。")

        np.savez_compressed(
            str(out_path),
            adjacency         = result.adjacency_sequence,
            emg_windows       = emg_windows,
            global_efficiency = result.global_efficiency,
            modularity        = result.modularity,
            degree_centrality = result.degree_centrality,
            window_starts     = result.window_start_indices,
            channel_names     = np.array(result.channel_names),
        )
        print(f"  [GNN Export] 保存 → {out_path}")
        print(f"    adjacency  : {result.adjacency_sequence.shape}")
        print(f"    emg_windows: {emg_windows.shape}  ← ノード特徴量")
        return out_path

    # ------------------------------------------------------------------
    # CSV 保存
    # ------------------------------------------------------------------

    def save_metrics_csv(
        self,
        result   : DynamicNetworkResult,
        out_dir  : str | Path,
        prefix   : str = "dynamic_network",
    ) -> None:
        """
        ネットワーク指標の時系列を CSV に保存する。
        """
        import pandas as pd
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        t_sec = result.window_start_indices / self.fs

        # 時系列指標
        df_ts = pd.DataFrame({
            'window_idx'       : range(result.n_windows),
            'time_sec'         : t_sec,
            'global_efficiency': result.global_efficiency,
            'modularity_Q'     : result.modularity,
        })
        df_ts.to_csv(out_dir / f"{prefix}_timeseries.csv",
                     index=False, float_format='%.6f')

        # 次数中心性
        dc_df = pd.DataFrame(
            result.degree_centrality,
            columns=result.channel_names,
        )
        dc_df.insert(0, 'window_idx', range(result.n_windows))
        dc_df.insert(1, 'time_sec', t_sec)
        dc_df.to_csv(out_dir / f"{prefix}_degree_centrality.csv",
                     index=False, float_format='%.4f')

        print(f"  [Dynamic CSV] 保存 → {out_dir}")


# =============================================================================
# ダミーEMGデータ生成（テスト用）
# =============================================================================

def generate_dummy_emg(
    n_channels  : int   = 16,
    n_samples   : int   = 10000,
    fs          : int   = 2000,
    seed        : int   = 42,
) -> np.ndarray:
    """
    非線形な依存関係を含むダミー EMG データを生成する。

    設計:
      - 左脚8筋: 正弦波 + ランダムノイズ
      - 右脚8筋: 左脚の50%位相遅れ + 二乗変調（非線形依存） + ノイズ
        → ピアソン相関では捉えられず、MIのみで検出できる関係

    Returns
    -------
    emg : shape (n_channels, n_samples)  非負、0〜1正規化済み
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, n_samples / fs, n_samples)
    emg = np.zeros((n_channels, n_samples))

    # 基本周波数（歩行リズム: ~1Hz）
    f_gait = 1.0

    # 左脚 (index 8〜15)
    for k in range(8):
        phase = rng.uniform(0, np.pi)
        amp   = rng.uniform(0.4, 0.9)
        emg[8 + k] = np.clip(
            amp * (np.sin(2 * np.pi * f_gait * t + phase) + 1) / 2
            + rng.normal(0, 0.05, n_samples),
            0, 1
        )

    # 右脚 (index 0〜7): 左脚を50%位相シフト + 二乗変調（非線形）
    shift = n_samples // 2
    for k in range(8):
        base    = np.roll(emg[8 + k], shift)
        nonlin  = base ** 2 * 0.5   # 非線形変調（ピアソン相関では検出困難）
        emg[k] = np.clip(
            base * 0.6 + nonlin + rng.normal(0, 0.08, n_samples),
            0, 1
        )

    return emg


# =============================================================================
# __main__: デモ実行
# =============================================================================

if __name__ == "__main__":
    import time
    print("=" * 60)
    print(" Dynamic Muscle Network Analyzer — デモ実行")
    print("=" * 60)

    # ── ダミーデータ生成 ─────────────────────────────────────────
    print("\n[1/4] ダミーEMGデータ生成 (16ch × 10000samples @ 2000Hz) ...")
    emg_dummy = generate_dummy_emg(n_channels=16, n_samples=10000, fs=2000)
    print(f"  shape: {emg_dummy.shape}  range: [{emg_dummy.min():.3f}, {emg_dummy.max():.3f}]")

    # ── アナライザー初期化 ────────────────────────────────────────
    print("\n[2/4] アナライザー初期化 ...")
    analyzer = DynamicMuscleNetworkAnalyzer(
        fs            = 2000,
        window_ms     = 200,     # 200ms ウィンドウ（= 400サンプル）
        step_ms       = 100,     # 100ms ステップ
        n_neighbors   = 5,
        n_jobs        = 1,       # デモはシリアル実行
        sparsify      = 'percentile',
        sparsify_param= 0.70,    # 上位30%のエッジを保持
    )
    print(f"  window_size={analyzer.window_size}pts, step={analyzer.step_size}pts")

    # ── 解析実行 ──────────────────────────────────────────────────
    print("\n[3/4] ネットワーク解析実行 ...")
    t0     = time.time()
    result = analyzer.run(emg_dummy, verbose=True)
    elapsed= time.time() - t0
    print(f"\n  完了: {elapsed:.1f}秒")
    print(f"  ウィンドウ数: {result.n_windows}")
    print(f"  E_mean={result.global_efficiency.mean():.4f}  "
          f"E_std={result.global_efficiency.std():.4f}")
    print(f"  Q_mean={result.modularity.mean():.4f}  "
          f"Q_std={result.modularity.std():.4f}")
    dc_mean = result.degree_centrality.mean(axis=0)
    top3_idx = np.argsort(dc_mean)[::-1][:3]
    print(f"  Top-3 ハブ筋: {[result.channel_names[i] for i in top3_idx]}")

    # ── 可視化 ────────────────────────────────────────────────────
    print("\n[4/4] グラフ出力 ...")
    out_dir = Path("./network_results/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer.plot_timeseries(result, save_path=out_dir / "timeseries_demo.png")
    analyzer.plot_network(result, window_idx=result.n_windows // 2,
                          save_path=out_dir / "network_demo.png")
    analyzer.save_metrics_csv(result, out_dir, prefix="demo")
    analyzer.export_for_gnn(result, out_dir / "demo_for_gnn.npz")

    print(f"\n=== デモ完了。結果: {out_dir.resolve()} ===")
