"""
colab/phase2_gnn_muscle_network.py
=====================================
Phase 2 (GPU / Google Colab): Graph Neural Network による非線形筋協調の深層抽出

【このスクリプトの位置づけ】
Phase 1 (dynamic_network.py) で構築した「MI ベースの時変グラフ列」を入力とし、
Graph AutoEncoder (GAE) と Graph Attention Network (GAT) を用いて
「データが自身に隠した未知の協調構造」を抽出する。

【使い方（Google Colab）】
  1. Phase 1 の出力 (demo_for_gnn.npz) を Colab にアップロード
  2. 以下のセルで実行:

     !pip install torch torch-geometric torch-scatter torch-sparse -q
     %run phase2_gnn_muscle_network.py --npz demo_for_gnn.npz

【参考文献】
  - Kipf & Welling, 2016: Graph Convolutional Networks (GCN)
  - Veličković et al., 2018: Graph Attention Networks (GAT)
  - Kipf & Welling, 2016: Variational Graph AutoEncoders (VGAE)
"""

# =============================================================================
# Colab インストールコマンド（コメント）
# =============================================================================
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install torch-geometric
# !pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


# =============================================================================
# GPU 環境チェック
# =============================================================================

def check_environment() -> str:
    """利用可能なデバイスを確認して返す。"""
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[環境] PyTorch {torch.__version__}  device={device}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return device
    except ImportError:
        print("[警告] PyTorch が見つかりません。CPU モードで動作します。")
        return 'cpu'


# =============================================================================
# データ読み込み
# =============================================================================

def load_phase1_data(npz_path: str | Path) -> dict:
    """
    Phase 1 の export_for_gnn() で作成した .npz を読み込む。

    Returns
    -------
    dict with keys:
      'adjacency'         : shape (T, N, N)
      'global_efficiency' : shape (T,)
      'modularity'        : shape (T,)
      'degree_centrality' : shape (T, N)
      'channel_names'     : list[str]
    """
    data = np.load(str(npz_path), allow_pickle=True)
    print(f"[データ] {npz_path}")
    print(f"  adjacency: {data['adjacency'].shape}")
    return {
        'adjacency'         : data['adjacency'],
        'global_efficiency' : data['global_efficiency'],
        'modularity'        : data['modularity'],
        'degree_centrality' : data['degree_centrality'],
        'channel_names'     : list(data['channel_names']),
    }


# =============================================================================
# 時間ベース分割（Data Leakage 防止）
# =============================================================================

def time_based_split(
    data_list   : list,
    labels      : np.ndarray,
    train_ratio : float = 0.67,
    val_ratio   : float = 0.17,
) -> tuple[list, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """
    グラフリストを時間軸でスパッと分割する（ランダムシャッフル禁止）。

    深層学習の鉄則: 時系列データのランダム分割は「データリーク」を引き起こす。
    前の時刻のウィンドウと後の時刻のウィンドウが学習・テストに混在すると、
    AIが「未来のデータをカンニング」した状態になり、精度が誇大評価される。

    分割例（デフォルト）:
      Train (67%): 最初の区間          ← 学習に使用
      Val   (17%): 次の区間            ← ハイパーパラメータ調整
      Test  (16%): 最後の区間          ← 最終評価（学習中は一切触れない）

    Parameters
    ----------
    data_list   : 全グラフのリスト（時間順に並んでいること）
    labels      : 全ラベルの配列
    train_ratio : 学習データの割合
    val_ratio   : 検証データの割合

    Returns
    -------
    train_data, val_data, test_data,
    train_labels, val_labels, test_labels
    """
    T       = len(data_list)
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)

    train_data   = data_list[:n_train]
    val_data     = data_list[n_train:n_train + n_val]
    test_data    = data_list[n_train + n_val:]
    train_labels = labels[:n_train]
    val_labels   = labels[n_train:n_train + n_val]
    test_labels  = labels[n_train + n_val:]

    print(f"  データ分割（時間軸）: Train={len(train_data)}  "
          f"Val={len(val_data)}  Test={len(test_data)}")
    return (train_data, val_data, test_data,
            train_labels, val_labels, test_labels)


# =============================================================================
# PyG データ変換
# =============================================================================

def adjacency_to_pyg(
    A         : np.ndarray,   # shape (N, N)  スパース化済みMI行列
    emg_window: np.ndarray,   # shape (N, W)  生EMGウィンドウ（ノード特徴）
    device    : str = 'cpu',
):
    """
    密な隣接行列 + 生EMGウィンドウ → PyTorch Geometric の Data オブジェクト。

    【ノード特徴量の設計（指摘反映版）】
    ✗ 旧版: x = A（隣接行ベクトル）→ 自己参照的で情報が循環する
    ✓ 新版: x = emg_window（各筋のEMG波形スライス）
      → ノードiの特徴量 = 筋肉iのウィンドウ内EMG波形（W次元）
      → GCNはこの特徴量を隣接関係を通じて伝搬させ、
         「どの筋が協調しているか」を自律的に学習できる

    Returns
    -------
    torch_geometric.data.Data
      x           : shape (N, W)   各ノード（筋）のEMG波形特徴
      edge_index  : shape (2, E)
      edge_attr   : shape (E,)     エッジ重み（MI値）
    """
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    A_t  = torch.tensor(A, dtype=torch.float32)
    ei, ew = dense_to_sparse(A_t)

    # ノード特徴 = 各筋のEMG波形（W次元ベクトル）
    x = torch.tensor(emg_window, dtype=torch.float32)   # shape (N, W)
    return Data(x=x, edge_index=ei, edge_attr=ew).to(device)


# =============================================================================
# Graph AutoEncoder (GAE / VGAE)
# =============================================================================

class MuscleGAE:
    """
    Graph AutoEncoder による潜在的エッジ構造の抽出。

    エンコーダ: GCN (2層) → 潜在表現 Z
    デコーダ  : Z @ Z^T   → 再構成隣接行列 A_hat

    学習目的: A_hat と A_original の差（再構成誤差）を最小化することで、
             GCN がグラフ構造に潜む隠れた協調パターンを学習する。

    「データが自身に隠した未知の協調リンク」= 再構成で現れるが
    MI 行列にはなかった非ゼロエッジ
    """

    def __init__(self, n_features: int, hidden_dim: int = 32, latent_dim: int = 16):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GCNConv

            class GCNEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(n_features, hidden_dim)
                    self.conv2 = GCNConv(hidden_dim, latent_dim)
                    self.relu  = nn.ReLU()

                def forward(self, x, edge_index, edge_weight=None):
                    x = self.relu(self.conv1(x, edge_index, edge_weight))
                    z = self.conv2(x, edge_index, edge_weight)
                    return z

            self.encoder = GCNEncoder()
            self.latent_dim = latent_dim
            self._torch = torch
            self._nn    = nn

        except ImportError:
            raise ImportError(
                "torch と torch-geometric が必要です。\n"
                "Google Colab で: !pip install torch torch-geometric"
            )

    def train(
        self,
        data_list   : list,    # list of torch_geometric.data.Data
        n_epochs    : int = 100,
        lr          : float = 1e-3,
        alpha_l1    : float = 1e-3,   # L1正則化強度（スパース化）
        device      : str = 'cpu',
    ) -> list[float]:
        """
        GAE を学習する。

        損失関数: MSE(A_hat, A_original) + alpha_l1 * ||A_hat||_1
          MSE 項  : 再構成の正確さを保証
          L1 正則化: 重要なエッジだけを残し、密すぎる再構成を防ぐ
                     alpha_l1 を大きくするほどスパースになる
        オプティマイザ: Adam
        """
        torch = self._torch
        self.encoder = self.encoder.to(device)
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        losses = []

        self.encoder.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for data in data_list:
                optimizer.zero_grad()
                z     = self.encoder(data.x, data.edge_index, data.edge_attr)
                A_hat = torch.sigmoid(z @ z.T)   # 再構成隣接行列

                # 正解: 密な隣接行列
                from torch_geometric.utils import to_dense_adj
                A_true = to_dense_adj(data.edge_index,
                                      edge_attr=data.edge_attr,
                                      max_num_nodes=data.num_nodes)[0]
                # 再構成損失（MSE）+ L1正則化（スパース化）
                # L1: エッジが少ない（スパースな）再構成を促す
                # alpha_l1（デフォルト 1e-3）でスパース度を制御
                recon_loss = torch.nn.functional.mse_loss(A_hat, A_true)
                l1_loss    = alpha_l1 * A_hat.abs().mean()
                loss       = recon_loss + l1_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(data_list)
            losses.append(epoch_loss)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}  Loss={epoch_loss:.6f}")

        return losses

    def get_latent(
        self,
        data_list : list,
        device    : str = 'cpu',
    ) -> np.ndarray:
        """
        学習済みエンコーダで潜在表現 Z を取得する。

        Returns
        -------
        Z_all : shape (T_windows, N, latent_dim)
        """
        torch = self._torch
        self.encoder.eval()
        Z_list = []
        with torch.no_grad():
            for data in data_list:
                z = self.encoder(data.x, data.edge_index, data.edge_attr)
                Z_list.append(z.cpu().numpy())
        return np.stack(Z_list, axis=0)

    def reconstruct(
        self,
        data_list : list,
        device    : str = 'cpu',
    ) -> np.ndarray:
        """
        再構成隣接行列 A_hat を返す。

        A_hat[t, i, j] > threshold かつ A_original[t, i, j] == 0
        → GAE が発見した「潜在的な未知の協調リンク」

        Returns
        -------
        A_hat_all : shape (T_windows, N, N)
        """
        import torch
        self.encoder.eval()
        A_hats = []
        with torch.no_grad():
            for data in data_list:
                z     = self.encoder(data.x, data.edge_index, data.edge_attr)
                A_hat = torch.sigmoid(z @ z.T).cpu().numpy()
                A_hats.append(A_hat)
        return np.stack(A_hats, axis=0)


# =============================================================================
# Graph Attention Network (GAT) 分類器
# =============================================================================

class MuscleGAT:
    """
    Graph Attention Network による動的アテンションの抽出。

    タスク: Task01（Futto着用）vs Task03（非装着）を分類する。
    副産物: 各エッジのアテンション重み α_ij
            → 「どの筋が、どの筋に注意を向けているか」を抽出
            → 特定の運動フェーズで重要な非線形協調関係の証拠

    使用例:
      gat = MuscleGAT(n_features, n_classes=2)
      gat.train(train_graphs, labels)
      attentions = gat.get_attention(test_graphs)
    """

    def __init__(
        self,
        n_features  : int,
        n_classes   : int = 2,
        hidden_dim  : int = 32,
        n_heads     : int = 4,
    ):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GATConv, global_mean_pool

            class GATClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gat1 = GATConv(n_features, hidden_dim,
                                        heads=n_heads, dropout=0.3)
                    self.gat2 = GATConv(hidden_dim * n_heads, 16,
                                        heads=1, concat=False, dropout=0.3)
                    self.fc   = nn.Linear(16, n_classes)
                    self.relu = nn.ELU()
                    self.drop = nn.Dropout(0.3)

                def forward(self, x, edge_index, batch):
                    x, alpha1 = self.gat1(x, edge_index,
                                          return_attention_weights=True)
                    x         = self.relu(self.drop(x))
                    x, alpha2 = self.gat2(x, alpha1[0],
                                          return_attention_weights=True)
                    x         = global_mean_pool(x, batch)
                    return self.fc(x), (alpha1, alpha2)

            self.model = GATClassifier()
            self._torch = torch

        except ImportError:
            raise ImportError(
                "torch と torch-geometric が必要です。\n"
                "!pip install torch torch-geometric"
            )

    def train(
        self,
        data_list : list,
        labels    : np.ndarray,
        n_epochs  : int = 100,
        lr        : float = 5e-4,
        device    : str = 'cpu',
    ) -> list[float]:
        """
        GAT 分類器を学習する。

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
        labels    : shape (T,)  0=Task03, 1=Task01/02
        """
        import torch
        from torch_geometric.data import DataLoader

        # ラベルをデータに付加
        for data, lbl in zip(data_list, labels):
            data.y = torch.tensor([int(lbl)], dtype=torch.long)

        loader   = DataLoader(data_list, batch_size=16, shuffle=True)
        opt      = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit     = torch.nn.CrossEntropyLoss()
        self.model = self.model.to(device)

        losses = []
        self.model.train()
        for epoch in range(n_epochs):
            ep_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss   = crit(out, batch.y)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            losses.append(ep_loss / len(loader))
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}  Loss={losses[-1]:.4f}")

        return losses

    def get_attention(
        self,
        data_list : list,
        device    : str = 'cpu',
    ) -> np.ndarray:
        """
        各グラフ・各エッジのアテンション重みを返す。

        Returns
        -------
        attn_matrices : shape (T_windows, N, N)  アテンション行列
        """
        import torch
        from torch_geometric.utils import to_dense_adj
        self.model.eval()
        attn_list = []

        with torch.no_grad():
            for data in data_list:
                data = data.to(device)
                from torch_geometric.data import Batch
                batch = Batch.from_data_list([data])
                _, (alpha1, alpha2) = self.model(
                    batch.x, batch.edge_index, batch.batch
                )
                # alpha1 = (edge_index, attention_weights)
                ei, aw = alpha1
                # 平均アテンション（マルチヘッド）
                aw_mean = aw.mean(dim=-1)   # (E,)
                A_attn  = to_dense_adj(
                    ei, edge_attr=aw_mean,
                    max_num_nodes=data.num_nodes,
                )[0].cpu().numpy()
                attn_list.append(A_attn)

        return np.stack(attn_list, axis=0)


# =============================================================================
# 結果比較・可視化
# =============================================================================

def compare_mi_vs_gae(
    adj_mi  : np.ndarray,    # Phase 1 の MI 行列  shape (T, N, N)
    adj_gae : np.ndarray,    # GAE 再構成行列     shape (T, N, N)
    channel_names : list[str],
    threshold     : float = 0.5,
    save_path     : str | Path = None,
) -> None:
    """
    MI 行列と GAE 再構成行列を比較し、
    「GAE が新たに発見した潜在的協調リンク」を可視化する。

    GAE New Link: adj_gae[t,i,j] > threshold かつ adj_mi[t,i,j] == 0
    → MI では検出できなかった非線形な「隠れたつながり」
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 代表時刻（中間点）のスナップショット
    T = adj_mi.shape[0]
    t = T // 2

    A_mi   = adj_mi[t]
    A_gae  = (adj_gae[t] > threshold).astype(float)
    A_new  = (A_gae - (A_mi > 0).astype(float)).clip(0)   # GAE のみのリンク

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    N = A_mi.shape[0]
    ch = channel_names

    for ax, data, title, cmap in zip(
        axes,
        [A_mi, A_gae, A_new],
        ['MI Network (Phase 1)', 'GAE Reconstruction (Phase 2)', 'GAE New Links (未知の協調)'],
        ['Blues', 'Greens', 'Reds'],
    ):
        im = ax.imshow(data, cmap=cmap, aspect='equal',
                       vmin=0, interpolation='nearest')
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(ch, rotation=90, fontsize=6)
        ax.set_yticklabels(ch, fontsize=6)
        ax.axhline(7.5, color='k', linewidth=1)
        ax.axvline(7.5, color='k', linewidth=1)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle('Phase 1 (MI) vs Phase 2 (GAE) — Muscle Network Comparison',
                 fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [比較図] 保存 → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_attention_heatmap(
    attn_matrices  : np.ndarray,    # shape (T, N, N)
    channel_names  : list[str],
    save_path      : str | Path = None,
) -> None:
    """
    GAT のアテンション重みを歩行周期にわたって平均し、
    「どの筋が、どの筋に最も注意を向けているか」を
    ヒートマップで可視化する。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    attn_mean = attn_matrices.mean(axis=0)   # (N, N) 時間平均
    N  = attn_mean.shape[0]
    ch = channel_names

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(attn_mean, cmap='RdYlBu_r', aspect='equal',
                   vmin=0, interpolation='nearest')
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(ch, rotation=90, fontsize=7)
    ax.set_yticklabels(ch, fontsize=7)
    ax.axhline(7.5, color='white', linewidth=1.5)
    ax.axvline(7.5, color='white', linewidth=1.5)
    ax.set_xlabel('Target Muscle (receiving attention)', fontsize=9)
    ax.set_ylabel('Source Muscle (sending attention)', fontsize=9)
    ax.set_title('GAT Attention Weights α_ij\n(Nonlinear Muscle Coordination)', fontsize=11)
    plt.colorbar(im, ax=ax, label='Mean Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Attention] 保存 → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Colab 実行エントリーポイント
# =============================================================================

def run_phase2(
    npz_path     : str | Path,
    output_dir   : str | Path = "./phase2_results",
    gae_epochs   : int   = 100,
    gat_epochs   : int   = 50,
    alpha_l1     : float = 1e-3,    # GAE L1正則化強度
    train_ratio  : float = 0.67,    # 学習データ割合
    val_ratio    : float = 0.17,    # 検証データ割合
    task_label   : int   = None,    # None=疑似ラベル / 0 or 1=実タスクラベル
):
    """
    Phase 2 のメイン実行関数（修正版）。

    【指摘反映箇所】
    ① ノード特徴量: 隣接行ベクトル → EMG波形スライス（window_size次元）
       → 各ノード（筋）の特徴量が独立した生理学的意味を持つ
    ② 時間分割（Time-based Split）: ランダムシャッフル禁止
       → Train前半 / Val中間 / Test後半 で時間軸でスパッと分割
    ③ L1正則化: GAEの損失に alpha_l1 * ||A_hat||_1 を追加
       → 本当に重要なエッジのみが残るスパースな再構成を促進

    【データフロー】
      .npz (Phase 1 出力)
        ↓ adjacency (T, N, N) + emg_windows (T, N, W)
        ↓ 時間分割 → Train/Val/Test
        ↓ GAE 学習（L1正則化付き）
        ↓ 再構成行列 A_hat: MI にない「潜在エッジ」を発見
        ↓ GAT 学習（Train ラベル付き）
        ↓ アテンション α_ij: 「筋の注意方向」を可視化
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = check_environment()
    data   = load_phase1_data(npz_path)

    adj      = data['adjacency']           # (T, N, N)
    T, N, _  = adj.shape
    ch_names = data['channel_names']

    # EMGウィンドウ特徴量（.npzに含まれる場合はそれを使用）
    if 'emg_windows' in data:
        emg_windows = data['emg_windows']  # (T, N, W)
    else:
        # emg_windows がない場合: 隣接行列の行を代替特徴として使用
        print("  [警告] emg_windows が .npz に含まれていません。")
        print("         adjacency の行ベクトルを代替特徴量として使用します。")
        print("         ★理想的には DynamicMuscleNetworkAnalyzer.export_for_gnn()")
        print("           を emg_data 引数付きで呼び直してください。")
        emg_windows = adj  # shape (T, N, N) → (T, N, W=N) で代替

    W = emg_windows.shape[2]
    print(f"\n[Phase 2] T={T} windows, N={N} muscles, W={W} features/node")

    # ──────────────────────────────────────────────────────────
    # Step 1: PyG データ変換（ノード特徴 = EMG波形スライス）
    # ──────────────────────────────────────────────────────────
    print("\n[Step 1] PyG Data 変換 (特徴量 = EMG波形) ...")
    pyg_list = [
        adjacency_to_pyg(adj[t], emg_windows[t], device)
        for t in range(T)
    ]

    # ──────────────────────────────────────────────────────────
    # Step 2: 時間ベース分割（Data Leakage 防止）
    # ──────────────────────────────────────────────────────────
    print("\n[Step 2] 時間ベース分割 ...")
    # ラベル設定: task_label が指定されていれば全データ同一ラベル
    # なければ「前半=立脚期優位, 後半=遊脚期優位」の疑似ラベル
    if task_label is not None:
        labels = np.full(T, task_label, dtype=int)
    else:
        labels = np.array([0] * (T // 2) + [1] * (T - T // 2))

    (train_data, val_data, test_data,
     train_labels, val_labels, test_labels) = time_based_split(
        pyg_list, labels, train_ratio, val_ratio
    )

    # ──────────────────────────────────────────────────────────
    # Step 3: GAE 学習（L1 正則化でスパース再構成）
    # ──────────────────────────────────────────────────────────
    print(f"\n[Step 3] GAE 学習 (epochs={gae_epochs}, L1={alpha_l1}) ...")
    gae = MuscleGAE(n_features=W, hidden_dim=32, latent_dim=16)
    losses = gae.train(train_data, n_epochs=gae_epochs,
                       alpha_l1=alpha_l1, device=device)
    print(f"  Train最終 Loss: {losses[-1]:.6f}")

    # 全データで再構成（比較用）
    A_gae_all = gae.reconstruct(pyg_list, device=device)
    A_gae_test = gae.reconstruct(test_data, device=device)
    np.save(str(out_dir / "gae_reconstructed_all.npy"), A_gae_all)
    np.save(str(out_dir / "gae_reconstructed_test.npy"), A_gae_test)

    # ──────────────────────────────────────────────────────────
    # Step 4: GAT 学習（アテンション抽出）
    # ──────────────────────────────────────────────────────────
    print(f"\n[Step 4] GAT 学習 (epochs={gat_epochs}) ...")
    gat = MuscleGAT(n_features=W, n_classes=2)
    gat.train(train_data, train_labels, n_epochs=gat_epochs, device=device)

    # テストデータでアテンション抽出（学習時に見ていないデータ）
    attentions = gat.get_attention(test_data, device=device)
    np.save(str(out_dir / "gat_attentions_test.npy"), attentions)

    # ──────────────────────────────────────────────────────────
    # Step 5: 可視化
    # ──────────────────────────────────────────────────────────
    print("\n[Step 5] 可視化 ...")
    compare_mi_vs_gae(
        adj, A_gae_all, ch_names,
        save_path=out_dir / "mi_vs_gae_comparison.png"
    )
    plot_attention_heatmap(
        attentions, ch_names,
        save_path=out_dir / "gat_attention_heatmap.png"
    )

    # 学習曲線
    _plot_learning_curve(losses, out_dir / "gae_learning_curve.png")

    print(f"\n=== Phase 2 完了。結果: {out_dir.resolve()} ===")
    return {
        'gae_loss_final'    : float(losses[-1]),
        'n_train'           : len(train_data),
        'n_val'             : len(val_data),
        'n_test'            : len(test_data),
        'A_gae'             : A_gae_all,
        'attentions'        : attentions,
    }


def _plot_learning_curve(losses: list[float], save_path: Path) -> None:
    """GAE学習曲線を保存する。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='steelblue', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE + L1)')
    ax.set_title('GAE Learning Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [学習曲線] 保存 → {save_path}")


# =============================================================================
# __main__
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: GNN Muscle Network")
    parser.add_argument('--npz',        default='demo_for_gnn.npz',
                        help='Phase 1 の export_for_gnn() 出力 (.npz)')
    parser.add_argument('--output_dir', default='./phase2_results')
    parser.add_argument('--gae_epochs', type=int, default=50)
    parser.add_argument('--gat_epochs', type=int, default=50)
    args = parser.parse_args()

    run_phase2(
        npz_path   = args.npz,
        output_dir = args.output_dir,
        gae_epochs = args.gae_epochs,
        gat_epochs = args.gat_epochs,
    )
