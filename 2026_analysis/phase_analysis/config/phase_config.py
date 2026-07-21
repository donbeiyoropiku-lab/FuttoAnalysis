# =============================================================================
# phase_analysis/config/phase_config.py
#
# 役割:
#   Futto ゴムの位相遅れ解析 (phase lag analysis) に関する設定を定義する。
#   関節角度・ゴム張力・EMG のパス取得と、歩行周期(%) ⇔ 時間(ms) の変換を行う。
#
# 参考: mechanics_analysis/io_loader.py, emg_synergy/config/emg_config.py
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# タスク・フェーズ定義 (mechanics_analysis / emg_synergy と共通)
# ---------------------------------------------------------------------------
TASKS = ['task01', 'task02', 'task03']
TASK_TITLES = {
    'task01': 'Full Futto (15 nodes)',
    'task02': '8-node Futto',
    'task03': 'No device',
}
PHASES = {
    1: {'speed': '0.7'},
    2: {'speed': '0.9'},
    3: {'speed': '1.1'},
    4: {'speed': '1.3'},
    5: {'speed': '1.5'},
}
SUBJECTS = ['Ide']

# ---------------------------------------------------------------------------
# データルート
# ---------------------------------------------------------------------------
# mechanics_analysis が張力・関節角度 CSV を出力するルート
FUTTO_RESULT_DIR = Path(r'C:\FuttoAnalysis\result')

# 位相解析結果の保存ルート (このパッケージ専用)
PHASE_RESULT_DIR = Path(r'C:\FuttoAnalysis\result\phase_analysis')

# EMG (emg_synergy) の average CSV ルート
EMG_ANALYSIS_RESULT_DIR = Path(r'C:\Users\ihika\2026_experiment')


# ---------------------------------------------------------------------------
# パス取得関数
# ---------------------------------------------------------------------------

def get_tension_csv_path(task: str, phase_num: int, speed: str) -> Path:
    """
    strength_visualize が出力した張力 CSV のパスを返す。

    出力形式 (ワイド): gait_cycle_%, Front_Upper_In, Back_Upper_In, ...
    """
    return (
        FUTTO_RESULT_DIR / '2026' / task / speed
        / f'{task}_Phase{phase_num}_{speed}ms_tension.csv'
    )


def get_joint_angles_csv_path(task: str, phase_num: int, speed: str) -> Path:
    """mechanics_analysis が出力した関節角度 CSV のパスを返す。"""
    return (
        FUTTO_RESULT_DIR / '2026' / task / speed
        / f'{task}_Phase{phase_num}_{speed}ms_joint_angles.csv'
    )


def get_torque_csv_path(task: str, phase_num: int, speed: str) -> Path:
    """mechanics_analysis が出力したトルク CSV のパスを返す。"""
    return (
        FUTTO_RESULT_DIR / '2026' / task / speed
        / f'{task}_Phase{phase_num}_{speed}ms_torque.csv'
    )


def get_emg_average_csv_path(subject: str, task: str, phase_num: int) -> Path:
    """emg_synergy (既存EMG前処理) が出力した average CSV のパスを返す。"""
    return (
        EMG_ANALYSIS_RESULT_DIR / subject / 'analysis_results'
        / f'{subject}_{task}_Phase{phase_num}_average.csv'
    )


def get_phase_result_dir(subject: str, task: str, speed: str) -> Path:
    """位相解析結果 (グラフ・CSV) の保存先ディレクトリを返す。"""
    return PHASE_RESULT_DIR / subject / task / speed


# ---------------------------------------------------------------------------
# 歩行周期(%) ⇔ 時間(ms) の変換
# ---------------------------------------------------------------------------
# 過去の解析 (mechanics_analysis) より、歩幅(stride) がおおむね速度に比例して
# 増加するモデルの下では 1歩行周期 T_cycle はどの速度でもほぼ一定 (≈1.3秒) と
# 近似できる。クロスコリレーション・CRP の結果を ms 単位で解釈する際に使う。
#
# 被験者ごとに実測の T_cycle が得られる場合はここを更新するか、
# cycle_pct_to_ms() の base_cycle_s 引数に実測値を渡す。
DEFAULT_T_CYCLE_SEC = 1.3


def cycle_pct_to_ms(delta_pct: float, base_cycle_s: float = None) -> float:
    """
    歩行周期に対する割合 [%] を時間 [ms] に変換する。

    Parameters
    ----------
    delta_pct    : float  歩行周期(%) での差分 (例: クロスコリレーションのラグ)
    base_cycle_s : float または None  1歩行周期の時間 [秒]
        None の場合 DEFAULT_T_CYCLE_SEC を使用。

    Returns
    -------
    float  時間 [ms]
    """
    T = base_cycle_s if base_cycle_s is not None else DEFAULT_T_CYCLE_SEC
    return delta_pct / 100.0 * T * 1000.0


def ms_to_cycle_pct(delta_ms: float, base_cycle_s: float = None) -> float:
    """時間 [ms] を歩行周期に対する割合 [%] に変換する (cycle_pct_to_ms の逆変換)。"""
    T = base_cycle_s if base_cycle_s is not None else DEFAULT_T_CYCLE_SEC
    return delta_ms / 1000.0 / T * 100.0


# ---------------------------------------------------------------------------
# ゴムセグメントのグループ分け (股関節周り・膝関節周りなど)
# ---------------------------------------------------------------------------
# futto_common/CONFIG.py の NATURAL_LENGTHS / SEGMENT_GROUPS に準拠した
# 実際のセグメント名。股関節を跨ぐセグメント (Upper系) と
# 膝関節を跨ぐセグメント (Knee系) を分けている。
#
# 位相解析では「股関節角度と最も直接的に連動するゴム」として
# HIP_SEGMENTS を既定の解析対象とする。
HIP_SEGMENTS = ['Front_Upper_In', 'Front_Upper_Out',
                'Back_Upper_In', 'Back_Upper_Out']
KNEE_SEGMENTS = ['Front_Knee_Upper_Out', 'Front_Knee_Upper_In',
                 'Front_Knee_Lower_Out', 'Front_Knee_Lower_In',
                 'Back_Knee_Out', 'Back_Knee_In']
ANKLE_SEGMENTS = ['Front_Shin', 'Toe_Out', 'Toe_In',
                   'Back_Shin_Out', 'Back_Shin_In']

# 全セグメント名 (task01 = 15ノード/17セグメント構成)
ALL_SEGMENTS = (
    HIP_SEGMENTS + KNEE_SEGMENTS + ANKLE_SEGMENTS
    + ['Back_Thigh_Out', 'Back_Thigh_In']
)


# ---------------------------------------------------------------------------
# LabChart 床反力 (GRF) 関連設定
# ---------------------------------------------------------------------------
# 生データ (.txt) と歩行周期リスト (_gait_cycles.csv) は
# gaitcycle/data_processing.py, gaitcycle_force_labchart.py が出力・使用する
# ものと同じディレクトリ構造を参照する。
LABCHART_DIR = Path(r'C:\FuttoAnalysis\labchart')

# LabChart サンプリング周波数 [Hz]
LABCHART_FS = 1000

# チャンネル列番号 (pandas の列インデックス。0列目が Time)
# ChannelTitle= チャンネル1 ... チャンネル13 の並びに対応。
# チャンネル7 (列7) は未接続。
GRF_CHANNEL_MAP = {
    'R': {'Fx': 1, 'Fy': 2, 'Fz': 3,  'Mx': 4,  'My': 5,  'Mz': 6},
    'L': {'Fx': 8, 'Fy': 9, 'Fz': 10, 'Mx': 11, 'My': 12, 'Mz': 13},
}

# ローパスフィルタ設定 (data_processing.py の lowpass_filter に準拠)
GRF_FILTER_CUTOFF_HZ = 13
GRF_FILTER_ORDER = 5

# ベースライン補正に使うサンプル区間 (data_processing.py の adjusted_data に準拠)
# 40-45 秒 (1000Hz なので 40000-45000 サンプル) の下位10%平均をゼロ点とする
GRF_BASELINE_SAMPLE_RANGE = (40000, 45000)

# 体重推定に使うサンプル区間 (最初の安定立位区間、1-10秒)
GRF_BODYWEIGHT_SAMPLE_RANGE = (1000, 10000)


def get_labchart_txt_path(date: str, task: str) -> Path:
    """
    LabChart の生データ TXT のパスを返す。

    Parameters
    ----------
    date : str  計測日 (例: '20260217')
    task : str  タスク名 (例: 'task01')
    """
    return LABCHART_DIR / date / f'{task}.txt'


def get_gait_cycles_csv_path(date: str, task: str) -> Path:
    """
    data_processing.py / gaitcycle_force_labchart.py が出力した
    歩行周期リスト CSV のパスを返す。

    列: hs_time, to_time, next_hs_time, hs_frame, to_frame, next_hs_frame
    """
    return LABCHART_DIR / date / f'{task}_gait_cycles.csv'