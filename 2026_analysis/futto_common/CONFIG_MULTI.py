# =============================================================================
# futto_common/CONFIG_MULTI.py
#
# 役割:
#   今回の実験(被験者7名 x 3日程、task01〜03すべてOptiマーカーを左右の脚関節に
#   直接配置する方式)専用の設定。
#
#   前回の実験(被験者Ideのみ、床反力・筋電・Futto本体に取り付けたOptiマーカーの
#   解析)の設定は futto_common/CONFIG.py に残しており、このファイルとは
#   完全に独立している(互いにimportしない。値の重複は許容する)。
#
#   新実験向けのプログラムはこちらを参照する:
#       from futto_common import CONFIG_MULTI as config
#   または
#       import CONFIG_MULTI as CONFIG
#
#   現在の利用箇所: opti_edit_for_joint.py, cot_analysis/*
# =============================================================================

from pathlib import Path

# --- ▼▼▼ 基本設定 (新実験のタスク共通) ▼▼▼ ---
# CONFIG.py (旧実験) と同じ値。新実験でもタスク構成(task01=T1通常配置,
# task02=T2改良配置, task03=N非着用)・結果出力先は共通のため、値のみ複製する。
RESULT_DIR = r"C:\FuttoAnalysis\result"
TASKS = ['task01', 'task02', 'task03']

# タスク(task01/02/03) <-> 実験プロトコル上の条件表記(T1/T2/N)
TASK_CONDITION_LABELS = {'task01': 'T1', 'task02': 'T2', 'task03': 'N'}

# --- ▼▼▼ 多被験者拡張 (7名 x 3日, 関節マーカー10点方式) ▼▼▼ ---

# 新規計測データのルート (opti / labchart の生データ)
RAW_ROOT = Path(r"C:\FuttoAnalysis")

DAYS = ["First", "Second", "Third"]

# 多被験者拡張の被験者ID。
SUBJECTS_MULTI = ["Okumura", "Sakai", "Kojima", "Hara",
                  "Yanagawa", "Sawada", "Kanai"]

# opti / labchart / cpet の生データ拡張子。EMG(Cometa)は別ルート・別関数
# (futto_network_real/emg_pipeline/emg_loader.py の build_emg_raw_path) を使う。
DEVICE_EXTENSIONS = {
    "opti": "csv",
    "labchart": "txt",
    "cpet": "txt",
}


def get_raw_path(device: str, subject: str, day: str, task: str) -> Path:
    """
    多被験者拡張後の生データパスを返す (opti / labchart のみ)。

    例: get_raw_path('opti', 'Ide', 'First', 'task01')
        -> C:\\FuttoAnalysis\\opti\\Ide\\First\\task01.csv
    """
    if device not in DEVICE_EXTENSIONS:
        raise ValueError(f"device は {list(DEVICE_EXTENSIONS)} のいずれか。指定値: {device}")
    if day not in DAYS:
        raise ValueError(f"day は {DAYS} のいずれか。指定値: {day}")
    ext = DEVICE_EXTENSIONS[device]
    return RAW_ROOT / device / subject / day / f"{task}.{ext}"


# --- 関節マーカー方式・新実験データルート ---
#
# 前実験では task03 (器具非着用) のみ関節に直接マーカーを貼付していたが、
# 今回の実験では task01〜03 すべてで関節マーカー方式(左右10点)を用いる。
# 生データは opti/{subject}/{day}/{task}.csv (get_raw_path) とは
# 別の "opti\joint" 配下に保存される。
JOINT_RAW_ROOT = RAW_ROOT / "opti" / "joint"


def get_joint_opti_path(subject: str, day: str, task: str) -> Path:
    """
    関節マーカー方式(新実験)の Opti 生データパスを返す。

    例: get_joint_opti_path('Ide', 'First', 'task01')
        -> C:\\FuttoAnalysis\\opti\\joint\\Ide\\First\\task01.csv
    """
    if day not in DAYS:
        raise ValueError(f"day は {DAYS} のいずれか。指定値: {day}")
    return JOINT_RAW_ROOT / subject / day / f"{task}.csv"


# --- 関節マーカー方式 (左右の脚関節のみ、計10点) ---
#
# OptiTrackの生マーカーIDは被験者・計測ごとに異なる(セッションごとの
# ストリーミングID)ため、被験者間で共通の「テンプレートID」を固定で
# 割り当てる。opti_edit_for_joint.py が生ID -> このテンプレートIDへの
# マッピングを行う。
#
# 体軸(体幹の向き)は地面に鉛直な軸に平行と近似する。関節角度はこの
# 鉛直軸に対するセグメント角度として算出するため、体幹の実測傾斜は
# 反映されない。
JOINT_MARKER_TEMPLATE_IDS = {
    'L_Hip': 1, 'L_Knee': 2, 'L_Ankle': 3, 'L_Heel': 4, 'L_Toe': 5,
    'R_Hip': 6, 'R_Knee': 7, 'R_Ankle': 8, 'R_Heel': 9, 'R_Toe': 10,
}

# 関節マーカー方式(新実験)の仮想関節定義。マーカーが関節上に直接あるため、
# 旧task03と同様 'single' でテンプレートIDをそのまま参照する
# (joint_analysis/joint_calc.py の calc_joint_centers() が参照する)。
JOINT_CENTER_DEFS_MULTI = {
    name: {'type': 'single', 'markers': [tid]}
    for name, tid in JOINT_MARKER_TEMPLATE_IDS.items()
}

# 関節マーカー方式(新実験)の基準マーカー。歩行周期平均化(create_anime_grad.py の
# calculate_mean_gait_cycle)で全身の並進基準として使う単一マーカーが必要なため、
# 左股関節マーカーを基準とする。
JOINT_REFERENCE_MARKER_ID = JOINT_MARKER_TEMPLATE_IDS['L_Hip']

# オクルージョン(マーカー遮蔽)判定しきい値。
# 連続してこの値以上フレームが欠損した区間を「遮蔽」としてログに記録し、
# 補間せず欠損として解析対象から除外する。
# FRAME_RATE(100Hz) 換算で 10フレーム = 0.1秒。
OCCLUSION_MIN_GAP_FRAMES = 10

# --- CPET(呼気代謝) / COT-Fr 解析 ---
#
# 実験プロトコル(Futto 歩行計測 実験プロトコル.docx 5節)に準拠:
#   計測は歩行開始1分前から開始し、0.5→1.5m/s の6段階を2分間ずつ
#   トレッドミルを止めずに連続歩行する。各段階の「後半1分間」を解析対象とする。
COT_SPEED_STAGES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]  # m/s, 6段階
STAGE_DURATION_SEC = 120     # 各速度段階の長さ(2分間)
STAGE_TAIL_WINDOW_SEC = 60   # 各段階のうち解析に使う後半区間の長さ(1分間)
WALKING_START_SEC = 60       # 計測開始(歩行開始1分前)から歩行開始までの秒数
REST_WINDOW_SEC = (0, 60)    # 安静時VO2のQC参照区間(COT計算式には使用しない)
COT_G = 9.8                  # 重力加速度 [m/s^2]

# 被験者の個人データ (体重[kg]・脚長[cm])。COT・フルード数の算出に使用する。
# 実測値が確定していないため仮に None を設定。計測後に必ず更新すること。
SUBJECT_PROFILES = {
    subject: {'weight_kg': None, 'leg_length_cm': None}
    for subject in SUBJECTS_MULTI
}
