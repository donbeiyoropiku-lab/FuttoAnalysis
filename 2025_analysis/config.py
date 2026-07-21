# =============================================================================
# CONFIG.py
#
# 概要:
# opti_edit_E.py, create_anime_grad_D.py, strength_visualize.py で使用する
# 設定値をタスクごとに管理するファイル。
#
# 構成:
# 1. 基本設定: 全タスクで共通の設定値
# 2. タスク別設定 (TASK_CONFIGS): task1, task2, task3 固有の設定値
#    - ファイルパス
#    - 時刻区間
#    - 座標範囲
#    - マーカーID定義 (テンプレート、ゴム線、筋肉インジケータ)
#    - 自然長
#    - キーフレーム情報
# =============================================================================

import numpy as np

# --- ▼▼▼ 基本設定 (全タスク共通) ▼▼▼ ---
# ファイルパス関連
RUBBER_PROPERTIES_EXCEL_PATH = r"C:\FuttoAnalysis\rubber_strength.xlsx"
RUBBER_PROPERTIES_SHEET_NAME = 'Sheet1'
RESULT_DIR = r"C:\FuttoAnalysis\result" # アニメーション保存先

# 計測・処理関連
FRAME_RATE = 100 # OptiTrackのフレームレート / アニメーションのフレームレート
TIME_OFFSET = 0.0  # LabChartとOptiTrackの時間オフセット(秒) - 同期済みなら0.0

# 位相加重ハイブリッド追跡 (opti_edit_E) 用
PHASE_WEIGHTING = {
    # 剛体追跡(opti_edit_C風)を適用する位相範囲 (%)
    'rigid_phase_ranges': [(0.0, 50.0), (95.0, 100.0)],
    'rigid_weight': 0.8,        # 剛体予測の重み (現在は未使用だが参考値として残す)

    # 近傍点追跡(opti_edit_D風)を適用する位相範囲 (%)
    'flexible_phase_range': (50.0, 95.0), # rigid_phase_ranges以外の区間
    'flexible_weight': 0.2,     # 剛体予測の重み (現在は未使用だが参考値として残す)

    # 参考: 移行区間 (線形補間は現在 calculate_phase_weight では使用せず、範囲で切り替え)
    'transition_ranges': [(50.0, 50.0), (95.0, 95.0)]
}
# セグメント階層 (opti_edit_E, create_anime, strength)
CHAIN_HIERARCHY = {
    'Hip': None, 'Thigh': 'Hip', 'Knee': 'Thigh', 'Shank': 'Knee', 'Foot': 'Shank'
}
# セグメント処理順序 (opti_edit_E)
PROCESSING_ORDER = ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']
# マッチング許容誤差 (opti_edit_E)
MATCHING_THRESHOLD_MM = 75.0

# 座標軸マッピング (create_anime, strength)
AXIS_MAPPING = {"final_x": 'z', "final_y": 'x', "final_z": 'y'}

# グラフ用セグメントグループ (create_anime, strength)
SEGMENT_GROUPS = {
    "Front_Knee": ["Front_Knee_Upper_Out", "Front_Knee_Upper_In", "Front_Knee_Lower_Out", "Front_Knee_Lower_In"],
    "Front_Ankle": ["Front_Shin", "Toe_Out", "Toe_In"],
    "Back_Knee": ["Back_Knee_Out", "Back_Knee_In"],
    "Back_Thigh": ["Back_Thigh_Out", "Back_Thigh_In"],
    "Back_Shin": ["Back_Shin_In", "Back_Shin_Out"],
    "Front_Hip": ["Front_Upper_In", "Front_Upper_Out"],
    "Back_Hip": ["Back_Upper_In", "Back_Upper_Out"]
}

# 筋肉インジケータ表示設定 (strength)
MUSCLE_MARKER_BASE_SIZE = 4
MUSCLE_MARKER_SCALE_FACTOR = 16
# --- ▲▲▲ 基本設定ここまで ▲▲▲ ---


# --- ▼▼▼ タスク別設定 ▼▼▼ ---
TASK_CONFIGS = {
    'task1': {
        # --- ファイルパス ---
        'OPTI_CSV_PATH': r"C:\FuttoAnalysis\opti\20250731\task1.csv",
        'OUTPUT_CSV_PATH': r"C:\FuttoAnalysis\opti\20250731\task1_corrected_A.csv",
        'LABCHART_CYCLES_PATH': r"C:\FuttoAnalysis\labchart\20250731\task1_gait_cycles.csv",
        'MEAN_CYCLE_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20250731\task1_mean_cycle.csv", # opti_edit_E / create_anime
        'MEAN_CYCLE_RANGED_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20250731\task1_mean_cycle_ranged.csv", # create_anime 出力
        'TENSION_DATA_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20250731\task1_tension_data.csv", # strength 出力
        'EMG_DATA_CSV_PATH': r"C:\EMG\2025_analysis\analysis_results\Subject B_task04_Phase3_average.csv", # strength 入力

        # --- ★ 再生速度 (task1 = 1.0倍速) ---
        'PLAYBACK_SPEED_MULTIPLIER': 1.0,

        # --- 時刻区間 ---
        'STATIC_START': 4.0, 'STATIC_END': 10.0,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 11.705,
        'T2_WALK_END': 50.0, 'T2_STATIC_START': 50.0, # 仮
        'KEYFRAME_TIME': 11.705,

        # --- OptiTrack 関連 ---
        'PLAUSIBLE_BOUNDS': {'x': (0, 1000), 'y': (0, 1100), 'z': (-200, 400)},
        'KEYFRAME_MAP': {
            15810: 15810, 15796: 15796, 15814: 15814, 15804: 15804, # Hip
            15832: 15816, # Thigh
            15808: 15808, 15794: 15794, 15800: 15800, 15798: 15798, # Knee
            15818: 15818, # Shank
            15802: 15802, # Foot (元 Shank)
            15812: 15812, 15806: 15806, 15792: 15792, 15830: 15820, # Foot
        },
        'SEGMENTS': {
            'Hip': [15810, 15796, 15814, 15804], 'Thigh': [15816],
            'Knee': [15808, 15794, 15800, 15798], 'Shank': [15818],
            'Foot': [15812, 15806, 15792, 15820, 15802]
        },
        'REFERENCE_MARKER_ID': 15810, # create_anime 用

        # --- ゴム & 筋肉 ---
        'LINES_TO_DRAW': {
            "Front_Upper_In": (15810, 15808), "Front_Upper_Out": (15796, 15808),
            "Front_Knee_Upper_Out": (15808, 15794), "Front_Knee_Upper_In": (15808, 15800),
            "Front_Knee_Lower_Out": (15794, 15798), "Front_Knee_Lower_In": (15800, 15798),
            "Front_Shin": (15798, 15802), "Toe_Out": (15802, 15812), "Toe_In": (15802, 15806),
            "Back_Upper_In": (15814, 15816), "Back_Upper_Out": (15804, 15816),
            "Back_Thigh_Out": (15816, 15794), "Back_Thigh_In": (15816, 15800),
            "Back_Knee_Out": (15794, 15818), "Back_Knee_In": (15800, 15818),
            "Back_Shin_In": (15818, 15792), "Back_Shin_Out": (15818, 15820),
        },
        'NATURAL_LENGTHS': {
            "Front_Upper_In": 260, "Front_Upper_Out": 300, "Front_Knee_Upper_Out": 80, "Front_Knee_Upper_In": 80,
            "Front_Knee_Lower_Out": 80, "Front_Knee_Lower_In": 80, "Front_Shin": 205, "Toe_Out": 72, "Toe_In": 72,
            "Back_Upper_In": 200, "Back_Upper_Out": 160, "Back_Thigh_Out": 263, "Back_Thigh_In": 215,
            "Back_Knee_Out": 162, "Back_Knee_In": 155, "Back_Shin_In": 135, "Back_Shin_Out": 135,
        },
        'MUSCLE_INDICATORS': {
            "Tibialis Anterior": {'emg_col': "L_TA_mean", 'type': 'midpoint', 'markers': [15798, 15802]},
            "Soleus": {'emg_col': "L_SOL_mean", 'type': 'centroid', 'markers': [15818, 15792, 15820]},
            "Rectus Femoris": {'emg_col': "L_RF_mean", 'type': 'centroid', 'markers': [15810, 15808, 15796]},
            "Vastus Lateralis": {'emg_col': "L_VL_mean", 'type': 'weighted_midpoint', 'markers': [15808,15796 ], 'weight': 0.25},
            "Biceps Femoris": {'emg_col': "L_BF_mean", 'type': 'midpoint', 'markers': [15816, 15794]},
            "Semitendinosus": {'emg_col': "L_ST_mean", 'type': 'single', 'markers': [15816]},
            "Gluteus Maximus Area": {'emg_col': "L_GM_mean", 'type': 'offset', 'markers': [15814, 15816, 15804], 'ref_marker': [15814, 15816], 'weight': 0.3},
            "Iliopsoas Area": {'emg_col': "L_ILIO_mean", 'type': 'midpoint', 'markers': [15810, 15796]},
        },
        # ★ 新関節定義 (Task1)
        'JOINT_CENTER_DEFS': {
            # Hip: Mid(15796,15804) と Mid(15810,15814) を 1:3 に内分
            'Hip': {'type': 'ratio_1_3_between_mids', 'markers': [15796, 15804, 15810, 15814]},
            # Knee: Mid(15794, 15800)
            'Knee': {'type': 'midpoint', 'markers': [15794, 15800]},
            # Ankle: Mid( (15812-15820 2:1), (15806-15792 2:1) )
            'Ankle': {'type': 'mid_of_ratio_2_1', 'markers': [15812, 15820, 15806, 15792]}
        },

        
        
        # ★ トルク計算用セグメントマップ (Task1 ID)
        # ここに含まれるIDにかかるゴム張力が、対応する関節トルク計算に使われる
        'SEGMENT_MAP': {
            'Foot':  [15812, 15820, 15806, 15792], # つま先、かかと付近 (要確認: ゴム終点ID)
            'Shank': [15798, 15802, 15818], # 膝下、すね (要確認)
            'Thigh': [15794, 15800, 15808, 15816]  # 膝上、大腿 (要確認)
        }, 

        'RUBBER_TORQUE_MAP': {
            # "ゴム名": [ "作用関節", "付着点(遠位)", "力の方向元(近位)" ]
            "Front_Upper_In": ("Hip", 15808, 15810), "Front_Upper_Out": ("Hip", 15808, 15796),
            "Front_Knee_Upper_Out": ("Knee", 15794, 15808), "Front_Knee_Upper_In":  ("Knee", 15800, 15808),
            "Front_Knee_Lower_Out": ("Knee", 15794, 15798), "Front_Knee_Lower_In":  ("Knee", 15800, 15798),
            "Front_Shin": ("Ankle", 15802, 15798), # 15802(Foot) に作用
            "Toe_Out": ("Ankle", 15812, 15802),    # 15812(Foot) に作用
            "Toe_In": ("Ankle", 15806, 15802),     # 15806(Foot) に作用
            "Back_Upper_In": ("Hip", 15816, 15814), "Back_Upper_Out": ("Hip", 15816, 15804),
            "Back_Thigh_Out": ("Knee", 15794, 15816), "Back_Thigh_In": ("Knee", 15800, 15816),
            "Back_Knee_Out": ("Knee", 15794, 15818), "Back_Knee_In": ("Knee", 15800, 15818),
            "Back_Shin_In": ("Ankle", 15792, 15818), # 15792(Foot) に作用
            "Back_Shin_Out": ("Ankle", 15820, 15818), # 15820(Ankle) に作用
        },
        # --- ▲▲▲ トルク計算用定義ここまで ▲▲▲ ---
    },
    'task2': {
        # --- ファイルパス ---
        'OPTI_CSV_PATH': r"C:\FuttoAnalysis\opti\20251027\task2.csv",
        'OUTPUT_CSV_PATH': r"C:\FuttoAnalysis\opti\20251027\task2_corrected_D.csv",
        'LABCHART_CYCLES_PATH': r"C:\FuttoAnalysis\labchart\20251027\task2_gait_cycles.csv",
        'MEAN_CYCLE_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task2_mean_cycle.csv",
        'MEAN_CYCLE_RANGED_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task2_mean_cycle_ranged.csv", # create_anime用
        'TENSION_DATA_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task2_ranged_tension_data.csv",
        'EMG_DATA_CSV_PATH': r"C:\EMG\2025_analysis\analysis_results\Subject B_task04_Phase7_average.csv",


        # --- ★ 再生速度 (task2 = 1.57倍速) ---
        'PLAYBACK_SPEED_MULTIPLIER': 1.57,

        # --- 時刻区間 ---
        'STATIC_START': 3.00, 'STATIC_END': 10.00,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 12.596,
        'T2_WALK_END': 50.0, 'T2_STATIC_START': 50.0,
        'KEYFRAME_TIME': 12.596,

        # --- OptiTrack 関連 ---
        'PLAUSIBLE_BOUNDS': {'x': (-300, 150), 'y': (0, 1100), 'z': (-1000, 200)},


        'SEGMENTS': {
            'Hip': [7264, 7260, 7250, 7258],
            'Thigh': [7266],
            'Knee': [7256, 7254, 7252, 7248],
            'Shank': [7240],
            'Foot': [7246, 7262, 7244, 7238, 7242],
        },

        'REFERENCE_MARKER_ID': 7264,

        'KEYFRAME_MAP': { # ★ 要手動確認/修正 ★
            # --- Hip ---
            7264: 7264,
            7260: 7260,
            7250: 7250,
            7258: 7258,
            # --- Thigh ---
            7266: 7266,
            # --- Knee ---
            7256: 7256,
            7254: 7254,
            7252: 7252,
            7248: 7248,
            # --- Shank ---
            7240: 7240,
            # --- Foot ---
            7270: 7246,
            7262: 7262,
            7244: 7244,
            7238: 7238,
            7242: 7242,
        },

        'LINES_TO_DRAW': { # ★ 要手動確認/修正 (ルールベース自動割り当て) ★
            "Front_Upper_In": (7258, 7256),
            "Front_Upper_Out": (7260, 7256),
            "Front_Knee_Upper_Out": (7256, 7252),
            "Front_Knee_Upper_In": (7256, 7254),
            "Front_Knee_Lower_Out": (7248, 7252),
            "Front_Knee_Lower_In": (7248, 7254),
            "Front_Shin": (7248, 7246),
            "Toe_Out": (7246, 7238),
            "Toe_In": (7246, 7242),
            "Back_Upper_In": (7264, 7266),
            "Back_Upper_Out": (7250, 7266),
            "Back_Thigh_Out": (7266, 7252),
            "Back_Thigh_In": (7266, 7254),
            "Back_Knee_Out": (7252, 7240),
            "Back_Knee_In": (7254, 7240),
            "Back_Shin_Out": (7240, 7262),
            "Back_Shin_In": (7240, 7244),
        },

        'MUSCLE_INDICATORS': { # ★ 要手動確認/修正 (IDは自動割り当て) ★
            "Tibialis Anterior": {'emg_col': "L_TA_mean", 'type': 'midpoint', 'markers': [np.int64(7248), np.int64(7246)]},
            "Soleus": {'emg_col': "L_SOL_mean", 'type': 'centroid', 'markers': [np.int64(7240), np.int64(7244), np.int64(7262)]},
            "Rectus Femoris": {'emg_col': "L_RF_mean", 'type': 'centroid', 'markers': [np.int64(7258), np.int64(7256), np.int64(7260)]},
            "Vastus Lateralis": {'emg_col': "L_VL_mean", 'type': 'weighted_midpoint', 'markers': [np.int64(7256), np.int64(7260)], 'weight': 0.25},
            "Biceps Femoris": {'emg_col': "L_BF_mean", 'type': 'midpoint', 'markers': [np.int64(7266), np.int64(7252)]},
            "Semitendinosus": {'emg_col': "L_ST_mean", 'type': 'single', 'markers': [np.int64(7266)]},
            "Gluteus Maximus Area": {'emg_col': "L_GM_mean", 'type': 'offset', 'markers': [np.int64(7264), np.int64(7266), np.int64(7250)], 'weight': 0.3, 'ref_marker': [np.int64(7264), np.int64(7266)]},     
            "Iliopsoas Area": {'emg_col': "L_ILIO_mean", 'type': 'midpoint', 'markers': [np.int64(7258), np.int64(7260)]},
        },
        
        'NATURAL_LENGTHS': {
            "Front_Upper_In": 260, "Front_Upper_Out": 305, "Front_Knee_Upper_Out": 75, "Front_Knee_Upper_In": 75,
            "Front_Knee_Lower_Out": 80, "Front_Knee_Lower_In": 80, "Front_Shin": 210, "Toe_Out": 50, "Toe_In": 60,
            "Back_Upper_In": 210, "Back_Upper_Out": 160, "Back_Thigh_Out": 265, "Back_Thigh_In": 220,
            "Back_Knee_Out": 160, "Back_Knee_In": 130, "Back_Shin_In": 130, "Back_Shin_Out": 130,
        },
        # --- ▼▼▼【トルク計算用定義 (task2)】▼▼▼ ---
        # ★ 新関節定義 (Task2)
        'JOINT_CENTER_DEFS': {
            # Hip: Mid(7260,7250) と Mid(7258,7264) を 1:3 に内分
            'Hip': {'type': 'ratio_1_3_between_mids', 'markers': [7260, 7250, 7258, 7264]},
            # Knee: Mid(7252, 7254)
            'Knee': {'type': 'midpoint', 'markers': [7252, 7254]},
            # Ankle: Mid( (7238-7262 2:1), (7242-7244 2:1) )
            'Ankle': {'type': 'mid_of_ratio_2_1', 'markers': [7238, 7262, 7242, 7244]}
        },
        
        # ★ トルク計算用セグメントマップ (Task2 ID)
        # 提示されたゴム構成に基づき分類
        'SEGMENT_MAP': {
            'Foot':  [7238, 7242, 7262, 7244], # つま先, かかと
            'Shank': [7246, 7248, 7240],       # すね, 膝下
            'Thigh': [7252, 7254, 7256, 7266]  # 膝軸(7252/54)は大腿に含める(Hipトルク用)
        },

        'RUBBER_TORQUE_MAP': { # ★task1の構造に基づきIDを置換 (要確認)
            "Front_Upper_In": ("Hip", 7256, 7250), "Front_Upper_Out": ("Hip", 7256, 7260),
            "Front_Knee_Upper_Out": ("Knee", 7252, 7256), "Front_Knee_Upper_In":  ("Knee", 7254, 7256),
            "Front_Knee_Lower_Out": ("Knee", 7252, 7248), "Front_Knee_Lower_In":  ("Knee", 7254, 7248),
            "Front_Shin": ("Ankle", 7246, 7248), "Toe_Out": ("Ankle", 7262, 7246), "Toe_In": ("Ankle", 7244, 7246),
            "Back_Upper_In": ("Hip", 7266, 7258), "Back_Upper_Out": ("Hip", 7266, 7264),
            "Back_Thigh_Out": ("Knee", 7252, 7266), "Back_Thigh_In": ("Knee", 7254, 7266),
            "Back_Knee_Out": ("Knee", 7252, 7240), "Back_Knee_In": ("Knee", 7254, 7240),
            "Back_Shin_In": ("Ankle", 7244, 7240), "Back_Shin_Out": ("Ankle", 7238, 7240),
        },
        # --- ▲▲▲ トルク計算用定義ここまで ▲▲▲ ---
    },
    'task3': {
        # --- ファイルパス ---
        'OPTI_CSV_PATH': r"C:\FuttoAnalysis\opti\20251027\task3.csv",
        'OUTPUT_CSV_PATH': r"C:\FuttoAnalysis\opti\20251027\task3_corrected_D.csv",
        'LABCHART_CYCLES_PATH': r"C:\FuttoAnalysis\labchart\20251027\task3_gait_cycles.csv",
        'MEAN_CYCLE_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task3_mean_cycle.csv",
        'MEAN_CYCLE_RANGED_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task3_mean_cycle_ranged.csv",
        'TENSION_DATA_OUTPUT_PATH': r"C:\FuttoAnalysis\opti\20251027\task3_ranged_tension_data.csv",
        'EMG_DATA_CSV_PATH': r"C:\EMG\2025_analysis\analysis_results\Subject B_task04_Phase11_average.csv",

        # --- ★ 再生速度 (task3 = 2.14倍速) ---
        'PLAYBACK_SPEED_MULTIPLIER': 2.14,

        # --- 時刻区間 ---
        'STATIC_START': 5.0, 'STATIC_END': 10.0,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 13.498,
        'T2_WALK_END': 50.0, 'T2_STATIC_START': 50.0,
        'KEYFRAME_TIME': 13.498,

        # --- OptiTrack 関連 ---
        'PLAUSIBLE_BOUNDS': {'x': (-300, 150), 'y': (0, 1100), 'z': (-1000, 200)},
        'SEGMENTS': {
            'Hip': [8686, 8682, 8680, 8688],
            'Thigh': [8674],
            'Knee': [8692, 8698, 8690, 8702],
            'Shank': [8676],
            'Foot': [8678, 8696, 8700, 8684, 8694],
        },

        'REFERENCE_MARKER_ID': 8680,

        'KEYFRAME_MAP': { # ★ 要手動確認/修正 ★
            # --- Hip ---
            8686: 8686,
            8682: 8682,
            8680: 8680,
            8688: 8688,
            # --- Thigh ---
            8674: 8674,
            # --- Knee ---
            8692: 8692,
            8698: 8698,
            8690: 8690,
            8702: 8702,
            # --- Shank ---
            8676: 8676,
            # --- Foot ---
            8678: 8678,
            8696: 8696,
            8700: 8700,
            8684: 8684,
            8694: 8694,
        },

        'LINES_TO_DRAW': { # ★ 要手動確認/修正 (ルールベース自動割り当て) ★
            "Front_Upper_In": (8688, 8692),
            "Front_Upper_Out": (8682, 8692),
            "Front_Knee_Upper_Out": (8692, 8690),
            "Front_Knee_Upper_In": (8692, 8698),
            "Front_Knee_Lower_Out": (8702, 8690),
            "Front_Knee_Lower_In": (8702, 8698),
            "Front_Shin": (8702, 8678),
            "Toe_Out": (8678, 8684),
            "Toe_In": (8678, 8694),
            "Back_Upper_In": (8686, 8674),
            "Back_Upper_Out": (8680, 8674),
            "Back_Thigh_Out": (8674, 8690),
            "Back_Thigh_In": (8674, 8698),
            "Back_Knee_Out": (8690, 8676),
            "Back_Knee_In": (8698, 8676),
            "Back_Shin_Out": (8676, 8696),
            "Back_Shin_In": (8676, 8700),
        },
        

        'MUSCLE_INDICATORS': { # ★ 要手動確認/修正 (IDは自動割り当て) ★
            "Tibialis Anterior": {'emg_col': "L_TA_mean", 'type': 'midpoint', 'markers': [np.int64(8702), np.int64(8678)]},
            "Soleus": {'emg_col': "L_SOL_mean", 'type': 'centroid', 'markers': [np.int64(8676), np.int64(8700), np.int64(8696)]},
            "Rectus Femoris": {'emg_col': "L_RF_mean", 'type': 'centroid', 'markers': [np.int64(8688), np.int64(8692), np.int64(8682)]},
            "Vastus Lateralis": {'emg_col': "L_VL_mean", 'type': 'weighted_midpoint', 'markers': [np.int64(8692), np.int64(8682)], 'weight': 0.25},
            "Biceps Femoris": {'emg_col': "L_BF_mean", 'type': 'midpoint', 'markers': [np.int64(8674), np.int64(8690)]},
            "Semitendinosus": {'emg_col': "L_ST_mean", 'type': 'single', 'markers': [np.int64(8674)]},
            "Gluteus Maximus Area": {'emg_col': "L_GM_mean", 'type': 'offset', 'markers': [np.int64(8686), np.int64(8674), np.int64(8680)], 'weight': 0.3, 'ref_marker': [np.int64(8686), np.int64(8674)]},     
            "Iliopsoas Area": {'emg_col': "L_ILIO_mean", 'type': 'midpoint', 'markers': [np.int64(8688), np.int64(8682)]},
        },
        
        'NATURAL_LENGTHS': { # task2 と同じと仮定
            "Front_Upper_In": 260, "Front_Upper_Out": 305, "Front_Knee_Upper_Out": 75, "Front_Knee_Upper_In": 75,
            "Front_Knee_Lower_Out": 80, "Front_Knee_Lower_In": 80, "Front_Shin": 210, "Toe_Out": 50, "Toe_In": 60,
            "Back_Upper_In": 210, "Back_Upper_Out": 160, "Back_Thigh_Out": 265, "Back_Thigh_In": 220,
            "Back_Knee_Out": 160, "Back_Knee_In": 130, "Back_Shin_In": 130, "Back_Shin_Out": 130,
        },
        # --- ▼▼▼【トルク計算用定義 (task3)】▼▼▼ ---
        # ★ 新関節定義 (Task3)
        'JOINT_CENTER_DEFS': {
            # Hip: Mid(8682,8680) と Mid(8688,8686) を 1:3 に内分
            'Hip': {'type': 'ratio_1_3_between_mids', 'markers': [8682, 8680, 8688, 8686]},
            # Knee: Mid(8690, 8698)
            'Knee': {'type': 'midpoint', 'markers': [8690, 8698]},
            # Ankle: Mid( (8684-8696 2:1), (8694-8700 2:1) )
            'Ankle': {'type': 'mid_of_ratio_2_1', 'markers': [8684, 8696, 8694, 8700]}
        },
        
        # ★ トルク計算用セグメントマップ (Task3 ID - Task2を参考に推定)
        'SEGMENT_MAP': {
            'Foot':  [8684, 8694, 8696, 8700], # 推定
            'Shank': [8702, 8678, 8676], # 推定
            'Thigh': [8690, 8698, 8692, 8674 ] # 推定
        },

        'RUBBER_TORQUE_MAP': { # ★task1の構造に基づきIDを置換 (要確認)
            "Front_Upper_In": ("Hip", 8692, 8686), "Front_Upper_Out": ("Hip", 8692, 8682),
            "Front_Knee_Upper_Out": ("Knee", 8690, 8692), "Front_Knee_Upper_In":  ("Knee", 8698, 8692),
            "Front_Knee_Lower_Out": ("Knee", 8690, 8702), "Front_Knee_Lower_In":  ("Knee", 8698, 8702),
            "Front_Shin": ("Ankle", 8678, 8702), "Toe_Out": ("Ankle", 8696, 8678), "Toe_In": ("Ankle", 8700, 8678),
            "Back_Upper_In": ("Hip", 8674, 8680), "Back_Upper_Out": ("Hip", 8674, 8688),
            "Back_Thigh_Out": ("Knee", 8690, 8674), "Back_Thigh_In": ("Knee", 8698, 8674),
            "Back_Knee_Out": ("Knee", 8690, 8676), "Back_Knee_In": ("Knee", 8698, 8676),
            "Back_Shin_In": ("Ankle", 8684, 8676), "Back_Shin_Out": ("Ankle", 8694, 8676),
        },
        # --- ▲▲▲ トルク計算用定義ここまで ▲▲▲ ---
    }
}
# --- ▲▲▲ タスク別設定ここまで ▲▲▲ ---

