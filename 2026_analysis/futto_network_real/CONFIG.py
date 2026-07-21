# =============================================================================
# futto_common/CONFIG.py  (共通パッケージ版)
#
# 変更点: なし
#   strength_visualize/CONFIG.py と完全に同一内容。
#   他のプログラムはすべてここを参照する。
#
#   from futto_common import CONFIG as config
# =============================================================================

import numpy as np
from pathlib import Path

# --- ▼▼▼ 基本設定 (全タスク共通) ▼▼▼ ---
RUBBER_PROPERTIES_EXCEL_PATH = r"C:\FuttoAnalysis\rubber_strength.xlsx"
RUBBER_PROPERTIES_SHEET_NAME = 'Sheet1'
RESULT_DIR = r"C:\FuttoAnalysis\result"

BASE_DIR = Path(r"C:\Users\ihika\2026_experiment")
SUBJECTS = ['Ide']

TASKS = ['task01', 'task02', 'task03']

TASK_TITLES = {
    'task01': 'Task 01 (normal)',
    'task02': 'Task 02 (enhance)',
    'task03': 'Task 03 (without)'
}

FRAME_RATE = 100
TIME_OFFSET = 0.0

PHASES = {
    1: {'name': '0.7m/s', 'start': 40.0,  'end': 100.0},
    2: {'name': '0.9m/s', 'start': 100.0, 'end': 160.0},
    3: {'name': '1.1m/s', 'start': 160.0, 'end': 220.0},
    4: {'name': '1.3m/s', 'start': 220.0, 'end': 280.0},
    5: {'name': '1.5m/s', 'start': 280.0, 'end': 340.0}
}

# --- 筋電図（EMG）設定 ---
MUSCLE_NAMES_BASE = ['GM','ILIO','ST', 'RF','VL','BF','SOL','TA']
MUSCLE_NAMES_R = [f'R_{name}' for name in MUSCLE_NAMES_BASE]
MUSCLE_NAMES_L = [f'L_{name}' for name in MUSCLE_NAMES_BASE]
MUSCLE_NAMES = MUSCLE_NAMES_R + MUSCLE_NAMES_L
EMG_CHANNELS = len(MUSCLE_NAMES)

FS_EMG = 2000
FS_FS  = 1000

EMG_NOTCH_FREQS    = [59.0, 61.0]
EMG_BANDPASS_FREQS = [20.0, 450.0]
EMG_LOWPASS_FREQ   = 10.0
FILTER_ORDER       = 4

FS_LZ_COL = 10
FS_RZ_COL = 4

PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
MUSCLE_MARKER_BASE_SIZE   = 5
MUSCLE_MARKER_SCALE_FACTOR = 25

# --- 剛体・追跡設定 ---
PHASE_WEIGHTING = {
    'rigid_phase_ranges':   [(0.0, 50.0), (95.0, 100.0)],
    'rigid_weight':         0.8,
    'flexible_phase_range': (50.0, 95.0),
    'flexible_weight':      0.2,
    'transition_ranges':    [(50.0, 50.0), (95.0, 95.0)]
}

CHAIN_HIERARCHY  = {'Hip': None, 'Thigh': 'Hip', 'Knee': 'Thigh', 'Shank': 'Knee', 'Foot': 'Shank'}
PROCESSING_ORDER = ['Hip', 'Thigh', 'Knee', 'Shank', 'Foot']
MATCHING_THRESHOLD_MM = 75.0

# --- 張力グラフ出力用グループ ---
SEGMENT_GROUPS = {
    'FK':      ["Front_Knee_Upper_Out", "Front_Knee_Upper_In",
                "Front_Knee_Lower_Out", "Front_Knee_Lower_In"],
    'FH_BK_BS':["Front_Upper_In", "Front_Upper_Out",
                "Back_Knee_Out",  "Back_Knee_In",
                "Back_Shin_In",   "Back_Shin_Out"],
    'BH_BT_FA':["Back_Upper_In",  "Back_Upper_Out",
                "Back_Thigh_Out", "Back_Thigh_In",
                "Front_Shin",     "Toe_Out", "Toe_In"]
}

# --- タスク別設定 ---
TASK_CONFIGS = {
    'task01': {
        'OPTI_CSV_PATH':        r"C:\FuttoAnalysis\opti\20260217\task01.csv",
        'OUTPUT_CSV_PATH':      r"C:\FuttoAnalysis\opti\20260217\task01_corrected_D.csv",
        'LABCHART_CYCLES_PATH': r"C:\FuttoAnalysis\labchart\20260217\task01_gait_cycles.csv",
        'MEAN_CYCLE_BASE_PATH': r"C:\FuttoAnalysis\opti\20260217\task01_mean_cycle",
        'TENSION_DATA_BASE_PATH': r"C:\FuttoAnalysis\opti\20260217\task01_tension_data",

        'REFERENCE_MARKER_ID': 16000,

        'STATIC_START': 0.0, 'STATIC_END': 10.0,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 10.0,
        'T2_WALK_END': 50.0,   'T2_STATIC_START': 50.0,

        'FORCE_MULTIPLIER': 1.0,
        'NATURAL_LENGTHS': {
            "Front_Upper_In": 160.0,  "Front_Upper_Out": 220.0,
            "Front_Knee_Upper_Out": 95.0,  "Front_Knee_Upper_In": 95.0,
            "Front_Knee_Lower_Out": 100.0, "Front_Knee_Lower_In": 106.0,
            "Front_Shin": 207.0, "Toe_Out": 105.0, "Toe_In": 136.0,
            "Back_Upper_In": 270.0,  "Back_Upper_Out": 200.0,
            "Back_Thigh_Out": 200.0, "Back_Thigh_In": 200.0,
            "Back_Knee_Out": 225.0,  "Back_Knee_In": 185.0,
            "Back_Shin_Out": 136.0,  "Back_Shin_In": 173.0
        },

        'SEGMENTS': {
            'Hip':   [16000, 16012, 16014, 15960],
            'Thigh': [15970, 15958],
            'Knee':  [15956, 15968],
            'Shank': [15974],
            'Foot':  [15950, 15972, 15964, 15918, 15966, 15948]
        },

        'KEYFRAME_MAP': {
            16000: 16000, 16012: 16012, 16014: 16014, 15960: 15960,
            15970: 15970, 15958: 15958, 15956: 15956, 15968: 15968,
            15974: 15974, 15950: 15950, 15972: 15972, 15964: 15964,
            15918: 15918, 15966: 15966, 15948: 15948,
        },

        'LINES_TO_DRAW': {
            "Front_Upper_In":       (16000, 15970),
            "Front_Upper_Out":      (16012, 15970),
            "Front_Knee_Upper_Out": (15970, 15956),
            "Front_Knee_Upper_In":  (15970, 15968),
            "Front_Knee_Lower_Out": (15974, 15956),
            "Front_Knee_Lower_In":  (15974, 15968),
            "Front_Shin":           (15974, 15950),
            "Toe_Out":              (15950, 15964),
            "Toe_In":               (15950, 15918),
            "Back_Upper_In":        (15960, 15958),
            "Back_Upper_Out":       (16014, 15958),
            "Back_Thigh_Out":       (15958, 15956),
            "Back_Thigh_In":        (15958, 15968),
            "Back_Knee_Out":        (15956, 15972),
            "Back_Knee_In":         (15968, 15972),
            "Back_Shin_Out":        (15972, 15948),
            "Back_Shin_In":         (15972, 15966),
        },

        'MUSCLE_INDICATORS': {
            "Tibialis Anterior":  {'emg_col': "L_TA_mean",   'type': 'midpoint',          'markers': [15974, 15950]},
            "Soleus":             {'emg_col': "L_SOL_mean",  'type': 'centroid',           'markers': [15972, 15964, 15918]},
            "Rectus Femoris":     {'emg_col': "L_RF_mean",   'type': 'centroid',           'markers': [16014, 15970, 15960]},
            "Vastus Lateralis":   {'emg_col': "L_VL_mean",   'type': 'weighted_midpoint',  'markers': [15960, 15970], 'weight': 0.25},
            "Biceps Femoris":     {'emg_col': "L_BF_mean",   'type': 'midpoint',           'markers': [15958, 15968]},
            "Semitendinosus":     {'emg_col': "L_ST_mean",   'type': 'single',             'markers': [15958]},
            "Gluteus Maximus Area":{'emg_col':"L_GM_mean",   'type': 'offset',             'markers': [16012, 15958, 16000], 'weight': 0.3, 'ref_marker': [16012, 15958]},
            "Iliopsoas Area":     {'emg_col': "L_ILIO_mean", 'type': 'midpoint',           'markers': [16014, 15960]},
        },

        # ★ 仮想関節定義 (Task01)
        'JOINT_CENTER_DEFS': {
            'Hip':   {'type': 'ratio_1_3_between_mids', 'markers': [16012, 16014, 16000, 15960]},
            'Knee':  {'type': 'midpoint',               'markers': [15956, 15968]},
            'Ankle': {'type': 'mid_of_ratio_2_1',       'markers': [15964, 15948, 15918, 15966]},
        },
    },

    'task02': {
        'OPTI_CSV_PATH':        r"C:\FuttoAnalysis\opti\20260217\task02.csv",
        'OUTPUT_CSV_PATH':      r"C:\FuttoAnalysis\opti\20260217\task02_corrected_D.csv",
        'LABCHART_CYCLES_PATH': r"C:\FuttoAnalysis\labchart\20260217\task02_gait_cycles.csv",
        'MEAN_CYCLE_BASE_PATH': r"C:\FuttoAnalysis\opti\20260217\task02_mean_cycle",
        'TENSION_DATA_BASE_PATH': r"C:\FuttoAnalysis\opti\20260217\task02_tension_data",

        'REFERENCE_MARKER_ID': 57960,

        'STATIC_START': 0.0, 'STATIC_END': 10.0,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 10.0,
        'T2_WALK_END': 50.0,   'T2_STATIC_START': 50.0,

        'FORCE_MULTIPLIER': 2.0,
        'NATURAL_LENGTHS': {
            "Front_Upper_In": 150.0,       "Front_Upper_Out": 220.0,
            "Front_Knee_Upper_Out": 165.0, "Front_Knee_Upper_In": 168.0,
            "Back_Knee_Out": 133.0,        "Back_Knee_In": 133.0,
            "Back_Shin_Out": 222.0,        "Back_Shin_In": 222.0
        },

        'SEGMENTS': {
            'Hip':   [57960, 57958],
            'Thigh': [],
            'Knee':  [57956, 57948, 57952],
            'Shank': [57946],
            'Foot':  [57954, 57950]
        },

        'KEYFRAME_MAP': {
            57960: 57960, 57958: 57958, 57956: 57956, 57948: 57948,
            57952: 57952, 57946: 57946, 57954: 57954, 57950: 57950,
        },

        'LINES_TO_DRAW': {
            "Front_Upper_In":       (57960, 57956),
            "Front_Upper_Out":      (57958, 57956),
            "Front_Knee_Upper_Out": (57956, 57952),
            "Front_Knee_Upper_In":  (57956, 57948),
            "Back_Knee_Out":        (57952, 57946),
            "Back_Knee_In":         (57948, 57946),
            "Back_Shin_Out":        (57946, 57954),
            "Back_Shin_In":         (57946, 57950),
        },

        'MUSCLE_INDICATORS': {
            "Tibialis Anterior":   {'emg_col': "L_TA_mean",   'type': 'midpoint',                     'markers': [57946, 57950]},
            "Soleus":              {'emg_col': "L_SOL_mean",  'type': 'centroid',                     'markers': [57946, 57954, 57950]},
            "Rectus Femoris":      {'emg_col': "L_RF_mean",   'type': 'centroid',                     'markers': [57958, 57956, 57960]},
            "Vastus Lateralis":    {'emg_col': "L_VL_mean",   'type': 'weighted_midpoint',            'markers': [57956, 57958], 'weight': 0.25},
            "Biceps Femoris":      {'emg_col': "L_BF_mean",   'type': 'midpoint',                     'markers': [57958, 57952]},
            "Semitendinosus":      {'emg_col': "L_ST_mean",   'type': 'double_midpoint_interpolation','markers': [57960, 57958, 57952, 57948], 'weight': 0.8},
            "Gluteus Maximus Area":{'emg_col': "L_GM_mean",   'type': 'double_midpoint_interpolation','markers': [57960, 57958, 57952, 57948], 'weight': 0.2},
            "Iliopsoas Area":      {'emg_col': "L_ILIO_mean", 'type': 'midpoint',                     'markers': [57958, 57960]},
        },

        # ★ 仮想関節定義 (Task02)
        'JOINT_CENTER_DEFS': {
            # Hip: 57960 と 57958 の中点 M から X軸(進行)方向に -30mm ずらした点
            #   offset_x < 0 = 後方にずらす (X=進行正のため)
            'Hip':   {'type': 'midpoint_offset_x',
                      'markers': [57960, 57958],
                      'offset_x': -30.0},
            # Knee: Mid(57948, 57952)
            'Knee':  {'type': 'midpoint', 'markers': [57948, 57952]},
            # Ankle: Mid(57950, 57954)
            'Ankle': {'type': 'midpoint', 'markers': [57950, 57954]},
        },
    },

    # =========================================================================
    # task03 : Futto非着用・マーカー5個 (関節直接配置)
    # =========================================================================
    # マーカー配置:
    #   67628 : 股関節 (Hip)
    #   67626 : 膝関節 (Knee)
    #   67632 : 足関節 (Ankle)
    #   67630 : 踵     (Heel) ← 進行方向後ろ (Z小)
    #   67634 : つま先 (Toe)  ← 進行方向前  (Z大)
    #
    # Futto非着用のため NATURAL_LENGTHS / LINES_TO_DRAW は空。
    # JOINT_CENTER_DEFS は実測値をそのまま参照する 'single' タイプ。
    # =========================================================================
    'task03': {
        'OPTI_CSV_PATH':          r"C:\FuttoAnalysis\opti\20260217\task03.csv",
        'OUTPUT_CSV_PATH':        r"C:\FuttoAnalysis\opti\20260217\task03_corrected_D.csv",
        'LABCHART_CYCLES_PATH':   r"C:\FuttoAnalysis\labchart\20260217\task03_gait_cycles.csv",

        'MEAN_CYCLE_BASE_PATH':   r"C:\FuttoAnalysis\opti\20260217\task03_mean_cycle",
        'TENSION_DATA_BASE_PATH': r"C:\FuttoAnalysis\opti\20260217\task03_tension_data",

        'REFERENCE_MARKER_ID': 67628,   # 股関節マーカーを基準点とする

        'STATIC_START': 0.0, 'STATIC_END': 10.0,
        'T1_STATIC_END': 10.0, 'T1_WALK_START': 10.0,
        'T2_WALK_END': 50.0,   'T2_STATIC_START': 50.0,

        # Futto非着用 → ゴム張力計算は不要
        'FORCE_MULTIPLIER': 0.0,
        'NATURAL_LENGTHS': {},

        'SEGMENTS': {
            'Hip':   [67628],
            'Thigh': [],
            'Knee':  [67626],
            'Shank': [67632],
            'Foot':  [67634, 67630],
        },

        'KEYFRAME_MAP': {
            67628: 67628,
            67626: 67626,
            67632: 67632,
            67634: 67634,
            67630: 67630,
        },

        # Resync判別モード: 解剖学的座標ルールによる判別
        #   Y降順: 67628(Hip) > 67626(Knee) > 67632(Ankle) > 67634/67630
        #   Z降順: 67634(Toe,Z大) > 67630(Heel,Z小)
        'RESYNC_MODE': 'anatomical',

        'LINES_TO_DRAW': {},    # Futto非着用 → ゴム線なし

        'MUSCLE_INDICATORS': {
            "Tibialis Anterior":   {'emg_col': "L_TA_mean",   'type': 'midpoint',         'markers': [67626, 67632]},
            "Soleus":              {'emg_col': "L_SOL_mean",  'type': 'centroid',          'markers': [67626, 67634, 67630]},
            "Rectus Femoris":      {'emg_col': "L_RF_mean",   'type': 'midpoint',          'markers': [67628, 67626]},
            "Vastus Lateralis":    {'emg_col': "L_VL_mean",   'type': 'weighted_midpoint', 'markers': [67628, 67626], 'weight': 0.25},
            "Biceps Femoris":      {'emg_col': "L_BF_mean",   'type': 'midpoint',          'markers': [67628, 67626]},
            "Semitendinosus":      {'emg_col': "L_ST_mean",   'type': 'midpoint',          'markers': [67628, 67626]},
            "Gluteus Maximus Area":{'emg_col': "L_GM_mean",   'type': 'single',            'markers': [67628]},
            "Iliopsoas Area":      {'emg_col': "L_ILIO_mean", 'type': 'single',            'markers': [67628]},
        },

        # ★ 実測関節定義 (Task03)
        # マーカーが直接関節上にあるため 'single' でIDを直接参照するだけ。
        # Heel / Toe は関節中心ではないが、足部解析用に追加。
        'JOINT_CENTER_DEFS': {
            'Hip':   {'type': 'single', 'markers': [67628]},
            'Knee':  {'type': 'single', 'markers': [67626]},
            'Ankle': {'type': 'single', 'markers': [67632]},
            'Heel':  {'type': 'single', 'markers': [67630]},
            'Toe':   {'type': 'single', 'markers': [67634]},
        },
    },
}