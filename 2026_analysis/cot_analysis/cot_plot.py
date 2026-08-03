# =============================================================================
# cot_analysis/cot_plot.py
#
# 役割:
#   COT-Fr散布図(全被験者・全日程・タスク別色分けで1枚に重ね描き)と、
#   速度別COT箱ひげ図(タスク別色分け)を描画する。
# =============================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cot_analysis.cot_calc import fit_cot_vs_froude

# タスク(task01/02/03) の色・表示ラベル (CONFIG.TASK_CONDITION_LABELS と対応)
TASK_COLORS = {'task01': 'tab:blue', 'task02': 'tab:orange', 'task03': 'tab:green'}
TASK_LABELS = {'task01': 'T1', 'task02': 'T2', 'task03': 'N'}


def plot_cot_froude_overlay(df_all, save_path=None):
    """
    全被験者・全日程のCOT-Frデータをタスク別に色分けして1枚のグラフに重ね描きする。
    各タスクごとに COT(Fr) = a/Fr + b*Fr をフィッティングし、最小点をプロットする。

    Parameters
    ----------
    df_all : pd.DataFrame  列 'task', 'froude', 'cot' を含む
    save_path : str/Path または None
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for task in sorted(df_all['task'].unique()):
        sub = df_all[df_all['task'] == task].dropna(subset=['froude', 'cot'])
        if sub.empty:
            continue
        color = TASK_COLORS.get(task, 'gray')
        label = TASK_LABELS.get(task, task)
        ax.plot(sub['froude'], sub['cot'], marker='o', markerfacecolor='white',
                markeredgecolor=color, linestyle='', label=label)

        try:
            fit = fit_cot_vs_froude(sub['froude'].values, sub['cot'].values)
            x_fit = np.linspace(sub['froude'].min(), sub['froude'].max(), 100)
            ax.plot(x_fit, fit['a'] / x_fit + fit['b'] * x_fit, color=color, linestyle='-')
            ax.plot(fit['fr_min'], fit['cot_min'], marker='o', markersize=8, color=color)
            ax.text(fit['fr_min'], fit['cot_min'], f"  Fr={fit['fr_min']:.2f}",
                    color=color, fontsize=9, va='bottom')
        except RuntimeError as e:
            print(f"  [警告] {label}: カーブフィッティングをスキップしました ({e})")

    ax.set_xlabel("Froude number Fr [-]", fontsize=12)
    ax.set_ylabel("Cost of Transport COT [-]", fontsize=12)
    ax.legend(title="Condition")
    ax.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> COT-Fr重ね描きグラフを保存しました: {save_path}")
    return fig


def plot_cot_boxplot_by_speed(df_all, save_path=None):
    """速度ごとのCOT箱ひげ図 (タスク別に色分け) を描画する。"""
    fig, ax = plt.subplots(figsize=(11, 6))
    order = sorted(df_all['speed'].unique())
    palette = {t: TASK_COLORS.get(t, 'gray') for t in df_all['task'].unique()}

    sns.boxplot(x='speed', y='cot', hue='task', data=df_all, order=order,
                palette=palette, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    labels = [TASK_LABELS.get(l, l) for l in labels]
    ax.legend(handles, labels, title="Condition")

    ax.set_xlabel("Walking speed v [m/s]", fontsize=12)
    ax.set_ylabel("Cost of Transport COT [-]", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> 速度別COT箱ひげ図を保存しました: {save_path}")
    return fig
