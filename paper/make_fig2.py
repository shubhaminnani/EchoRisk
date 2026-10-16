"""Regenerate the Task 1-3 validation-result figures.

Produces THREE separate PNGs (no titles), base font size 18:
  fig2a_task1_scatter.png  — predicted vs true LVEF, both backbones
  fig2b_task2_roc.png      — Task 2 ROC, both backbones
  fig2c_task3_roc.png      — Task 3 ROC, both backbones, FPR=10% marker

Reads the real validation prediction CSVs + ground-truth labels.
Run with the `challenge` conda env python.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = "/mnt/e/challenge/echo_risk"
OUT = os.path.join(ROOT, "paper", "figures")
KEY = ["patient_id", "timepoint"]

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

BASE = "#4C72B0"   # R(2+1)D-18
ECHO = "#C44E52"   # EchoPrime


def _merge(pred_csv, gt_csv, pred_col, gt_col):
    p = pd.read_csv(pred_csv)[KEY + [pred_col]]
    g = pd.read_csv(gt_csv)[KEY + [gt_col]]
    return p.merge(g, on=KEY)


# ---------------- Panel A: Task 1 LVEF scatter ----------------
def task1():
    b = _merge(f"{ROOT}/task1_val_preds.csv",
               f"{ROOT}/data/task1/Task1/public/labels/task1_val.csv",
               "lvef_pred", "lvef")
    e = _merge(f"{ROOT}/task1_val_preds_echoprime.csv",
               f"{ROOT}/data/task1/Task1/public/labels/task1_val.csv",
               "lvef_pred", "lvef")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(b["lvef"], b["lvef_pred"], s=28, alpha=0.6, color=BASE,
               label="R(2+1)D-18", edgecolors="none")
    ax.scatter(e["lvef"], e["lvef_pred"], s=28, alpha=0.6, color=ECHO,
               label="EchoPrime", edgecolors="none")
    lo = min(b["lvef"].min(), b["lvef_pred"].min(), e["lvef_pred"].min())
    hi = max(b["lvef"].max(), b["lvef_pred"].max(), e["lvef_pred"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.5)
    ax.set_xlabel("True LVEF (%)")
    ax.set_ylabel("Predicted LVEF (%)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2a_task1_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2a_task1_scatter.png")


# ---------------- ROC helper ----------------
def _roc_panel(task_no, pred_col, gt_col, out_name, mark_fpr=None):
    b = _merge(f"{ROOT}/task{task_no}_val_preds.csv",
               f"{ROOT}/data/task{task_no}/Task{task_no}/public/labels/task{task_no}_val.csv",
               pred_col, gt_col)
    e = _merge(f"{ROOT}/task{task_no}_val_preds_echoprime.csv",
               f"{ROOT}/data/task{task_no}/Task{task_no}/public/labels/task{task_no}_val.csv",
               pred_col, gt_col)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for df, color, name in [(b, BASE, "R(2+1)D-18"), (e, ECHO, "EchoPrime")]:
        y, s = df[gt_col].values, df[pred_col].values
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    if mark_fpr is not None:
        ax.axvline(mark_fpr, color="black", linestyle=":", linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{out_name}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_name}")


if __name__ == "__main__":
    task1()
    _roc_panel(2, "dysfunction_pred", "lv_dysfunction", "fig2b_task2_roc.png")
    _roc_panel(3, "cardiotoxicity_pred", "cardiotoxicity", "fig2c_task3_roc.png", mark_fpr=0.10)
