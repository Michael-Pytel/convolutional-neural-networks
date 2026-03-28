import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_HERE, '..',"exp4_ensemble" ,"final_ensemble.json")
OUTPUT_DIR = os.path.join(_HERE, "../plots/ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)


raw = json.load(open(DATA_FILE))

METHODS = {}
METHODS["Hard voting"]    = raw["hard"]
METHODS["Soft voting"]    = raw["soft"]
for k, v in raw["stacking"].items():
    label = {
        "logreg": "Stack — LogReg",
        "ridge":  "Stack — Ridge",
        "rf":     "Stack — RF",
        "gb":     "Stack — GradBoost",
        "svm":    "Stack — SVM",
        "knn":    "Stack — KNN",
    }[k]
    METHODS[label] = v


METHODS_SORTED = dict(sorted(METHODS.items(),
                             key=lambda x: x[1]["accuracy"], reverse=True))

N_CLASSES   = 10
CLASS_LABELS = [f"Class {i}" for i in range(N_CLASSES)]


def method_color(name):
    if "Stack" in name:
        palette = ["#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1", "#ff9da7"]
        idx = list(METHODS.keys()).index(name) - 2   
        return palette[idx % len(palette)]
    return {"Hard voting": "#4878d0", "Soft voting": "#2563eb"}[name]

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#d1d5db",
    "axes.labelcolor":   "#111827",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#111827",
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         9.5,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#d1d5db",
    "savefig.facecolor": "white",
})



def plot_accuracy_bar():
    names  = list(METHODS_SORTED.keys())
    accs   = [METHODS_SORTED[n]["accuracy"]  for n in names]
    f1s    = [METHODS_SORTED[n]["macro_f1"]  for n in names]
    colors = [method_color(n) for n in names]

    x   = np.arange(len(names))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Ensemble Methods — Test Accuracy & Macro F1 (ranked)",
                 fontsize=13, fontweight="bold")

    b1 = ax.bar(x - w/2, accs, w, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3, label="Accuracy", alpha=0.9)
    b2 = ax.bar(x + w/2, f1s,  w, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3, label="Macro F1",  alpha=0.55,
                hatch="//")

    for bar, v in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0004,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, v in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0004,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color="#4b5563")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(min(accs + f1s) - 0.01, max(accs + f1s) + 0.008)
    ax.grid(axis="y", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor="#9ca3af", edgecolor="white",         label="Accuracy (solid)"),
        mpatches.Patch(facecolor="#9ca3af", edgecolor="white", hatch="//", label="Macro F1 (hatched)"),
        mpatches.Patch(facecolor="#4878d0", edgecolor="white",         label="Voting methods"),
        mpatches.Patch(facecolor="#f28e2b", edgecolor="white",         label="Stacking methods"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_plot1_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 1 saved")


def plot_macro_prf():
    names = list(METHODS_SORTED.keys())
    prec  = [METHODS[n]["macro_precision"] for n in names]
    rec   = [METHODS[n]["macro_recall"]    for n in names]
    f1    = [METHODS[n]["macro_f1"]        for n in names]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Macro Precision / Recall / F1 per Ensemble Method",
                 fontsize=13, fontweight="bold")

    ax.bar(x - w,  prec, w, label="Macro Precision", color="#4878d0",
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x,      rec,  w, label="Macro Recall",    color="#e15759",
           edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar(x + w,  f1,   w, label="Macro F1",        color="#59a14f",
           edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    all_vals = prec + rec + f1
    ax.set_ylim(min(all_vals) - 0.01, max(all_vals) + 0.006)
    ax.grid(axis="y", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_plot2_macro_prf.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 2 saved")


def plot_f1_delta():
    baseline_f1 = np.array(METHODS["Hard voting"]["f1"])
    compare = {k: v for k, v in METHODS.items() if k != "Hard voting"}
    names   = list(compare.keys())
    deltas  = np.array([np.array(compare[n]["f1"]) - baseline_f1 for n in names])

    abs_max = max(abs(deltas.min()), abs(deltas.max()))
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Per-Class F1 Δ vs Hard Voting Baseline",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(deltas, cmap="RdYlGn",
                   vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im, ax=ax, label="ΔF1 (positive = better than hard voting)",
                 fraction=0.025, pad=0.02)

    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    for i in range(len(names)):
        for j in range(N_CLASSES):
            v = deltas[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > abs_max * 0.55 else "#111827")

    ax.axhline(0.5, color="white", linewidth=2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_plot4_f1_delta.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 4 saved")

if __name__ == "__main__":
    print("Generating ensemble plots...\n")
    plot_accuracy_bar()
    plot_macro_prf()
    plot_f1_delta()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")