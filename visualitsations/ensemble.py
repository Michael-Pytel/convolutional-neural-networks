import json, os
import numpy as np
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_HERE, '..', "exp4_ensemble", "final_ensemble.json")
OUTPUT_DIR = os.path.join(_HERE, "../plots/ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw = json.load(open(DATA_FILE))

METHODS = {}
METHODS["Hard voting"]     = raw["hard"]
METHODS["Soft voting"]     = raw["soft"]
METHODS["Weighted voting"] = raw["weighted"]
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
    "savefig.facecolor": "white",
})


def plot_accuracy_bar():
    names  = list(METHODS_SORTED.keys())
    accs   = [METHODS_SORTED[n]["accuracy"] for n in names]

    x   = np.arange(len(names))
    w   = 0.55
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Ensemble Methods — Test Accuracy",
                 fontsize=13, fontweight="bold")

    bars = ax.bar(x, accs, w, color="#4878d0", edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ensemble_plot1_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Plot saved")


if __name__ == "__main__":
    plot_accuracy_bar()
    print(f"\nPlot saved to {OUTPUT_DIR}/")