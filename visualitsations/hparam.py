import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "../exp1_hparams")
OUTPUT_DIR = os.path.join(_HERE, "../plots/hparams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG   = "white"
CARD_BG   = "white"
ACCENT_A  = "#2563eb"   
ACCENT_B  = "#dc2626"  
ACCENT_C  = "#16a34a"   
GRID_COL  = "#e5e7eb"
TEXT_COL  = "#111827"
MUTED     = "#6b7280"

DROPOUT_COLORS = {
    0.0: "#16a34a", 0.1: "#d97706", 0.3: "#2563eb", 0.5: "#dc2626"
}
BATCH_SIZES = {64: 80, 128: 200}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#d1d5db",
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#d1d5db",
    "legend.labelcolor": TEXT_COL,
    "savefig.facecolor": "white",
})


def load_experiments(pattern):
    records = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        d = json.load(open(fp))
        cfg = d["config"]
        test_accs  = [r["test"]["accuracy"]    for r in d["runs"]]
        val_accs   = [r["val"]["accuracy"]     for r in d["runs"]]
        macro_f1s  = [r["test"]["macro_f1"]    for r in d["runs"]]
        per_class_f1s = np.mean([r["test"]["f1"] for r in d["runs"]], axis=0)
        val_curves = [r["logs"]["val_acc"] for r in d["runs"]]

        records.append({
            "exp":          os.path.basename(fp).split("_")[1],
            "model":        cfg["model"],
            "batch_size":   cfg["batch_size"],
            "lr":           cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "dropout":      cfg["dropout"],
            "test_acc_mean": np.mean(test_accs),
            "test_acc_std":  np.std(test_accs),
            "val_acc_mean":  np.mean(val_accs),
            "macro_f1_mean": np.mean(macro_f1s),
            "per_class_f1":  per_class_f1s,
            "val_curves":    val_curves,
            "all_test_accs": test_accs,
        })
    return records

cnn_recs  = load_experiments("exp_*_cnn.json")
res_recs  = load_experiments("exp_*_resnet18.json")
all_recs  = cnn_recs + res_recs

cnn_all_test = [v for r in cnn_recs for v in r["all_test_accs"]]
res_all_test = [v for r in res_recs for v in r["all_test_accs"]]


def plot_model_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Test Accuracy Distribution — CNN vs ResNet18", fontsize=14, fontweight="bold", y=0.97)

    data   = [cnn_all_test, res_all_test]
    labels = ["CNN", "ResNet18"]
    colors = [ACCENT_A, ACCENT_B]

    for i, (vals, lbl, col) in enumerate(zip(data, labels, colors)):
        x = i + 1
        vp = ax.violinplot(vals, positions=[x], widths=0.5, showmedians=False, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(col); pc.set_alpha(0.45); pc.set_edgecolor(col)

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([x - 0.12, x + 0.12], [med, med], color=col, linewidth=2.5, zorder=5)
        ax.plot([x, x], [q1, q3], color=col, linewidth=6, alpha=0.6, solid_capstyle="round", zorder=4)
        ax.plot([x, x], [min(vals), max(vals)], color=col, linewidth=1.5, alpha=0.4, zorder=3)

        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter(x + jitter, vals, color=col, alpha=0.7, s=28, zorder=6, edgecolors="white", linewidths=0.4)

        ax.text(x, max(vals) + 0.002, f"max {max(vals):.4f}", ha="center", va="bottom",
                fontsize=8.5, color=col, fontweight="bold")
        ax.text(x, min(vals) - 0.003, f"min {min(vals):.4f}", ha="center", va="top",
                fontsize=8.5, color=col)

    ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Test Accuracy"); ax.grid(axis="y", alpha=0.4)
    ax.set_xlim(0.4, 2.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot1_model_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Plot 1 saved")


def plot_hp_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle("Hyperparameter Search — LR vs Test Accuracy", fontsize=14, fontweight="bold", y=0.99)

    for ax, recs, title, base_col in zip(axes, [cnn_recs, res_recs], ["CNN", "ResNet18"], [ACCENT_A, ACCENT_B]):
        for r in recs:
            col  = DROPOUT_COLORS[r["dropout"]]
            size = BATCH_SIZES[r["batch_size"]]
            ax.scatter(r["lr"], r["test_acc_mean"], c=col, s=size,
                       alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
            ax.errorbar(r["lr"], r["test_acc_mean"], yerr=r["test_acc_std"],
                        fmt="none", ecolor=col, alpha=0.4, elinewidth=1.2, capsize=3)

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)"); ax.set_ylabel("Mean Test Accuracy")
        ax.set_title(title, color="black"); ax.grid(alpha=0.35)

    dropout_handles = [mpatches.Patch(color=DROPOUT_COLORS[d], label=f"dropout={d}")
                       for d in sorted(DROPOUT_COLORS)]
    size_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=MUTED,
                           markersize=np.sqrt(s), label=f"batch size={bs}", linewidth=0)
                    for bs, s in BATCH_SIZES.items()]
    fig.legend(handles=dropout_handles + size_handles, fontsize=8.5, loc="lower center",
               ncol=6, bbox_to_anchor=(0.5, -0.06), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot2_hp_scatter.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Plot 2 saved")


def plot_topk_bar(k=8):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Top-{k} Experiments by Mean Test Accuracy", fontsize=14, fontweight="bold", y=0.99)
 
    for ax, recs, title, col in zip(axes, [cnn_recs, res_recs], ["CNN", "ResNet18"], [ACCENT_A, ACCENT_B]):
        top = sorted(recs, key=lambda r: r["test_acc_mean"], reverse=True)[:k]
        labels = [f"exp_{r['exp']}\nlr={r['lr']:.1e}\ndo={r['dropout']}" for r in top]
        means  = [r["test_acc_mean"] for r in top]
        stds   = [r["test_acc_std"]  for r in top]
 
        bars = ax.barh(range(k), means, xerr=stds, color=col, alpha=0.75,
                       error_kw={"ecolor": "black", "capsize": 4, "alpha": 0.7}, height=0.6)
        for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
            ax.text(m + s + 0.0005, i, f"{m:.4f}", va="center", fontsize=8, color=TEXT_COL)
 
        ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("Mean Test Accuracy"); ax.set_title(title, color="black")
        ax.set_xlim(min(means) - 0.01, max(means) + 0.015)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.35)
 
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot3_topk_bar.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Plot 3 saved")
 


def plot_learning_curves(top_n=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Validation Accuracy Curves — Top-{top_n} Configurations", fontsize=14, fontweight="bold", y=0.99)

    cmap_a = cm.Blues;  cmap_b = cm.Oranges

    for ax, recs, title, cmap, base_col in zip(
            axes, [cnn_recs, res_recs], ["CNN", "ResNet18"], [cmap_a, cmap_b], [ACCENT_A, ACCENT_B]):

        top = sorted(recs, key=lambda r: r["test_acc_mean"], reverse=True)[:top_n]
        colors = [cmap(0.45 + 0.55 * i / (top_n - 1)) for i in range(top_n)]

        for i, (r, col) in enumerate(zip(top, colors)):
            min_len    = min(len(c) for c in r["val_curves"])
            trimmed    = [c[:min_len] for c in r["val_curves"]]
            mean_curve = np.mean(trimmed, axis=0)
            std_curve  = np.std( trimmed, axis=0)
            epochs = range(1, len(mean_curve) + 1)
            ax.plot(epochs, mean_curve, color=col, linewidth=2,
                    label=f"exp_{r['exp']} lr={r['lr']:.1e}")
            ax.fill_between(epochs,
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            color=col, alpha=0.12)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
        ax.set_title(title, color="black"); ax.grid(alpha=0.35)
        ax.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot4_learning_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Plot 4 saved")


def plot_val_vs_test():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.suptitle("Val vs Test Accuracy — Generalization Check", fontsize=14, fontweight="bold", y=0.99)

    for recs, col, marker, label in [
        (cnn_recs, ACCENT_A, "o", "CNN"),
        (res_recs, ACCENT_B, "s", "ResNet18")
    ]:
        xs = [r["val_acc_mean"]  for r in recs]
        ys = [r["test_acc_mean"] for r in recs]
        xerr = [np.std([run["val"]["accuracy"] for run in
                        json.load(open(
                            glob.glob(os.path.join(DATA_DIR,
                            f"exp_{r['exp']}_{'cnn' if r['model']=='cnn' else 'resnet18'}.json"))[0]))["runs"]])
                for r in recs]
        yerr = [r["test_acc_std"] for r in recs]

        ax.scatter(xs, ys, c=col, s=70, marker=marker, alpha=0.8,
                   edgecolors="white", linewidths=0.4, label=label, zorder=4)
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=col, alpha=0.3, elinewidth=1, zorder=3)


    all_vals = [r["val_acc_mean"]  for r in all_recs]
    all_test = [r["test_acc_mean"] for r in all_recs]
    lo = min(min(all_vals), min(all_test)) - 0.003
    hi = max(max(all_vals), max(all_test)) + 0.003
    ax.plot([lo, hi], [lo, hi], "--", color=ACCENT_C, linewidth=1.2, alpha=0.6, label="y = x  (above = test > val, ResNet18 tends here)")
    ax.fill_between([lo, hi], [lo - 0.005, hi - 0.005], [lo + 0.005, hi + 0.005],
                    color=ACCENT_C, alpha=0.07)

    ax.set_xlabel("Mean Validation Accuracy"); ax.set_ylabel("Mean Test Accuracy")
    ax.grid(alpha=0.35)
    ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(facecolor=ACCENT_C, alpha=0.15, edgecolor="none",
                       label="±0.005 tolerance band around y = x"),
        Line2D([0],[0], color=ACCENT_A, lw=1, alpha=0.3,
               marker="o", markersize=0, label="Error bars: x = val std, y = test std (CNN)"),
        Line2D([0],[0], color=ACCENT_B, lw=1, alpha=0.3,
               marker="s", markersize=0, label="Error bars: x = val std, y = test std (ResNet18)"),
    ]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.14), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plot5_val_vs_test.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✓ Plot 5 saved")


if __name__ == "__main__":
    print("Generating plots...\n")
    plot_model_comparison()
    plot_hp_scatter()
    plot_topk_bar()
    plot_learning_curves()
    plot_val_vs_test()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")