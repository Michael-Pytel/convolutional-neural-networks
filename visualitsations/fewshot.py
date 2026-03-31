import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


_HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_FILE    = os.path.join(_HERE, '..', 'exp3_fewshot', "protonet.json")
CLASSIC_DIR  = os.path.join(_HERE, '..', 'exp3_fewshot', "classic")
OUTPUT_DIR   = os.path.join(_HERE, '..', 'plots', "fewshot")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_classic():
    mapping = {"cnn": "exp_0_cnn.json", "resnet18": "exp_1_resnet18.json"}
    out = {}
    for model, fname in mapping.items():
        runs = json.load(open(os.path.join(CLASSIC_DIR, fname)))["runs"]
        accs = [r["test"]["accuracy"] for r in runs]
        out[model] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    return out

CLASSIC_REF   = _load_classic()
RANDOM_CHANCE = 0.10


MODEL_COLS  = {"resnet18": "#2563eb", "cnn": "#dc2626"}
MODEL_FILLS = {"resnet18": "#bfdbfe", "cnn": "#fecaca"}

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

raw = json.load(open(DATA_FILE))

data = {}
for entry in raw:
    m, k = entry["model"], entry["k_shot"]
    data.setdefault(m, {})[k] = entry["runs"]

models  = ["resnet18", "cnn"]
k_shots = [5, 10, 15]


def plot_shot_efficiency():
    # Groups: one per k-shot + one "classic" group on the right
    group_labels = [f"{k}-shot" for k in k_shots] + ["Classic\n(supervised)"]
    n_groups = len(group_labels)
    x = np.arange(n_groups)
    w = 0.35
    err_kw = {"ecolor": "#374151", "capsize": 5,
              "elinewidth": 1.4, "capthick": 1.4}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("ProtoNet Few-Shot vs Classic Supervised — Test Accuracy",
                 fontsize=13, fontweight="bold")

    for i, model in enumerate(models):
        col   = MODEL_COLS[model]
        fill  = MODEL_FILLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"
        offset = (i - 0.5) * w

        # Few-shot bars
        means = [np.mean([r["test_acc"] for r in data[model][k]]) for k in k_shots]
        stds  = [np.std( [r["test_acc"] for r in data[model][k]]) for k in k_shots]

        # Classic bar
        cl_m = CLASSIC_REF[model]["mean"]
        cl_s = CLASSIC_REF[model]["std"]
        means_all = means + [cl_m]
        stds_all  = stds  + [cl_s]

        bars = ax.bar(x + offset, means_all, w,
                      color=[col] * len(k_shots) + [fill],
                      yerr=stds_all, error_kw=err_kw,
                      edgecolor=[col] * n_groups, linewidth=1.2,
                      zorder=3, alpha=0.88,
                      label=f"{label} (few-shot)")
        # Hatch the classic bar to distinguish it
        bars[-1].set_hatch("///")
        bars[-1].set_label(f"_nolegend_")

        for bar, m, s in zip(bars, means_all, stds_all):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.012,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=col, fontweight="bold")

    # Separator line between few-shot and classic groups
    ax.axvline(len(k_shots) - 0.5, color="#9ca3af", lw=1, linestyle="--", alpha=0.5)
    ax.axhline(RANDOM_CHANCE, color="#9ca3af", lw=1.2, linestyle=":",
               zorder=2, label="Random chance (0.10)")

    # Hatch patch for legend
    from matplotlib.patches import Patch
    classic_patch = Patch(facecolor="#e5e7eb", edgecolor="#374151",
                          hatch="///", label="Classic supervised (mean ± std)")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=10.5)
    ax.set_xlabel("Setting")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.10)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(classic_patch)
    fig.legend(handles=handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.06), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot1_efficiency.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 1 saved")


def plot_learning_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("ProtoNet — Train Loss & Val Accuracy per Model & k-Shot",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, model in enumerate(models):
        col  = MODEL_COLS[model]
        fill = MODEL_FILLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"

        for ci, k in enumerate(k_shots):
            ax   = axes[row, ci]
            runs = data[model][k]

            train_losses = np.array([r["logs"]["train_loss"] for r in runs])
            val_accs     = np.array([r["logs"]["val_acc"]    for r in runs])
            eps = range(1, train_losses.shape[1] + 1)

            ax2 = ax.twinx()

            m_loss = train_losses.mean(0); s_loss = train_losses.std(0)
            ax.plot(eps, m_loss, color=col, lw=2, label="Train loss")
            ax.fill_between(eps, m_loss - s_loss, m_loss + s_loss,
                            color=fill, alpha=0.3)
            ax.set_ylabel("Train Loss", color=col, fontsize=8.5)
            ax.tick_params(axis="y", labelcolor=col)

            m_vacc = val_accs.mean(0); s_vacc = val_accs.std(0)
            ax2.plot(eps, m_vacc, color="#f59e0b", lw=2, linestyle="--")
            ax2.fill_between(eps, m_vacc - s_vacc, m_vacc + s_vacc,
                             color="#fde68a", alpha=0.3)
            ax2.set_ylabel("Val Accuracy", color="#f59e0b", fontsize=8.5)
            ax2.tick_params(axis="y", labelcolor="#f59e0b")

            ax.set_title(f"{label} — {k}-shot", color="black")
            ax.set_xlabel("Episode")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.35)
            ax.spines[["top"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0],[0], color="#2563eb", lw=2,         label="Train loss (ResNet18)"),
        Line2D([0],[0], color="#dc2626", lw=2,         label="Train loss (CNN)"),
        Line2D([0],[0], color="#f59e0b", lw=2, ls="--",label="Val accuracy (mean)"),
        Patch(facecolor="#fde68a", alpha=0.5, edgecolor="#f59e0b",
              label="Val accuracy ± std across runs"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.03), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot2_curves.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 2 saved")


def plot_train_val_gap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ProtoNet — Train vs Val Accuracy (Support-Set Overfitting)",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, models):
        col  = MODEL_COLS[model]
        fill = MODEL_FILLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"

        for k, ls in zip(k_shots, ["-", "--", "-."]):
            runs       = data[model][k]
            train_accs = np.array([r["logs"]["train_acc"] for r in runs])
            val_accs   = np.array([r["logs"]["val_acc"]   for r in runs])
            eps        = range(1, train_accs.shape[1] + 1)

            m_train = train_accs.mean(0)
            m_val   = val_accs.mean(0)
            s_val   = val_accs.std(0)

            ax.plot(eps, m_train, color=col, lw=1.8, ls=ls, alpha=0.35)
            ax.plot(eps, m_val,   color=col, lw=2.2, ls=ls,
                    label=f"{k}-shot (val)")
            ax.fill_between(eps, m_val - s_val, m_val + s_val,
                            color=fill, alpha=0.15)

        ax.set_title(label, color="black")
        ax.set_xlabel("Episode")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        # k-shot line styles (neutral color)
        Line2D([0],[0], color="#374151", lw=2,   ls="-",  label="5-shot"),
        Line2D([0],[0], color="#374151", lw=2,   ls="--", label="10-shot"),
        Line2D([0],[0], color="#374151", lw=2,   ls="-.", label="15-shot"),
        # model colors
        Line2D([0],[0], color="#2563eb", lw=2.2, ls="-",  label="ResNet18 (val, bold)"),
        Line2D([0],[0], color="#dc2626", lw=2.2, ls="-",  label="CNN (val, bold)"),
        # train vs val distinction
        Line2D([0],[0], color="#6b7280", lw=1.8, ls="-",  alpha=0.35, label="Train accuracy (faint)"),
        Line2D([0],[0], color="#6b7280", lw=2.2, ls="-",               label="Val accuracy (bold)"),
        # shaded area
        Patch(facecolor="#bfdbfe", alpha=0.4, edgecolor="none",
              label="Val accuracy ± std across runs"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot3_overfitting.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 3 saved")


def plot_train_val_loss():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("ProtoNet — Train & Val Loss per Model & k-Shot",
                 fontsize=13, fontweight="bold", y=1.01)

    for row, model in enumerate(models):
        col  = MODEL_COLS[model]
        fill = MODEL_FILLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"

        for ci, k in enumerate(k_shots):
            ax   = axes[row, ci]
            runs = data[model][k]

            train_losses = np.array([r["logs"]["train_loss"] for r in runs])
            val_losses   = np.array([r["logs"]["val_loss"]   for r in runs])
            eps = range(1, train_losses.shape[1] + 1)

            m_train = train_losses.mean(0); s_train = train_losses.std(0)
            ax.plot(eps, m_train, color=col, lw=2.2, label="Train loss")
            ax.fill_between(eps, m_train - s_train, m_train + s_train,
                            color=col, alpha=0.15)

            m_val = val_losses.mean(0); s_val = val_losses.std(0)
            ax.plot(eps, m_val, color=col, lw=2.2, linestyle="--",
                    label="Val loss")
            ax.fill_between(eps, m_val - s_val, m_val + s_val,
                            color=col, alpha=0.08)

            ax.set_title(f"{label} — {k}-shot", color="black")
            ax.set_xlabel("Episode")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0],[0], color="#2563eb", lw=2,         label="Train loss (ResNet18)"),
        Line2D([0],[0], color="#2563eb", lw=2, ls="--",label="Val loss (ResNet18)"),
        Line2D([0],[0], color="#dc2626", lw=2,         label="Train loss (CNN)"),
        Line2D([0],[0], color="#dc2626", lw=2, ls="--",label="Val loss (CNN)"),
        Patch(facecolor="#9ca3af", alpha=0.35, edgecolor="none",
              label="Loss ± std across runs (shaded)"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.04), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot4_val_loss.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 4 saved")

def plot_gap_closing():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle("ProtoNet vs Classic Supervised — Accuracy Gap by k-Shot",
                 fontsize=13, fontweight="bold")

    for model in models:
        col   = MODEL_COLS[model]
        fill  = MODEL_FILLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"
        cl_mean = CLASSIC_REF[model]["mean"]
        cl_std  = CLASSIC_REF[model]["std"]

        gaps, gap_stds = [], []
        for k in k_shots:
            accs = [r["test_acc"] for r in data[model][k]]
            p_mean = np.mean(accs)
            p_std  = np.std(accs)
            gaps.append(p_mean - cl_mean)
            gap_stds.append(np.sqrt(p_std**2 + cl_std**2))

        gaps      = np.array(gaps)
        gap_stds  = np.array(gap_stds)

        ax.plot(k_shots, gaps, color=col, lw=2.5, marker="o", markersize=8, label=label)
        ax.fill_between(k_shots, gaps - gap_stds, gaps + gap_stds, color=fill, alpha=0.25)

        for k, g in zip(k_shots, gaps):
            ax.annotate(f"{g:+.3f}", (k, g), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=8.5,
                        color=col, fontweight="bold")

    ax.axhline(0, color="#374151", lw=1.3, linestyle="--", alpha=0.7,
               label="Parity with classic (gap = 0)")

    ax.set_xticks(k_shots)
    ax.set_xticklabels([f"{k}-shot" for k in k_shots], fontsize=11)
    ax.set_xlabel("k-Shot Setting")
    ax.set_ylabel("ProtoNet − Classic Test Accuracy")
    ax.grid(alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#d1d5db", alpha=0.5, edgecolor="none",
                         label="± propagated std (ProtoNet std + Classic std)"))
    fig.legend(handles=handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.07), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot5_gap_closing.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 5 (gap closing) saved")


if __name__ == "__main__":
    print("Generating few-shot plots...\n")
    plot_shot_efficiency()
    plot_learning_curves()
    plot_train_val_gap()
    plot_train_val_loss()
    plot_gap_closing()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")