"""
Few-Shot Learning (ProtoNet) — Visualisation Suite
===================================================
Model: Prototypical Network
Backbones: ResNet18, CNN
k-shot settings: 5, 10, 15  |  3 runs each

Plots generated
  1. fewshot_plot1_efficiency.png  — grouped bar chart: test acc by k-shot,
                                     with fully-supervised & random-chance refs
  2. fewshot_plot2_curves.png      — train loss + val acc (dual axis) per cell
  3. fewshot_plot3_overfitting.png — train vs val accuracy gap per model
  4. fewshot_plot4_val_loss.png    — train loss + val loss on same axis per cell
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(_HERE, '..', 'exp3_fewshot',"protonet.json")
OUTPUT_DIR = os.path.join(_HERE, '..', 'plots',"fewshot")
os.makedirs(OUTPUT_DIR, exist_ok=True)


SUPERVISED_REF = {
    "resnet18": 0.9001,   
    "cnn":      0.8239,  
}
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
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("ProtoNet — Test Accuracy by k-Shot Setting\n"
                 "(with fully-supervised reference)",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(k_shots))
    w = 0.35
    err_kw = {"ecolor": "#374151", "capsize": 5,
              "elinewidth": 1.4, "capthick": 1.4}

    for i, model in enumerate(models):
        col   = MODEL_COLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"
        means = [np.mean([r["test_acc"] for r in data[model][k]]) for k in k_shots]
        stds  = [np.std( [r["test_acc"] for r in data[model][k]]) for k in k_shots]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, means, w, color=col, yerr=stds,
                      error_kw=err_kw, edgecolor="white", linewidth=0.8,
                      zorder=3, label=f"{label} (few-shot)", alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.003,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=8, color=col, fontweight="bold")

    for model, ref in SUPERVISED_REF.items():
        col   = MODEL_COLS[model]
        label = "CNN" if model == "cnn" else "ResNet18"
        ax.axhline(ref, color=col, lw=1.4, linestyle="--", alpha=0.55,
                   label=f"{label} supervised ({ref:.4f})")

    ax.axhline(RANDOM_CHANCE, color="#9ca3af", lw=1.2, linestyle=":",
               label="Random chance (0.10)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-shot" for k in k_shots], fontsize=11)
    ax.set_xlabel("k-Shot Setting")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)

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

            test_mean = np.mean([r["test_acc"] for r in runs])
            ax.text(0.97, 0.08, f"test acc = {test_mean:.4f}",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color="#374151",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#d1d5db", alpha=0.9))

            ax.set_title(f"{label} — {k}-shot", color="black")
            ax.set_xlabel("Episode")
            ax.grid(alpha=0.35)
            ax.spines[["top"]].set_visible(False)
            if ci == 0:
                ax.legend(fontsize=8, loc="upper right")

    legend_handles = [
        Line2D([0],[0], color="#2563eb", lw=2,         label="Train loss (ResNet18)"),
        Line2D([0],[0], color="#dc2626", lw=2,         label="Train loss (CNN)"),
        Line2D([0],[0], color="#f59e0b", lw=2, ls="--",label="Val accuracy"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.03), framealpha=0.95)

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
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)

    fig.text(0.5, -0.03,
             "Faint lines = train accuracy (support set)   |   "
             "Bold lines = val accuracy   |   "
             "Gap = support-set memorisation",
             ha="center", fontsize=8.5, color="#6b7280", style="italic")

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

            test_mean = np.mean([r["test_acc"] for r in runs])
            ax.text(0.97, 0.93, f"test acc = {test_mean:.4f}",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color="#374151",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#d1d5db", alpha=0.9))

            ax.set_title(f"{label} — {k}-shot", color="black")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)
            if ci == 0:
                ax.legend(fontsize=8.5, loc="upper right")

    legend_handles = [
        Line2D([0],[0], color="#2563eb", lw=2,         label="Train loss (ResNet18)"),
        Line2D([0],[0], color="#2563eb", lw=2, ls="--",label="Val loss (ResNet18)"),
        Line2D([0],[0], color="#dc2626", lw=2,         label="Train loss (CNN)"),
        Line2D([0],[0], color="#dc2626", lw=2, ls="--",label="Val loss (CNN)"),
    ]
    fig.legend(handles=legend_handles, fontsize=8.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.03), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fewshot_plot4_val_loss.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 4 saved")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating few-shot plots...\n")
    plot_shot_efficiency()
    plot_learning_curves()
    plot_train_val_gap()
    plot_train_val_loss()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")