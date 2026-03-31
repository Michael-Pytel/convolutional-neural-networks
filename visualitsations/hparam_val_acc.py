import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "../exp1_hparams")
OUTPUT_DIR = os.path.join(_HERE, "../plots/hparams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_COL  = "#2563eb"  
VAL_COL    = "#dc2626"   
SHADE_ALPHA = 0.10
RUN_ALPHA   = 0.30
MEAN_LW     = 2.2
RUN_LW      = 1.0
GRID_COL    = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#d1d5db",
    "axes.labelcolor":   "#111827",
    "xtick.color":       "#6b7280",
    "ytick.color":       "#6b7280",
    "text.color":        "#111827",
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         9.5,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#d1d5db",
    "legend.labelcolor": "#111827",
    "savefig.facecolor": "white",
})

def load_experiments(pattern):
    records = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        d = json.load(open(fp))
        cfg = d["config"]
        test_accs = [r["test"]["accuracy"] for r in d["runs"]]
        records.append({
            "fname":        fp,
            "exp":          os.path.basename(fp),
            "model":        cfg["model"],
            "batch_size":   cfg["batch_size"],
            "lr":           cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "dropout":      cfg["dropout"],
            "test_acc_mean": np.mean(test_accs),
            "test_acc_std":  np.std(test_accs),
            "runs":         d["runs"],
        })
    return records

cnn_recs = load_experiments("exp_*_cnn.json")
res_recs = load_experiments("exp_*_resnet18.json")


def plot_train_val(rec, tag, save_path):
    runs = rec["runs"]
    n_runs = len(runs)

    fig = plt.figure(figsize=(5 * (n_runs + 1), 8))
    fig.patch.set_facecolor("white")

    cfg_str = (f"lr={rec['lr']:.2e}   wd={rec['weight_decay']:.1e}   "
               f"dropout={rec['dropout']}   batch={rec['batch_size']}")
    acc_str = f"mean test acc = {rec['test_acc_mean']:.4f} ± {rec['test_acc_std']:.4f}"
    fig.suptitle(
        f"{tag}\n{cfg_str}\n{acc_str}",
        fontsize=11, fontweight="bold", y=0.99, linespacing=1.6
    )

    ncols = n_runs + 1 
    gs = gridspec.GridSpec(2, ncols, figure=fig,
                           hspace=0.42, wspace=0.35,
                           top=0.88, bottom=0.07, left=0.06, right=0.98)


    min_len_loss = min(len(r["logs"]["train_loss"]) for r in runs)
    min_len_acc  = min(len(r["logs"]["val_acc"])    for r in runs)

    all_train_loss = np.array([r["logs"]["train_loss"][:min_len_loss] for r in runs])
    all_val_loss   = np.array([r["logs"]["val_loss"]  [:min_len_loss] for r in runs])
    all_val_acc    = np.array([r["logs"]["val_acc"]   [:min_len_acc]  for r in runs])

    mean_train_loss = all_train_loss.mean(0); std_train_loss = all_train_loss.std(0)
    mean_val_loss   = all_val_loss.mean(0);   std_val_loss   = all_val_loss.std(0)
    mean_val_acc    = all_val_acc.mean(0);    std_val_acc    = all_val_acc.std(0)

    def style_ax(ax, ylabel, title=None):
        ax.grid(True, alpha=0.55)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if title:
            ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    for col, run in enumerate(runs):
        tl = run["logs"]["train_loss"]
        vl = run["logs"]["val_loss"]
        va = run["logs"]["val_acc"]
        ep_loss = range(1, len(tl) + 1)
        ep_acc  = range(1, len(va) + 1)

        ax_loss = fig.add_subplot(gs[0, col])
        ax_loss.plot(ep_loss, tl, color=TRAIN_COL, lw=MEAN_LW, label="Train loss")
        ax_loss.plot(ep_loss, vl, color=VAL_COL,   lw=MEAN_LW, label="Val loss",
                     linestyle="--")
        ax_loss.legend(fontsize=8, loc="upper right")
        style_ax(ax_loss, "Loss", f"Run {col + 1}")

        ax_acc = fig.add_subplot(gs[1, col])
        ax_acc.plot(ep_acc, va, color=VAL_COL, lw=MEAN_LW, label="Val acc")
        ax_acc.set_ylim(max(0, min(va) - 0.02), min(1.0, max(va) + 0.02))
        ax_acc.legend(fontsize=8, loc="lower right")
        style_ax(ax_acc, "Validation Accuracy")

    ep_loss_agg = range(1, min_len_loss + 1)
    ep_acc_agg  = range(1, min_len_acc  + 1)

    ax_loss_agg = fig.add_subplot(gs[0, n_runs])
    ax_loss_agg.plot(ep_loss_agg, mean_train_loss, color=TRAIN_COL, lw=MEAN_LW, label="Train loss (mean)")
    ax_loss_agg.fill_between(ep_loss_agg,
                             mean_train_loss - std_train_loss,
                             mean_train_loss + std_train_loss,
                             color=TRAIN_COL, alpha=SHADE_ALPHA)
    ax_loss_agg.plot(ep_loss_agg, mean_val_loss, color=VAL_COL, lw=MEAN_LW,
                     linestyle="--", label="Val loss (mean)")
    ax_loss_agg.fill_between(ep_loss_agg,
                             mean_val_loss - std_val_loss,
                             mean_val_loss + std_val_loss,
                             color=VAL_COL, alpha=SHADE_ALPHA)
    for r in runs:
        ax_loss_agg.plot(range(1, min_len_loss + 1), r["logs"]["train_loss"][:min_len_loss],
                         color=TRAIN_COL, lw=RUN_LW, alpha=RUN_ALPHA)
        ax_loss_agg.plot(range(1, min_len_loss + 1), r["logs"]["val_loss"][:min_len_loss],
                         color=VAL_COL,   lw=RUN_LW, alpha=RUN_ALPHA, linestyle="--")
    ax_loss_agg.legend(fontsize=8, loc="upper right")
    style_ax(ax_loss_agg, "Loss", "Mean ± Std (all runs)")

    ax_acc_agg = fig.add_subplot(gs[1, n_runs])
    ax_acc_agg.plot(ep_acc_agg, mean_val_acc, color=VAL_COL, lw=MEAN_LW, label="Val acc (mean)")
    ax_acc_agg.fill_between(ep_acc_agg,
                            mean_val_acc - std_val_acc,
                            mean_val_acc + std_val_acc,
                            color=VAL_COL, alpha=SHADE_ALPHA)
    for r in runs:
        ax_acc_agg.plot(range(1, min_len_acc + 1), r["logs"]["val_acc"][:min_len_acc],
                        color=VAL_COL, lw=RUN_LW, alpha=RUN_ALPHA)
    lo = max(0,   mean_val_acc.min() - 3 * std_val_acc.max() - 0.01)
    hi = min(1.0, mean_val_acc.max() + 3 * std_val_acc.max() + 0.01)
    ax_acc_agg.set_ylim(lo, hi)
    ax_acc_agg.legend(fontsize=8, loc="lower right")
    style_ax(ax_acc_agg, "Validation Accuracy")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved → {os.path.basename(save_path)}")


print("Generating train/val curves...\n")

for model_name, recs in [("CNN", cnn_recs), ("ResNet18", res_recs)]:
    best  = max(recs, key=lambda r: r["test_acc_mean"])
    worst = min(recs, key=lambda r: r["test_acc_mean"])

    print(f"[{model_name}]")
    print(f"  BEST  → {best['exp']}   acc={best['test_acc_mean']:.4f}")
    print(f"  WORST → {worst['exp']}  acc={worst['test_acc_mean']:.4f}")

    plot_train_val(
        best,
        tag=f"{model_name} — BEST  ({best['exp']})",
        save_path=f"{OUTPUT_DIR}/curves_{model_name.lower()}_best.png"
    )
    plot_train_val(
        worst,
        tag=f"{model_name} — WORST ({worst['exp']})",
        save_path=f"{OUTPUT_DIR}/curves_{model_name.lower()}_worst.png"
    )

print("\nDone.")