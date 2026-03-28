

import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_HERE, "../exp2_aug")
OUTPUT_DIR = os.path.join(_HERE, "../plots/augumentations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


MIX_COLS = {
    "none":   ("#93c5fd", "#1d4ed8"), 
    "mixup":  ("#86efac", "#15803d"),  
    "cutmix": ("#fca5a5", "#b91c1c"),   
    "both":   ("#d8b4fe", "#7e22ce"),   
}
AUG_HATCHES = {False: "", True: "//"}

COND_ORDER = [
    (False, "none"),   (False, "mixup"),
    (False, "cutmix"), (False, "both"),
    (True,  "none"),   (True,  "mixup"),
    (True,  "cutmix"), (True,  "both"),
]
COND_LABELS = {
    (False, "none"):   "Baseline",
    (False, "mixup"):  "Mixup",
    (False, "cutmix"): "CutMix",
    (False, "both"):   "Mix+CutMix",
    (True,  "none"):   "Aug",
    (True,  "mixup"):  "Aug+Mixup",
    (True,  "cutmix"): "Aug+CutMix",
    (True,  "both"):   "Aug+\nMix+CutMix",
}

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

def load_aug_experiments(model_tag, file_pattern):
    records = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, file_pattern))):
        d = json.load(open(fp))
        cfg = d["config"]
        if "augmentation" not in cfg:
            continue
        aug      = cfg["augmentation"]
        mix_raw  = cfg.get("mix_type") or "none"
        mix      = mix_raw.lower() if mix_raw else "none"

        test_accs   = [r["test"]["accuracy"]  for r in d["runs"]]
        val_accs    = [r["val"]["accuracy"]   for r in d["runs"]]
        macro_f1s   = [r["test"]["macro_f1"]  for r in d["runs"]]
        per_cls_f1  = np.mean([r["test"]["f1"] for r in d["runs"]], axis=0)

        min_len_loss = min(len(r["logs"]["train_loss"]) for r in d["runs"])
        min_len_acc  = min(len(r["logs"]["val_acc"])    for r in d["runs"])

        train_curves = np.array([r["logs"]["train_loss"][:min_len_loss] for r in d["runs"]])
        val_curves   = np.array([r["logs"]["val_loss"]  [:min_len_loss] for r in d["runs"]])
        vacc_curves  = np.array([r["logs"]["val_acc"]   [:min_len_acc]  for r in d["runs"]])

        records.append({
            "model":          model_tag,
            "aug":            aug,
            "mix":            mix,
            "cond":           (aug, mix),
            "label":          COND_LABELS[(aug, mix)],
            "test_acc_mean":  np.mean(test_accs),
            "test_acc_std":   np.std(test_accs),
            "val_acc_mean":   np.mean(val_accs),
            "per_cls_f1":     per_cls_f1,
            "train_curves":   train_curves,
            "val_curves":     val_curves,
            "vacc_curves":    vacc_curves,
            "all_test_accs":  test_accs,
        })
    order = {c: i for i, c in enumerate(COND_ORDER)}
    records.sort(key=lambda r: order.get(r["cond"], 99))
    return records

cnn_recs = load_aug_experiments("CNN",     "exp_*_cnn.json")
res_recs = load_aug_experiments("ResNet18","exp_*_resnet18.json")

def recs_by_cond(recs):
    return {r["cond"]: r for r in recs}


_COL_NO_AUG = "#4878d0"   
_COL_AUG    = "#e15759"   

def plot_bar_comparison():
    mix_keys   = ["none", "mixup", "cutmix", "both"]
    mix_labels = ["None", "Mixup", "CutMix", "Mix+CutMix"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Test Accuracy by Augmentation Strategy",
                 fontsize=13, fontweight="bold", y=1.01)

    bar_w = 0.35
    for ax, recs, title in zip(axes, [cnn_recs, res_recs], ["CNN", "ResNet18"]):
        by_cond = recs_by_cond(recs)
        xs = np.arange(len(mix_keys))

        no_aug_means = [by_cond[(False, m)]["test_acc_mean"] for m in mix_keys]
        no_aug_stds  = [by_cond[(False, m)]["test_acc_std"]  for m in mix_keys]
        aug_means    = [by_cond[(True,  m)]["test_acc_mean"] for m in mix_keys]
        aug_stds     = [by_cond[(True,  m)]["test_acc_std"]  for m in mix_keys]

        err_kw = {"ecolor": "#374151", "capsize": 4, "elinewidth": 1.3, "capthick": 1.3}

        bars_no = ax.bar(xs - bar_w/2, no_aug_means, bar_w, yerr=no_aug_stds,
                         color=_COL_NO_AUG, edgecolor="white", linewidth=0.8,
                         error_kw=err_kw, zorder=3, label="No augmentation")
        bars_au = ax.bar(xs + bar_w/2, aug_means,    bar_w, yerr=aug_stds,
                         color=_COL_AUG,    edgecolor="white", linewidth=0.8,
                         error_kw=err_kw, zorder=3, label="With augmentation")

        for x, m, s in zip(xs - bar_w/2, no_aug_means, no_aug_stds):
            ax.text(x, m + s + 0.001, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=7, color=_COL_NO_AUG, fontweight="bold")
        for x, m, s in zip(xs + bar_w/2, aug_means, aug_stds):
            ax.text(x, m + s + 0.001, f"{m:.4f}", ha="center", va="bottom",
                    fontsize=7, color=_COL_AUG, fontweight="bold")

        all_means = no_aug_means + aug_means
        all_stds  = no_aug_stds  + aug_stds
        ax.set_ylim(min(all_means) - 0.015,
                    max(m + s for m, s in zip(all_means, all_stds)) + 0.018)
        ax.set_xticks(xs)
        ax.set_xticklabels(mix_labels, fontsize=10)
        ax.set_xlabel("Mix Strategy")
        ax.set_ylabel("Mean Test Accuracy")
        ax.set_title(title, color="black")
        ax.grid(axis="y", alpha=0.45, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aug_plot1_bar.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 1 saved")
    plt.close(); print("✓ Plot 1 saved")



def plot_learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Validation Accuracy Curves by Augmentation Strategy",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, recs, title in zip(axes, [cnn_recs, res_recs], ["CNN", "ResNet18"]):
        for r in recs:
            mean_c = r["vacc_curves"].mean(axis=0)
            std_c  = r["vacc_curves"].std(axis=0)
            col    = MIX_COLS[r["mix"]][1 if r["aug"] else 0]
            ls     = "--" if r["aug"] else "-"
            ep     = range(1, len(mean_c) + 1)
            ax.plot(ep, mean_c, color=col, lw=2, linestyle=ls, label=r["label"])
            ax.fill_between(ep, mean_c - std_c, mean_c + std_c,
                            color=col, alpha=0.10)

        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
        ax.set_title(title, color="black"); ax.grid(alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7.5, loc="lower right", ncol=2)

    fig.text(0.5, -0.03, "Solid lines = no augmentation   |   Dashed lines = with augmentation",
             ha="center", fontsize=9, color="#6b7280", style="italic")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aug_plot2_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 2 saved")


def plot_loss_gap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Val − Train Loss Gap (lower = less overfitting)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, recs, title in zip(axes, [cnn_recs, res_recs], ["CNN", "ResNet18"]):
        for r in recs:
            col = _COL_AUG if r["aug"] else _COL_NO_AUG
            ls  = {
                "none":   "-",
                "mixup":  "--",
                "cutmix": "-.",
                "both":   ":",
            }[r["mix"]]
            ep       = range(1, r["train_curves"].shape[1] + 1)
            mean_gap = (r["val_curves"] - r["train_curves"]).mean(axis=0)
            ax.plot(ep, mean_gap, color=col, lw=2, linestyle=ls,
                    label=r["label"].replace("\n", " "))

        ax.axhline(0, color="#9ca3af", lw=1, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val loss − Train loss")
        ax.set_title(title, color="black")
        ax.grid(alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.95)

    handles = [
        mpatches.Patch(facecolor=_COL_NO_AUG, edgecolor="#9ca3af", label="No augmentation"),
        mpatches.Patch(facecolor=_COL_AUG,    edgecolor="#9ca3af", label="With augmentation"),
    ]
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0],[0], color="#6b7280", lw=1.8, ls="-",  label="No mix"),
        Line2D([0],[0], color="#6b7280", lw=1.8, ls="--", label="Mixup"),
        Line2D([0],[0], color="#6b7280", lw=1.8, ls="-.", label="CutMix"),
        Line2D([0],[0], color="#6b7280", lw=1.8, ls=":",  label="Mix+CutMix"),
    ]
    fig.legend(handles=handles, fontsize=8, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.08), framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aug_plot3_loss_gap.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 3 saved")


def plot_perclass_f1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Per-Class Test F1 Across Augmentation Conditions",
                 fontsize=13, fontweight="bold", y=1.01)

    n_cls   = 10
    cls_lbs = [f"C{i}" for i in range(n_cls)]

    for ax, recs, title, cmap in zip(axes, [cnn_recs, res_recs],
                                      ["CNN", "ResNet18"], ["Blues", "Oranges"]):
        matrix = np.array([r["per_cls_f1"] for r in recs])
        ylabels = [r["label"].replace("\n", " ") for r in recs]

        vmin = matrix.min() - 0.005
        vmax = matrix.max() + 0.005
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(cls_lbs, fontsize=9)
        ax.set_yticks(range(len(recs))); ax.set_yticklabels(ylabels, fontsize=8.5)
        ax.set_xlabel("Class"); ax.set_title(title, color="black")
        plt.colorbar(im, ax=ax, label="F1 Score", fraction=0.03)

        for i in range(len(recs)):
            for j in range(n_cls):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v < (vmin + vmax) / 2 else "#111827")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aug_plot4_perclass_f1.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✓ Plot 4 saved")

if __name__ == "__main__":
    print(f"CNN conditions found:     {[r['label'] for r in cnn_recs]}")
    print(f"ResNet18 conditions found: {[r['label'] for r in res_recs]}\n")
    print("Generating plots...\n")
    plot_bar_comparison()
    plot_learning_curves()
    plot_loss_gap()
    plot_perclass_f1()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")