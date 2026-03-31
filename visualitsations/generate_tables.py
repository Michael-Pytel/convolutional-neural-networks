import json, glob, os
import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "../tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def _w(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  saved -> {os.path.relpath(path, _HERE)}")

def pm(mean, std, digits=4):
    return rf"${mean:.{digits}f} \pm {std:.{digits}f}$"

def midrule(): return "\\midrule"
def toprule(): return "\\toprule"
def bottomrule(): return "\\bottomrule"

def booktabs_table(caption, label, col_spec, header, rows, notes=""):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        toprule(),
        header,
        midrule(),
        *rows,
        bottomrule(),
        "\\end{tabular}",
    ]
    if notes:
        lines.append(f"\\\\[4pt]\\footnotesize {notes}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# EXP 1 — Hyperparameter Search
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp1():
    print("\n[Exp1] Hyperparameter search tables...")

    def load_recs(pattern):
        recs = []
        for fp in sorted(glob.glob(os.path.join(_HERE, "../exp1_hparams", pattern))):
            d   = json.load(open(fp))
            cfg = d["config"]
            test_accs = [r["test"]["accuracy"] for r in d["runs"]]
            val_accs  = [r["val"]["accuracy"]  for r in d["runs"]]
            recs.append({
                "exp":    os.path.basename(fp).split("_")[1],
                "lr":     cfg["lr"],
                "wd":     cfg["weight_decay"],
                "do":     cfg["dropout"],
                "bs":     cfg["batch_size"],
                "test_m": np.mean(test_accs),
                "test_s": np.std(test_accs),
                "val_m":  np.mean(val_accs),
                "val_s":  np.std(val_accs),
            })
        recs.sort(key=lambda r: r["test_m"], reverse=True)
        return recs

    col_spec = "clllllll"
    header   = (r"\textbf{Rank} & \textbf{LR} & \textbf{Weight Decay} & "
                r"\textbf{Dropout} & \textbf{Batch Size} & "
                r"\textbf{Test Acc} ($\mu\pm\sigma$) & "
                r"\textbf{Val Acc} ($\mu\pm\sigma$) \\")

    for model_tag, pattern, label_tag in [
        ("CNN",      "exp_*_cnn.json",      "cnn"),
        ("ResNet18", "exp_*_resnet18.json", "resnet18"),
    ]:
        recs = load_recs(pattern)
        rows = []
        for rank, r in enumerate(recs, 1):
            is_best = rank == 1
            def b(s, best=is_best): return f"\\textbf{{{s}}}" if best else s
            cells = [
                b(str(rank)),
                b(f"${r['lr']:.2e}$"),
                b(f"${r['wd']:.2e}$"),
                b(f"{r['do']:.1f}"),
                b(str(r['bs'])),
                b(pm(r['test_m'], r['test_s'])),
                b(pm(r['val_m'],  r['val_s'])),
            ]
            rows.append(" & ".join(cells) + " \\\\")
            if rank == 1:
                rows.append(midrule())

        tex = booktabs_table(
            caption=(f"Exp.\\ 1 --- {model_tag} hyperparameter search: "
                     f"all {len(recs)} configurations sorted by mean test accuracy (descending). "
                     f"Each configuration is evaluated over 3 independent runs."),
            label=f"tab:exp1_{label_tag}",
            col_spec=col_spec,
            header=header,
            rows=rows,
            notes=(r"Best configuration in \textbf{bold}. "
                   r"$\sigma$ = standard deviation across 3 runs."),
        )
        _w(os.path.join(OUT_DIR, f"exp1_{label_tag}.tex"), tex)


# ═══════════════════════════════════════════════════════════════════════════
# EXP 2 — Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp2():
    print("\n[Exp2] Augmentation tables...")

    AUG_ORDER = [
        (False, "none"),   (False, "mixup"),
        (False, "cutmix"), (False, "both"),
        (True,  "none"),   (True,  "mixup"),
        (True,  "cutmix"), (True,  "both"),
    ]
    COND_LABELS = {
        (False, "none"):   "Baseline",
        (False, "mixup"):  "Mixup",
        (False, "cutmix"): "CutMix",
        (False, "both"):   "Mixup + CutMix",
        (True,  "none"):   "Std Aug",
        (True,  "mixup"):  "Std Aug + Mixup",
        (True,  "cutmix"): "Std Aug + CutMix",
        (True,  "both"):   "Std Aug + Mixup + CutMix",
    }

    def load_recs(pattern):
        recs = {}
        for fp in sorted(glob.glob(os.path.join(_HERE, "../exp2_aug2", pattern))):
            d   = json.load(open(fp))
            cfg = d["config"]
            if "augmentation" not in cfg:
                continue
            aug  = cfg["augmentation"]
            mix  = (cfg.get("mix_type") or "none").lower()
            test_accs = [r["test"]["accuracy"] for r in d["runs"]]
            val_accs  = [r["val"]["accuracy"]  for r in d["runs"]]
            recs[(aug, mix)] = {
                "test_m": np.mean(test_accs), "test_s": np.std(test_accs),
                "val_m":  np.mean(val_accs),  "val_s":  np.std(val_accs),
            }
        return recs

    col_spec = "llcccc"
    header   = (r"\textbf{Model} & \textbf{Condition} & "
                r"\textbf{Std Aug} & \textbf{Mix Strategy} & "
                r"\textbf{Test Acc} ($\mu\pm\sigma$) & "
                r"\textbf{Val Acc} ($\mu\pm\sigma$) \\")

    rows = []
    for model_tag, pattern in [("CNN", "exp_*_cnn.json"), ("ResNet18", "exp_*_resnet18.json")]:
        recs     = load_recs(pattern)
        best_acc = max(recs[c]["test_m"] for c in AUG_ORDER)
        for i, cond in enumerate(AUG_ORDER):
            aug, mix = cond
            r        = recs[cond]
            is_best  = abs(r["test_m"] - best_acc) < 1e-9
            aug_str  = r"\checkmark" if aug else "---"
            mix_str  = ("Mixup + CutMix" if mix == "both"
                        else mix.capitalize() if mix != "none" else "---")
            acc_cell = (f"\\textbf{{{r['test_m']:.4f}}}$\\pm {r['test_s']:.4f}$"
                        if is_best else pm(r["test_m"], r["test_s"]))
            model_cell = f"\\textbf{{{model_tag}}}" if i == 0 else ""
            rows.append(
                f"{model_cell} & "
                f"{COND_LABELS[cond]} & {aug_str} & {mix_str} & "
                f"{acc_cell} & "
                f"{pm(r['val_m'], r['val_s'])} \\\\"
            )
        rows.append(midrule())

    tex = booktabs_table(
        caption=(r"Exp.\ 2 --- Augmentation strategy comparison for CNN and ResNet18. "
                 r"Each condition is evaluated over 3 independent runs. "
                 r"Best test accuracy per model is highlighted in \textbf{bold}."),
        label="tab:exp2_aug",
        col_spec=col_spec,
        header=header,
        rows=rows,
        notes=(r"\checkmark\ = standard augmentation (random crop + horizontal flip). "
               r"Mix strategies are applied on top of (or independently from) standard augmentation."),
    )
    _w(os.path.join(OUT_DIR, "exp2_augmentation.tex"), tex)


# ═══════════════════════════════════════════════════════════════════════════
# EXP 3 — Few-Shot
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp3():
    print("\n[Exp3] Few-shot tables...")

    proto = json.load(open(os.path.join(_HERE, "../exp3_fewshot/protonet.json")))

    classic = {}
    for model, fname in [("cnn", "exp_0_cnn.json"), ("resnet18", "exp_1_resnet18.json")]:
        runs = json.load(open(os.path.join(_HERE, "../exp3_fewshot/classic", fname)))["runs"]
        accs = [r["test"]["accuracy"] for r in runs]
        classic[model] = {"test_m": np.mean(accs), "test_s": np.std(accs)}

    proto_data = {}
    for entry in proto:
        accs = [r["test_acc"] for r in entry["runs"]]
        proto_data[(entry["model"], entry["k_shot"])] = {
            "test_m": np.mean(accs), "test_s": np.std(accs),
        }

    col_spec = "llccc"
    header   = (r"\textbf{Model} & \textbf{Method} & \textbf{k-Shot} & "
                r"\textbf{Test Acc} ($\mu\pm\sigma$) & "
                r"\textbf{Gap vs Classic} \\")

    rows = []
    for model_tag, model_key in [("CNN", "cnn"), ("ResNet18", "resnet18")]:
        cl = classic[model_key]
        rows.append(
            f"\\textbf{{{model_tag}}} & Classic (Supervised) & --- & "
            f"{pm(cl['test_m'], cl['test_s'])} & --- \\\\"
        )
        rows.append(midrule())
        for k in [5, 10, 15]:
            p   = proto_data[(model_key, k)]
            gap = p["test_m"] - cl["test_m"]
            rows.append(
                f" & ProtoNet & {k} & "
                f"{pm(p['test_m'], p['test_s'])} & "
                f"${gap:+.4f}$ \\\\"
            )
        rows.append(midrule())

    tex = booktabs_table(
        caption=(r"Exp.\ 3 --- ProtoNet few-shot test accuracy vs classic supervised baseline "
                 r"for CNN and ResNet18 on CIFAR-10. "
                 r"ProtoNet results are averaged over 3 independent runs per k-shot setting. "
                 r"Gap = ProtoNet accuracy $-$ classic accuracy."),
        label="tab:exp3_fewshot",
        col_spec=col_spec,
        header=header,
        rows=rows,
        notes=(r"Negative gap indicates few-shot underperforms the classic baseline. "
               r"Classic models are trained with standard supervised training using only "
               r"\textbf{20 samples per class} (200 labelled images total), "
               r"matching the same data budget as the few-shot setting."),
    )
    _w(os.path.join(OUT_DIR, "exp3_fewshot.tex"), tex)


# ═══════════════════════════════════════════════════════════════════════════
# EXP 4 — Ensemble
# ═══════════════════════════════════════════════════════════════════════════

def gen_exp4():
    print("\n[Exp4] Ensemble tables...")

    d = json.load(open(os.path.join(_HERE, "../exp4_ensemble/final_ensemble.json")))

    VOTING_METHODS   = ["hard", "soft", "weighted"]
    STACKING_METHODS = ["logreg", "ridge", "rf", "gb", "svm", "knn"]
    METHOD_LABELS = {
        "hard":    "Hard Voting",
        "soft":    "Soft Voting",
        "weighted":"Weighted Voting",
        "logreg":  "Logistic Regression",
        "ridge":   "Ridge",
        "rf":      "Random Forest",
        "gb":      "Gradient Boosting",
        "svm":     "SVM",
        "knn":     "k-NN",
    }

    all_acc = {}
    for m in VOTING_METHODS:
        all_acc[m] = d[m]["accuracy"]
    for m in STACKING_METHODS:
        all_acc[m] = d["stacking"][m]["accuracy"]

    best_acc = max(all_acc.values())

    col_spec = "lc"
    header   = r"\textbf{Method} & \textbf{Test Accuracy} \\"

    # ── voting ───────────────────────────────────────────────────────────
    rows = []
    for m in VOTING_METHODS:
        acc     = all_acc[m]
        is_best = abs(acc - best_acc) < 1e-9
        acc_str = f"\\textbf{{{acc:.4f}}}" if is_best else f"{acc:.4f}"
        rows.append(f"{METHOD_LABELS[m]} & {acc_str} \\\\")

    tex = booktabs_table(
        caption=(r"Exp.\ 4 --- Ensemble voting methods. "
                 r"All methods combine the same five ResNet18 model checkpoints. "
                 r"Best accuracy across all ensemble methods is highlighted in \textbf{bold}."),
        label="tab:exp4_voting",
        col_spec=col_spec,
        header=header,
        rows=rows,
        notes=(r"Weighted voting uses per-model validation accuracy as weights. "
               r"Soft and weighted voting coincide when all models have equal validation performance."),
    )
    _w(os.path.join(OUT_DIR, "exp4_voting.tex"), tex)

    # ── stacking ──────────────────────────────────────────────────────────
    rows = []
    for m in STACKING_METHODS:
        acc     = all_acc[m]
        is_best = abs(acc - best_acc) < 1e-9
        acc_str = f"\\textbf{{{acc:.4f}}}" if is_best else f"{acc:.4f}"
        rows.append(f"{METHOD_LABELS[m]} & {acc_str} \\\\")

    tex = booktabs_table(
        caption=(r"Exp.\ 4 --- Ensemble stacking classifiers trained on the concatenated "
                 r"softmax outputs of the five base ResNet18 models. "
                 r"Best accuracy across all ensemble methods is highlighted in \textbf{bold}."),
        label="tab:exp4_stacking",
        col_spec=col_spec,
        header=header,
        rows=rows,
    )
    _w(os.path.join(OUT_DIR, "exp4_stacking.tex"), tex)

    # ── combined voting + stacking ────────────────────────────────────────
    rows = []
    rows.append(rf"\multicolumn{{2}}{{l}}{{\textit{{Voting methods}}}} \\")
    rows.append(midrule())
    for m in VOTING_METHODS:
        acc     = all_acc[m]
        is_best = abs(acc - best_acc) < 1e-9
        acc_str = f"\\textbf{{{acc:.4f}}}" if is_best else f"{acc:.4f}"
        rows.append(f"{METHOD_LABELS[m]} & {acc_str} \\\\")
    rows.append(midrule())
    rows.append(rf"\multicolumn{{2}}{{l}}{{\textit{{Stacking methods}}}} \\")
    rows.append(midrule())
    for m in STACKING_METHODS:
        acc     = all_acc[m]
        is_best = abs(acc - best_acc) < 1e-9
        acc_str = f"\\textbf{{{acc:.4f}}}" if is_best else f"{acc:.4f}"
        rows.append(f"{METHOD_LABELS[m]} & {acc_str} \\\\")

    tex = booktabs_table(
        caption=(r"Exp.\ 4 --- All ensemble methods (voting and stacking) and their "
                 r"test accuracy on CIFAR-10. "
                 r"Best result across all methods is highlighted in \textbf{bold}."),
        label="tab:exp4_all",
        col_spec=col_spec,
        header=header,
        rows=rows,
        notes=(r"Base models: five ResNet18 checkpoints selected from Exp.\ 1. "
               r"Stacking meta-classifiers are trained on held-out softmax probability vectors."),
    )
    _w(os.path.join(OUT_DIR, "exp4_all.tex"), tex)


# ═══════════════════════════════════════════════════════════════════════════
# Master include file
# ═══════════════════════════════════════════════════════════════════════════

def gen_master():
    files = [
        ("exp1_cnn.tex",         "Exp. 1 - CNN HP Search"),
        ("exp1_resnet18.tex",    "Exp. 1 - ResNet18 HP Search"),
        ("exp2_augmentation.tex","Exp. 2 - Augmentation Comparison"),
        ("exp3_fewshot.tex",     "Exp. 3 - Few-Shot vs Classic"),
        ("exp4_voting.tex",      "Exp. 4 - Voting Methods"),
        ("exp4_stacking.tex",    "Exp. 4 - Stacking Methods"),
        ("exp4_all.tex",         "Exp. 4 - All Ensemble Methods Combined"),
    ]
    lines = [
        "% Auto-generated appendix table includes",
        "% \\input this file inside your appendix chapter",
        "",
    ]
    for fname, desc in files:
        lines.append(f"% {desc}")
        lines.append(f"\\input{{tables/{fname}}}")
        lines.append("")
    _w(os.path.join(OUT_DIR, "all_tables.tex"), "\n".join(lines))


# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating LaTeX tables...\n")
    gen_exp1()
    gen_exp2()
    gen_exp3()
    gen_exp4()
    gen_master()
    print(f"\nDone - all tables written to {os.path.relpath(OUT_DIR, _HERE)}/")
