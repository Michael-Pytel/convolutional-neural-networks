import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def load_results(results_dir):
    rows = []

    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        path = os.path.join(results_dir, file)

        with open(path, "r") as f:
            data = json.load(f)

        config = data["config"]

        for run in data["runs"]:
            row = {
                **config,


                "val_acc": run["val"]["accuracy"],
                "test_acc": run["test"]["accuracy"],

                "val_macro_f1": run["val"]["macro_f1"],
                "test_macro_f1": run["test"]["macro_f1"],

                "val_f1_mean": np.mean(run["val"]["f1"]),
                "test_f1_mean": np.mean(run["test"]["f1"]),
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    return df




def summarize(df, groupby_cols):
    summary = df.groupby(groupby_cols).agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),

        mean_f1=("test_macro_f1", "mean"),
        std_f1=("test_macro_f1", "std"),

        mean_val_acc=("val_acc", "mean"),
        mean_val_f1=("val_macro_f1", "mean"),

        mean_test_macro_f1=("test_macro_f1", "mean"),
        std_test_macro_f1=("test_macro_f1", "std")
    ).reset_index()

    return summary.sort_values("mean_acc", ascending=False)




def plot_metric(df, x, y="test_acc", hue=None):
    plt.figure()

    if hue:
        for key, grp in df.groupby(hue):
            grp_sorted = grp.sort_values(x)
            plt.plot(grp_sorted[x], grp_sorted[y], marker="o", label=str(key))
        plt.legend()
    else:
        df_sorted = df.sort_values(x)
        plt.plot(df_sorted[x], df_sorted[y], marker="o")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} vs {x}")
    plt.grid()
    plt.show()




def boxplot_metric(df, x, y="test_acc"):
    import seaborn as sns

    plt.figure(figsize=(8, 5))

    sns.boxplot(data=df, x=x, y=y, hue="model")

    plt.title(f"{y} grouped by {x} and model")
    plt.xticks(rotation=45)
    plt.grid()

    plt.tight_layout()
    plt.show()




def violin_plot(df, x, y="test_acc"):
    import seaborn as sns

    plt.figure()
    sns.violinplot(data=df, x=x, y=y)

    plt.title(f"{y} distribution by {x}")
    plt.xticks(rotation=45)

    plt.show()




def scatter_plot(df, x, y="test_acc", hue="model"):
    import seaborn as sns

    plt.figure()
    sns.scatterplot(data=df, x=x, y=y, hue=hue)

    plt.title(f"{y} vs {x}")
    plt.grid()

    plt.show()




def correlation_plot(df):
    import seaborn as sns

    numeric = df.select_dtypes(include=[np.number])

    plt.figure()
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")

    plt.title("Correlation matrix")
    plt.show()



def get_best_configs(df, top_k=5, metric="test_acc", model=None):
    if model is not None:
        df = df[df["model"] == model]
    return df.sort_values(metric, ascending=False).head(top_k)




def compare_models(df):
    summary = df.groupby("model").agg(
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_f1=("test_macro_f1", "mean"),
    ).reset_index()

    return summary.sort_values("mean_acc", ascending=False)


def save_summary(df, path):
    df.to_csv(path, index=False)
    print(f"Saved summary to {path}")