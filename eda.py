import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import os

OUT = "/app/output"
os.makedirs(OUT, exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# ── 1. Scatter petal_length vs petal_width ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for species, group in df.groupby("species"):
    ax.scatter(group["petal length (cm)"], group["petal width (cm)"], label=species, s=50)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal Length vs Petal Width")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/plot_scatter_petal.png", dpi=150)
plt.close()

# ── 2. Pair plot ─────────────────────────────────────────────────────────────
pair = sns.pairplot(df, hue="species", diag_kind="kde", plot_kws={"alpha": 0.6})
pair.fig.suptitle("Pair Plot — tutte le feature", y=1.02)
plt.savefig(f"{OUT}/plot_pairplot.png", dpi=150)
plt.close()

# ── 3. Box plot per feature e specie ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, iris.feature_names):
    df.boxplot(column=col, by="species", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Box Plot per specie", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/plot_boxplot.png", dpi=150)
plt.close()

# ── 4. Distribuzione KDE per feature ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, iris.feature_names):
    for species, group in df.groupby("species"):
        group[col].plot.kde(ax=ax, label=species)
    ax.set_title(col)
    ax.legend()
fig.suptitle("Distribuzione KDE per specie")
plt.tight_layout()
plt.savefig(f"{OUT}/plot_kde.png", dpi=150)
plt.close()

# ── 5. Heatmap correlazione ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.drop(columns="species").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Heatmap Correlazione")
plt.tight_layout()
plt.savefig(f"{OUT}/plot_heatmap.png", dpi=150)
plt.close()
