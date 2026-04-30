"""Generate the 3 key figures for the README.

Outputs (PNG, ~150 dpi) in docs/images/:
  - class_imbalance.png      → donut chart of class distribution
  - confusion_matrix.png     → heatmap of test confusion matrix
  - feature_importance.png   → horizontal bar chart of feature importances

Run from the projet_final_merged/ folder:
    python3 generate_figures.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Try the full dataset first, fall back to the small sample
CSV_FULL = ROOT.parent / "SAML-D_sample_800k.csv"
CSV_SAMPLE = ROOT / "data" / "SAML-D_sample_1k.csv"

if CSV_FULL.exists():
    csv_path = CSV_FULL
    print(f"Loading full dataset: {csv_path.name}")
else:
    csv_path = CSV_SAMPLE
    print(f"Loading sample: {csv_path.name}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows ({df['Is_laundering'].sum()} suspicious, {df['Is_laundering'].mean()*100:.3f}%)")


# ============================================================
# FIGURE 1 — Class imbalance (donut chart)
# ============================================================
counts = df["Is_laundering"].value_counts()
labels = [f"Normal\n{counts[0]:,} ({counts[0]/len(df)*100:.2f}%)",
          f"Suspicious\n{counts[1]:,} ({counts[1]/len(df)*100:.3f}%)"]
sizes = [counts[0], counts[1]]
colors = ["#2E86C1", "#E74C3C"]

fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts = ax.pie(
    sizes, labels=labels, colors=colors, startangle=90,
    wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11, "fontweight": "bold"},
)
ax.set_title("Class Distribution — Extreme Imbalance",
             fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'class_imbalance.png'}")


# ============================================================
# Preprocessing for modeling (Decision Tree max_depth=10)
# Same preprocessing as notebook cell #40 (Modele 4)
# ============================================================
df_tree = df.copy()

# Encode categorical columns (no .astype(str) — same as notebook)
cat_cols = ["Payment_type", "Payment_currency", "Received_currency",
            "Sender_bank_location", "Receiver_bank_location"]
for col in cat_cols:
    df_tree[col] = LabelEncoder().fit_transform(df_tree[col])

features = ["Amount", "Payment_type", "Payment_currency",
            "Received_currency", "Sender_bank_location", "Receiver_bank_location"]
X = df_tree[features]
y = df_tree["Is_laundering"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(
    max_depth=10, class_weight="balanced", random_state=42
)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)


# ============================================================
# FIGURE 2 — Confusion matrix (test)
# ============================================================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

annot = np.array([
    [f"TN\n{tn:,}", f"FP\n{fp:,}"],
    [f"FN\n{fn:,}", f"TP\n{tp:,}"],
])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=annot, fmt="", cmap="Blues",
    xticklabels=["Predicted Normal", "Predicted Suspicious"],
    yticklabels=["Actual Normal", "Actual Suspicious"],
    annot_kws={"fontsize": 13, "fontweight": "bold"},
    cbar_kws={"label": "Count"}, ax=ax,
)
ax.set_title(f"Confusion Matrix — Decision Tree (max_depth=10)\nRecall = {recall*100:.1f}%",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'confusion_matrix.png'} (recall={recall*100:.1f}%)")


# ============================================================
# FIGURE 3 — Feature importance
# ============================================================
importances = pd.Series(clf.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))
ax.barh(importances.index, importances.values * 100, color=colors_bar, edgecolor="white")
for i, (name, val) in enumerate(importances.items()):
    ax.text(val * 100 + 0.5, i, f"{val*100:.1f}%",
            va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Importance (%)", fontsize=11)
ax.set_title("Feature Importance — Decision Tree (max_depth=10)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlim(0, importances.max() * 100 * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'feature_importance.png'}")

# ============================================================
# FIGURE 4 — Train vs Test comparison
# ============================================================
metrics_train = [
    recall_score(y_train, y_train_pred),
    precision_score(y_train, y_train_pred, zero_division=0),
    f1_score(y_train, y_train_pred),
]
metrics_test = [
    recall_score(y_test, y_pred),
    precision_score(y_test, y_pred, zero_division=0),
    f1_score(y_test, y_pred),
]
labels = ["Recall", "Precision", "F1-Score"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, metrics_train, width, label="Train", color="#2E86C1")
bars2 = ax.bar(x + width / 2, metrics_test, width, label="Test", color="#E74C3C")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=11, fontweight="bold")

ax.axhline(metrics_test[0], color="#E74C3C", linestyle="--", alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, max(metrics_train) * 1.15)
ax.set_title("Train vs Test — Decision Tree (max_depth=10)\nSuspicious Class (1)",
             fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "train_test_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'train_test_metrics.png'}")

print("\nAll 4 figures generated.")
