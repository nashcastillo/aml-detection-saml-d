"""Build the English version of the unified AML detection notebook."""
import json
import codecs
from pathlib import Path

ROOT = Path("/Users/nash/Documents/jedha/projet final")
SRC = ROOT / "projet_final_merged" / "aml_detection_saml_d.ipynb"
OUT = ROOT / "projet_final_merged" / "aml_detection_saml_d_en.ipynb"


def load_nb(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with codecs.open(path, "r", "utf-8-sig") as f:
            return json.load(f)


nb = load_nb(SRC)


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if "\n" in text else [text],
    }


def code(source_lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }


def get_code(idx):
    """Reuse a code cell from the FR notebook by index, stripping outputs."""
    c = nb["cells"][idx]
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(c["source"]) if isinstance(c["source"], list) else [c["source"]],
    }


def translate_code_comments(idx, replacements):
    """Take code cell at idx and apply text replacements (FR→EN comments)."""
    c = nb["cells"][idx]
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for fr, en in replacements:
        src = src.replace(fr, en)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

# ============================================================
# 0. HEADER & CONTEXT
# ============================================================
cells.append(md("""# Suspicious Transaction Detection — Jedha Bootcamp Final Project

**Data Essentials Bootcamp — Jedha**
**Team:** Nashely Castillo, Gwladys Pioche, Robin Pradier

---

## 0. Project context

### The 3 phases of money laundering

1. **Placement** — Introducing illegal funds into the financial system
2. **Layering** — Hiding the origin through complex transactions
3. **Integration** — Returning the funds to the legal economy

### The current context

In 2024, **TRACFIN** (the French anti-money laundering authority) received **215,410 suspicious activity reports**, but only **3,998** were forwarded to the judiciary — that is, **about 98% false positives**.

> **Project question:** *Can we predict whether a transaction is potentially related to money laundering?*

The goal is not to certify that a transaction is suspicious, but to **flag a risk** in order to reduce the workload of human analysts.

### Data source

**SAML-D** dataset (Synthetic Anti-Money Laundering Dataset) — [BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)

- **9,504,852 transactions** in total
- **Working sample: 800,000 rows** (recommended by the trainer, distribution preserved)
- **0.1% suspicious transactions** (833 cases) — extreme imbalance
"""))

# ============================================================
# 1. IMPORTS & LOADING
# ============================================================
cells.append(md("""---

## 1. Imports & data loading"""))

# Imports cell — already in English, keep as-is
cells.append(get_code(2))

# Smart loader — translate the comments and prints
cells.append(code([
    "# Data loading\n",
    "# Tries the full 800k dataset first (for local use), otherwise falls back to the 1k sample\n",
    "import os\n",
    "\n",
    "CSV_FULL = '../SAML-D_sample_800k.csv'\n",
    "CSV_SAMPLE = 'data/SAML-D_sample_1k.csv'\n",
    "\n",
    "if os.path.exists(CSV_FULL):\n",
    "    csv_path = CSV_FULL\n",
    "    print(f\"Full dataset: {csv_path}\")\n",
    "elif os.path.exists(CSV_SAMPLE):\n",
    "    csv_path = CSV_SAMPLE\n",
    "    print(f\"Sample: {csv_path}\")\n",
    "    print(\"Note: using data/SAML-D_sample_1k.csv (1,000 rows). For real results, get the 800k dataset.\")\n",
    "else:\n",
    "    raise FileNotFoundError(\"No dataset found. See README to download SAML-D.\")\n",
    "\n",
    "df = pd.read_csv(csv_path)\n",
    "df.head()\n",
]))

# .shape, .info, .describe, .isnull are language-neutral
cells.append(get_code(4))  # df.shape
cells.append(get_code(5))  # df.info
cells.append(get_code(6))  # df.describe
cells.append(get_code(7))  # df.isnull

# total transactions print — translate
cells.append(translate_code_comments(8, [
    ('Total transactions', 'Total transactions'),
    ('# is_laundering : 1 (laundering), 0 (not laundering)',
     '# is_laundering: 1 (laundering), 0 (not laundering)'),
    ('df_projet_final', 'df'),
]))

# ============================================================
# 2. EDA
# ============================================================
cells.append(md("""---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Class imbalance

The dataset is extremely imbalanced: only **0.1% suspicious transactions**. This imbalance drives every choice that follows — especially the evaluation metric."""))

cells.append(translate_code_comments(10, [
    ('df_projet_final', 'df'),
]))
cells.append(get_code(11))  # add_bar_labels helper (no FR text)

cells.append(md("""### 2.2 Countries involved in suspicious transactions"""))

cells.append(translate_code_comments(13, [
    ('#Pays impliqués dans des transactions suspectes <-> Nb de transactions suspectes par pays',
     '# Countries involved in suspicious transactions <-> Number of suspicious transactions per country'),
    ('suspectes', 'suspicious'),
]))
cells.append(translate_code_comments(14, [
    ('#Taux de transactions suspectes par pays émetteur',
     '# Suspicious transaction rate by sender country'),
    ('suspectes', 'suspicious'),
]))
cells.append(translate_code_comments(15, [
    ('#Taux de transactions suspectes par pays destinataire',
     '# Suspicious transaction rate by receiver country'),
    ('suspectes', 'suspicious'),
]))

cells.append(md("""**Observations:**
- **Sender side**: Albania, Italy, Netherlands → high suspicious rate (~0.3% to 0.45%)
- **Receiver side**: Nigeria, Saudi Arabia, Morocco, Mexico → highest rates (~0.5% to 0.8%)
- The UK dominates in volume (96.6% of transactions) but acts as a hub"""))

cells.append(md("""### 2.3 Payment types"""))

cells.append(translate_code_comments(18, [
    ('# %tage des cas suspects par type de payments',
     '# Percentage of suspicious cases by payment type'),
    ('suspectes', 'suspicious'),
]))
cells.append(translate_code_comments(19, [
    ('df_projet_final', 'df'),
    ('suspectes_paiement', 'suspicious_payment'),
    ('couleurs', 'colors'),
]))

cells.append(md("""**Observations:**
**Cash transactions** (cash deposit, cash withdrawal) show the highest suspicion rates. Electronic payments are the least risky."""))

cells.append(md("""### 2.4 Money laundering typologies"""))

cells.append(translate_code_comments(22, [
    ('df_projet_final', 'df'),
    ('suspectes', 'suspicious'),
]))

cells.append(md("""### Money laundering typology descriptions

- **Over-Invoicing**: artificially inflating an invoice price to legally transfer large sums of money between two companies.
- **Single_Large**: a single very large transaction to move funds quickly.
- **Structuring**: deliberately splitting transactions to avoid reporting thresholds.
- **Smurfing**: small repeated deposits by multiple people to avoid detection.
- **Cash_Withdrawal**: cash withdrawals to make funds untraceable."""))

cells.append(md("""### 2.5 Suspicious accounts"""))

cells.append(translate_code_comments(25, [
    ('df_projet_final', 'df'),
    ('# Top 10 comptes expéditeurs les plus suspects',
     '# Top 10 most suspicious sender accounts'),
]))

# ============================================================
# 3. PREPROCESSING
# ============================================================
cells.append(md("""---

## 3. Preprocessing

| Step | Detail |
|---|---|
| **Missing values** | None (synthetic dataset) |
| **Categorical encoding** | `LabelEncoder` on `Payment_type`, currencies, countries |
| **Feature engineering** | `sender_tx_count`, `receiver_tx_count` |
| **Train/Test split** | 80 / 20 with `stratify=y` (preserves the 0.1% suspicious in each set) |
| **Normalization** | `StandardScaler` (z-score) on numerical features |
| **Imbalance** | `class_weight='balanced'` — higher weight on the minority class |
| **Reproducibility** | `random_state=42` |"""))

cells.append(translate_code_comments(27, [
    ('# 1. Créer une copie du dataset', '# 1. Make a copy of the dataset'),
    ('df_projet_final', 'df'),
    ('df_ml', 'df_ml'),
]))
cells.append(translate_code_comments(28, [
    ('Distribution du nombre de transactions par expéditeur',
     'Distribution of transaction count per sender'),
    ('Nombre de transactions envoyées par compte',
     'Number of transactions sent per account'),
    ('Nombre de comptes', 'Number of accounts'),
]))

# ============================================================
# 4. MODELING
# ============================================================
cells.append(md("""---

## 4. Modeling

Five models tested progressively, from simplest to most advanced:

| # | Model | Configuration |
|---|---|---|
| M1 | Logistic Regression | `X = Amount` only |
| M2 | Logistic Regression | + `class_weight='balanced'` |
| M3 | Logistic Regression | + 6 features |
| M4 | Decision Tree | `max_depth=10`, `class_weight='balanced'` |
| M5 | Random Forest | 100 trees, `max_depth=15` |"""))

cells.append(md("""### Model 1 — Logistic Regression (Amount only)

**Initial hypothesis:** is the transaction amount alone enough to predict suspiciousness?"""))

cells.append(translate_code_comments(31, [
    ("# l'algorithme de régression logistique.", "# the logistic regression algorithm."),
    ("# la fonction qui sépare les données en train et test.",
     "# the function that splits data into train and test."),
    ("# les outils de preprocessing — StandardScaler pour normaliser les features.",
     "# preprocessing tools — StandardScaler to normalize features."),
    ('df_projet_final', 'df'),
]))

cells.append(md("""**Result: Recall = 0%.** Amount alone does not discriminate — many normal transactions have high amounts and many suspicious ones have small amounts (smurfing, structuring)."""))

cells.append(md("""### Model 2 — Logistic Regression + class_weight balanced"""))

cells.append(translate_code_comments(34, [
    ("# l'algorithme de régression logistique.", "# the logistic regression algorithm."),
    ("# la fonction qui sépare les données en train et test.",
     "# the function that splits data into train and test."),
    ("# les outils de preprocessing — StandardScaler pour normaliser les features.",
     "# preprocessing tools — StandardScaler to normalize features."),
    ('df_projet_final', 'df'),
]))

cells.append(md("""**Result: Recall ~22%.** Adding `class_weight='balanced'` forces the model to pay attention to the minority class, but predictive power is still limited with a single feature."""))

cells.append(md("""### Model 3 — Logistic Regression + 6 features"""))

cells.append(translate_code_comments(37, [
    ('## importation des librairies pour le machine learning',
     '## importing machine learning libraries'),
    ('df_projet_final', 'df'),
]))

cells.append(md("""**Result: Recall ~38%.** Significantly better. Logistic regression remains linear and plateaus here."""))

cells.append(md("""### Model 4 — Decision Tree (max_depth=10) ⭐ SELECTED MODEL

A decision tree asks a sequence of yes/no questions on the features to reach a prediction. Limiting `max_depth=10` prevents overfitting."""))

cells.append(translate_code_comments(40, [
    ('Modele 4 : Arbre de Decision', 'Model 4: Decision Tree'),
    ('Modèle 4 : Arbre de Décision', 'Model 4: Decision Tree'),
    ('df_projet_final', 'df'),
]))

cells.append(md("""### Model 5 — Random Forest (100 trees, max_depth=15)

A Random Forest trains 100 trees on random samples of the data and aggregates their votes."""))

cells.append(translate_code_comments(42, [
    ('Modele 5 : Foret Aleatoire', 'Model 5: Random Forest'),
    ('Modèle 5 : Forêt Aléatoire', 'Model 5: Random Forest'),
    ('df_projet_final', 'df'),
]))

# ============================================================
# 5. EVALUATION
# ============================================================
cells.append(md("""---

## 5. Evaluation of the selected model (Decision Tree, max_depth=10)

### Confusion matrix (test)

|  | Predicted Normal | Predicted Suspicious |
|---|---|---|
| **Actual Normal** | TN = 140,864 | FP = 18,969 |
| **Actual Suspicious** | FN = 96 | TP = 71 |

### Metrics

- **Recall = TP / (TP + FN) = 71 / (71 + 96) ≈ 43%** on test
- **Recall train ≈ 70%** vs **test ≈ 43%** → mild overfitting, controlled

### Feature importance

| Feature | Importance |
|---|---|
| Amount | 42.5% |
| Payment_type | 26.6% |
| Received_currency | 20.3% |
| Payment_currency | 7.2% |
| Receiver_bank_location | 2.5% |
| Sender_bank_location | 0.9% |"""))

# ============================================================
# 6. CONCLUSION
# ============================================================
cells.append(md("""---

## 6. Conclusion & Improvement areas

### Business reading

In an AML context, **it is better to generate false alerts than to let real money laundering slip through** — **recall takes priority over precision**.

Field comparison — share of suspicious activity reports leading to an investigation:

- **EU (Europol, 2017)**: slightly more than 10%
- **France (TRACFIN, 2024)**: slightly less than 2% (215,410 received → 3,998 forwarded)
- **Our model**: 43% recall — significant, but still many false positives

### 4 improvement areas

| # | Area | Technique | Expected impact |
|---|---|---|---|
| 1 | Resampling | **SMOTE** — synthetic minority examples | Recall ↑ |
| 2 | Feature engineering | Temporal & behavioral patterns | Recall & Precision ↑ |
| 3 | Model | **XGBoost** — strong on tabular, imbalanced data | Overall score ↑ |
| 4 | Calibration | Decision threshold tuning | Precision ↑ |

---

*Unified notebook — Jedha Bootcamp Data Essentials final project*"""))


# Build final notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Sanitize sources
for c in notebook["cells"]:
    src = c["source"]
    if isinstance(src, str):
        c["source"] = [src]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {OUT}")
print(f"Total cells: {len(notebook['cells'])}")
md_count = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
code_count = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
print(f"  Markdown: {md_count}")
print(f"  Code:     {code_count}")
