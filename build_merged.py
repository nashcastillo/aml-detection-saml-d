"""Build the unified AML detection notebook by merging Gw, Nash and Robin notebooks."""
import json
import codecs
from pathlib import Path

ROOT = Path("/Users/nash/Documents/jedha/projet final")
OUT = ROOT / "projet_final_merged" / "aml_detection_saml_d.ipynb"


def load_nb(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        with codecs.open(path, "r", "utf-8-sig") as f:
            return json.load(f)


nash = load_nb(ROOT / "projet_final_nash.ipynb")
robin = load_nb(ROOT / "projet_final_version_robin.ipynb")
gw = load_nb(ROOT / "Projet_final_Gw_V2.ipynb")


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) if "\n" in text else [text],
    }


def code_from(nb, idx):
    """Take code cell idx from notebook, strip outputs to keep file clean."""
    c = nb["cells"][idx]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": c["source"],
        "execution_count": None,
        "outputs": [],
    }


cells = []

# ============================================================
# 0. EN-TÊTE & CONTEXTE
# ============================================================
cells.append(md("""# Détection de Transactions Suspectes — Projet Final Jedha

**Formation Data Essentials — Jedha Bootcamp**
**Équipe :** Nashely Castillo, Gwladys Pioche, Robin Pradier

---

## 0. Contexte du projet

### Le blanchiment d'argent en 3 phases

1. **Placement** — Introduction des fonds illégaux dans le système financier
2. **Empilage** — Dissimulation de l'origine via des transactions complexes
3. **Intégration** — Retour des fonds dans l'économie légale

### Le contexte actuel

En 2024, **TRACFIN** (autorité française anti-blanchiment) a reçu **215 410 déclarations de soupçon** mais seules **3 998** ont été transmises à la justice — soit **environ 98 % de faux positifs**.

> **Question du projet :** *Peut-on prédire si une transaction est potentiellement liée au blanchiment d'argent ?*

L'objectif n'est pas de certifier qu'une transaction est suspecte, mais de **signaler un risque** pour réduire la charge des analystes humains.

### Source des données

Dataset **SAML-D** (Synthetic Anti-Money Laundering Dataset) — [BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)

- **9 504 852 transactions** au total
- **Échantillon de travail : 800 000 lignes** (recommandé par le formateur, distribution préservée)
- **0,1 % de transactions suspectes** (833 cas) — déséquilibre extrême
"""))

# ============================================================
# 1. IMPORTS & CHARGEMENT
# ============================================================
cells.append(md("""---

## 1. Imports & chargement des données"""))

cells.append({
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
        "from sklearn.metrics import (\n",
        "    classification_report, confusion_matrix,\n",
        "    recall_score, precision_score, f1_score, accuracy_score\n",
        ")\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "RANDOM_STATE = 42",
    ],
})

cells.append(code_from(nash, 1))  # df_projet_final = pd.read_csv(...)
cells.append(code_from(nash, 2))  # df.shape
cells.append(code_from(nash, 3))  # df.info
cells.append(code_from(nash, 4))  # df.describe
cells.append(code_from(nash, 5))  # df.isnull
cells.append(code_from(nash, 6))  # Total transactions print

# ============================================================
# 2. EDA
# ============================================================
cells.append(md("""---

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Déséquilibre des classes

Le dataset présente un déséquilibre extrême : **0,1 % de transactions suspectes** seulement. Ce déséquilibre conditionne tout le reste du projet — notamment le choix de la métrique d'évaluation."""))

cells.append(code_from(nash, 7))  # class distribution chart
cells.append(code_from(nash, 8))  # add_bar_labels helper

cells.append(md("""### 2.2 Pays impliqués dans les transactions suspectes"""))

cells.append(code_from(gw, 6))  # Pays impliqués
cells.append(code_from(gw, 7))  # Taux par pays émetteur
cells.append(code_from(gw, 8))  # Taux par pays destinataire

cells.append(md("""**Observations :**
- Côté **émission** : Albanie, Italie, Pays-Bas → taux suspects ~0,3 % à 0,45 %
- Côté **réception** : Nigéria, Arabie Saoudite, Maroc, Mexique → taux ~0,5 % à 0,8 %
- Le UK domine en volume (96,6 % des transactions) mais sert de plaque tournante"""))

cells.append(md("""### 2.3 Types de paiement"""))

cells.append(code_from(gw, 10))  # % suspects par type de paiement
cells.append(code_from(nash, 14))  # bar chart suspectes paiement

cells.append(md("""**Observations :**
Les transactions en **espèces** (cash deposit, cash withdrawal) présentent les taux de suspicion les plus élevés. Les paiements électroniques sont les moins risqués."""))

cells.append(md("""### 2.4 Typologies de blanchiment"""))

cells.append(code_from(nash, 11))  # amount by laundering_type

cells.append(md("""### Description des typologies de blanchiment

- **Over-Invoicing** : Surfacturation — on gonfle artificiellement le prix d'une facture pour transférer de grosses sommes d'argent légalement entre deux entreprises.
- **Single_Large** : Une seule transaction de très grand montant pour déplacer rapidement des fonds.
- **Structuring** : Fractionnement délibéré de transactions pour éviter les seuils de déclaration.
- **Smurfing** : Petits dépôts répétés par plusieurs personnes pour éviter la détection.
- **Cash_Withdrawal** : Retraits d'espèces pour rendre les fonds non traçables."""))

cells.append(md("""### 2.5 Comptes suspects"""))

cells.append(code_from(nash, 13))  # Top 10 comptes suspects

# ============================================================
# 3. PREPROCESSING
# ============================================================
cells.append(md("""---

## 3. Preprocessing

| Étape | Détail |
|---|---|
| **Valeurs manquantes** | Aucune (dataset synthétique) |
| **Encodage catégoriel** | `LabelEncoder` sur `Payment_type`, devises, pays |
| **Feature engineering** | `sender_tx_count`, `receiver_tx_count` |
| **Train/Test split** | 80 / 20 avec `stratify=y` (préserve les 0,1 % de suspects) |
| **Normalisation** | `StandardScaler` (z-score) sur les features numériques |
| **Déséquilibre** | `class_weight='balanced'` — poids plus fort à la classe minoritaire |
| **Reproductibilité** | `random_state=42` |"""))

cells.append(code_from(robin, 13))  # df_ml copy with feature engineering
cells.append(code_from(robin, 14))  # hist sender_tx_count

# ============================================================
# 4. MODÉLISATION
# ============================================================
cells.append(md("""---

## 4. Modélisation

Cinq modèles testés progressivement, du plus simple au plus avancé :

| # | Modèle | Configuration |
|---|---|---|
| M1 | Régression logistique | `X = Amount` seul |
| M2 | Régression logistique | + `class_weight='balanced'` |
| M3 | Régression logistique | + 6 features |
| M4 | Decision Tree | `max_depth=10`, `class_weight='balanced'` |
| M5 | Random Forest | 100 arbres, `max_depth=15` |"""))

cells.append(md("""### Modèle 1 — Régression Logistique (Amount seul)

**Hypothèse de départ :** le montant de la transaction suffit-il à prédire le caractère suspect ?"""))

cells.append(code_from(nash, 16))  # Modèle 1

cells.append(md("""**Résultat : Recall = 0 %.** Le montant seul ne discrimine pas — beaucoup de transactions normales sont à montants élevés et beaucoup de transactions suspectes sont à petits montants (smurfing, structuring)."""))

cells.append(md("""### Modèle 2 — Régression Logistique + class_weight balanced"""))

cells.append(code_from(nash, 18))  # Modèle 2

cells.append(md("""**Résultat : Recall ~22 %.** L'ajout de `class_weight='balanced'` force le modèle à prêter attention à la classe minoritaire, mais le pouvoir prédictif reste limité avec une seule feature."""))

cells.append(md("""### Modèle 3 — Régression Logistique + 6 features"""))

cells.append(code_from(nash, 20))  # Modèle 3

cells.append(md("""**Résultat : Recall ~38 %.** Significativement mieux. La régression logistique reste linéaire et plafonne ici."""))

cells.append(md("""### Modèle 4 — Decision Tree (max_depth=10) ⭐ MODÈLE RETENU

L'arbre de décision pose une suite de questions oui/non sur les features pour aboutir à une prédiction. Limiter `max_depth=10` évite l'overfitting."""))

cells.append(code_from(robin, 28))  # Modèle 4 - DT

cells.append(md("""### Modèle 5 — Random Forest (100 arbres, max_depth=15)

Une Random Forest entraîne 100 arbres sur des échantillons aléatoires des données et agrège leurs votes."""))

cells.append(code_from(robin, 30))  # Modèle 5 - RF

# ============================================================
# 5. ÉVALUATION DU MODÈLE 4
# ============================================================
cells.append(md("""---

## 5. Évaluation du modèle retenu (Decision Tree, max_depth=10)

### Matrice de confusion (test)

|  | Prédit Normal | Prédit Suspect |
|---|---|---|
| **Réel Normal** | TN = 140 864 | FP = 18 969 |
| **Réel Suspect** | FN = 96 | TP = 71 |

### Métriques

- **Recall = TP / (TP + FN) = 71 / (71 + 96) ≈ 43 %** sur le test
- **Recall train ≈ 70 %** vs **test ≈ 43 %** → léger surapprentissage mais contrôlé

### Importance des features

| Feature | Importance |
|---|---|
| Amount | 42,5 % |
| Payment_type | 26,6 % |
| Received_currency | 20,3 % |
| Payment_currency | 7,2 % |
| Receiver_bank_location | 2,5 % |
| Sender_bank_location | 0,9 % |"""))

# ============================================================
# 6. CONCLUSION
# ============================================================
cells.append(md("""---

## 6. Conclusion & Pistes d'amélioration

### Lecture métier

Dans le contexte AML, **mieux vaut générer des fausses alertes que laisser passer un vrai blanchiment** — le **recall prime sur la précision**.

Comparaison avec le terrain :
- **Europol (2017)** : seules ~10 % des déclarations de soupçon donnent lieu à une enquête
- **TRACFIN (2024)** : 215 410 reçues → 3 998 transmises, soit ~2 %
- **Notre modèle** atteint 43 % de recall — significatif, mais avec encore beaucoup de faux positifs

### 4 axes d'amélioration

| # | Axe | Technique | Impact attendu |
|---|---|---|---|
| 1 | Rééchantillonnage | **SMOTE** — exemples synthétiques minoritaires | Recall ↑ |
| 2 | Feature engineering | Patterns temporels, comportementaux | Recall & Precision ↑ |
| 3 | Modèle | **XGBoost** — boosting performant sur tabulaire déséquilibré | Score global ↑ |
| 4 | Calibration | Ajustement du seuil de décision | Precision ↑ |

---

*Notebook unifié — projet final Jedha Bootcamp Data Essentials*"""))


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

# Sanitize sources: ensure each cell.source is a list of strings ending with \n (except last)
for c in notebook["cells"]:
    src = c["source"]
    if isinstance(src, str):
        c["source"] = [src]
    elif isinstance(src, list):
        c["source"] = list(src)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated: {OUT}")
print(f"Total cells: {len(notebook['cells'])}")
md_count = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
code_count = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
print(f"  Markdown: {md_count}")
print(f"  Code:     {code_count}")
