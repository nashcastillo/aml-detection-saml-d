# AML Detection — Projet Final Jedha Bootcamp

> **Construction d'un modèle de Machine Learning pour la détection de transactions suspectes (Anti Money Laundering — AML).**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📓 **Lire le notebook en ligne (sans installation) :**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nashcastillo/aml-detection-saml-d/blob/main/aml_detection_saml_d.ipynb)
[![Voir sur nbviewer](https://img.shields.io/badge/nbviewer-render-orange.svg)](https://nbviewer.org/github/nashcastillo/aml-detection-saml-d/blob/main/aml_detection_saml_d.ipynb)

🇬🇧 *English version available: [README.md](README.md)*

## 👥 Équipe

Projet réalisé dans le cadre de la formation **Data Essentials — Jedha Bootcamp** (Avril 2026).

- **Nashely Castillo** — Compliance Officer LCB-FT (cadrage métier, EDA, modélisation)
- **Gwladys Pioche** — Analyse exploratoire des données
- **Robin Pradier** — Modélisation Machine Learning

---

## 🎯 Question du projet

> **Peut-on prédire si une transaction est potentiellement liée au blanchiment d'argent ?**

L'objectif n'est pas de certifier qu'une transaction est suspecte, mais de **signaler un risque** pour réduire la charge des analystes humains.

### Le contexte

En 2024, **TRACFIN** (autorité française anti-blanchiment) a reçu **215 410 déclarations de soupçon** mais seules **3 998** ont été transmises à la justice — soit **environ 98 % de faux positifs**. Le Machine Learning peut-il améliorer ce ratio ?

---

## 📊 Le dataset — SAML-D

- **Source** : [BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D)
- **Taille originale** : 9 504 852 transactions
- **Échantillon de travail** : **800 000 lignes** (distribution préservée)
- **Variable cible** : `Is_laundering` (0 = normal / 1 = suspect)
- **Déséquilibre** : seulement **0,1 %** de transactions suspectes (833 cas)

![Distribution des classes](docs/images/class_imbalance.png)

| Catégorie | Variables |
|---|---|
| **Temps & Montant** | Date, Time, Amount |
| **Comptes** | Sender_account, Receiver_account |
| **Localisation** | Sender_bank_location, Receiver_bank_location |
| **Paiement** | Payment_type, Payment_currency, Received_currency |
| **Cible** | Is_laundering, Laundering_type |

---

## 🧪 Méthodologie

```
1. EDA          → Exploration & visualisation (déséquilibre, pays, typologies)
2. Preprocessing → LabelEncoder, StandardScaler, train/test stratifié, class_weight balanced
3. Modélisation  → 5 modèles testés (Logistic Regression × 3, Decision Tree, Random Forest)
4. Évaluation    → Recall comme métrique clé (en AML, mieux vaut une fausse alerte qu'un vrai blanchiment manqué)
```

---

## 🤖 Modèles testés & résultats

| # | Modèle | Configuration | Recall (test) | Statut |
|---|---|---|---|---|
| 1 | Régression logistique | `X = Amount` seul | 0 % | ❌ Inutilisable |
| 2 | Régression logistique | + `class_weight='balanced'` | 22 % | Faible |
| 3 | Régression logistique | + 6 features | 38 % | Moyen |
| 4 | **Decision Tree** | `max_depth=10`, balanced | **43 %** | ⭐ **RETENU** |
| 5 | Random Forest | 100 arbres, `max_depth=15` | 13 % | Rechute |

### Modèle retenu : Decision Tree (max_depth=10)

![Matrice de confusion](docs/images/confusion_matrix.png)

**Recall = TP / (TP + FN) = 71 / (71 + 96) ≈ 43 %**

### Train vs Test — diagnostic du sur-apprentissage

![Train vs Test metrics](docs/images/train_test_metrics.png)

Le recall passe de 0,71 (train) à 0,43 (test) — sur-apprentissage modéré. La précision et le F1-score restent quasi nuls à cause du déséquilibre 0,1 % : la plupart des transactions signalées sont des faux positifs. C'est pour cela qu'on **se concentre sur le recall**, pas la précision, en LCB-FT.

### Importance des features

![Importance des features](docs/images/feature_importance.png)

---

## 🚀 Reproduire le projet

### 1. Cloner le repo

```bash
git clone https://github.com/nashcastillo/aml-detection-saml-d.git
cd aml-detection-saml-d
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Récupérer le dataset

Le fichier complet (~84 Mo, 800 000 lignes) **n'est pas inclus** dans le repo (limite GitHub). Deux options :

- **Option A** : récupérer le dataset original sur [BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D](https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D) et générer un échantillon de 800k lignes
- **Option B** : utiliser l'échantillon réduit `data/SAML-D_sample_1k.csv` (1 000 lignes) inclus dans le repo pour tester le pipeline

### 4. Lancer le notebook

```bash
jupyter notebook aml_detection_saml_d.ipynb
```

---

## 🎯 Pistes d'amélioration

| # | Axe | Technique | Impact attendu |
|---|---|---|---|
| 1 | Rééchantillonnage | **SMOTE** | Recall ↑ |
| 2 | Feature engineering | Patterns temporels et comportementaux | Recall & Precision ↑ |
| 3 | Modèle | **XGBoost** | Score global ↑ |
| 4 | Calibration | Ajustement du seuil de décision | Precision ↑ |

---

## 📚 Sources

- **Europol (2017)** — *From suspicion to action: Converting financial intelligence into greater operational impact*
- **TRACFIN (2024)** — Rapport d'activité annuel
- **Dataset SAML-D** — Oztas, B. (2023)

---

## 📄 Licence

Distribué sous [licence MIT](LICENSE) — utilisation, modification et redistribution libres avec attribution.

Projet pédagogique réalisé dans le cadre du Jedha Bootcamp.
