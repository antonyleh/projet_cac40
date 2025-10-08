# Objet : Projet ML Finance – Clustering Spectral & Analyse Discriminante (15 actions CAC40)

Nous avons choisi comme projet un jeu de données réel issu du domaine financier : nous travaillons sur **15 actions du CAC 40** sur une période de plusieurs années (prix journaliers).

L’objectif est de mettre en œuvre, sur ces données, les méthodes de clustering non supervisé et de classification supervisée vues en cours, en suivant la logique suivante :

---

## 1. Préparation des données
- Téléchargement des prix de clôture journaliers (Yahoo Finance).
- Calcul des rendements log `r_t = ln(P_t / P_{t-1})`.
- Construction de features classiques en finance :
  - Volatilité (écart-type des rendements).
  - Drawdown max.
  - VaR (Value-at-Risk).
  - Rendement moyen et Sharpe ratio.
- Standardisation des variables (z-score).

---

## 2. Partie non supervisée (Clustering)
- **K-Means** comme baseline.
- **Clustering spectral** (affinité RBF, Laplacien normalisé, plongement spectral, puis K-Means dans l’espace spectral).
- Visualisations prévues :
  - Courbe des petites valeurs propres du Laplacien (spectre).
  - Embedding spectral 2D coloré par cluster.
  - Comparaison avec PCA 2D + K-Means.
- Interprétation : mise en évidence de groupes d’actions (par ex. “défensives” vs “volatiles”).

---

## 3. Partie supervisée (Classification)
- Définition d’un label de risque (ex. : volatilité supérieure à un seuil → “risqué”, sinon “non-risqué”).
- Application de :
  - **LDA** (frontières linéaires),
  - **QDA** (frontières quadratiques).
- Comparaison visuelle des frontières de décision sur 2 variables (ex. Volatilité vs Sharpe).
- Évaluation via matrices de confusion.

---

## 4. Résultats attendus
- Montrer que le clustering spectral permet de détecter des structures plus fines que K-Means.
- Illustrer la différence entre frontières linéaires (LDA) et quadratiques (QDA).
- Relier les résultats à une interprétation financière simple (classification d’actifs selon leur profil de risque).

---

## 5. Livrables
- **Notebook Python** : code et visualisations (K-Means, Spectral, LDA/QDA).
- **Vidéo explicative (5–8 min)** :
  1. Présentation du jeu de données (actions CAC40).
  2. Clustering (K-Means vs Spectral).
  3. Classification (LDA vs QDA).
  4. Conclusion et lien avec les notions vues en cours.

---

## Conclusion
Ce projet nous semble cohérent avec le contenu du cours :
- Non supervisé → K-Means & Spectral Clustering.
- Supervisé → Analyse discriminante (LDA/QDA).
- Et il s’appuie sur des données réelles et interprétables.

Nous restons bien sûr ouverts à vos retours ou suggestions d’amélioration.
