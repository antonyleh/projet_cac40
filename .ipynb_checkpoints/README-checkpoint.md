# Analyse des profils de risque des actions du CAC 40

## 1. Contexte et objectif

L’objectif du projet est d’extraire, à partir de séries de prix journaliers des actions du CAC 40, des **profils de risque / performance** et d’étudier dans quelle mesure ces profils permettent :

- de **regrouper** les actions en familles homogènes (apprentissage non supervisé) ;
- de **prédire** ensuite des indicateurs de risque, voire le **signe du rendement du lendemain** (apprentissage supervisé).

Tout le travail est fait avec une logique **temporelle** (fenêtres roulantes, split train/test chronologique) pour rester réaliste du point de vue financier.

---

## 2. Données et construction des features

À partir des prix journaliers `prices_cac40.csv`, on calcule d’abord les rendements `returns_cac40.csv`, puis, sur des **fenêtres roulantes de 60 jours**, les descripteurs financiers suivants :

- **Rendement moyen annualisé**  
  \(\mu_{\text{ann}} = \bar r \times 252\)

- **Volatilité annualisée**  
  \(\sigma_{\text{ann}} = \text{std}(r)\sqrt{252}\)

- **Sharpe annualisé (sans taux sans risque)**  
  \(S = \mu_{\text{ann}} / (\sigma_{\text{ann}} + \varepsilon)\)

- **VaR 95 % (positive)**  
  \(VaR_{95} = -q_{5\%}(r)\) : perte attendue dans les 5 % pires cas

- **Drawdowns**  
  - drawdown courant : perte par rapport au plus-haut de la fenêtre ;
  - max drawdown sur 60 jours : pire drawdown dans la fenêtre.

- **Skewness & kurtosis** des rendements : asymétrie et queue épaisse.

- **Bêta “leave-one-out”** :  
  régression de l’actif \(i\) sur l’indice égal-pondéré des **autres** titres du CAC (on exclut \(i\) du “marché” pour éviter l’auto-corrélation).

Les features sont ensuite **standardisés globalement** (`features_scaled_global.csv`), puis nettoyés (`features_scaled_global_clean.csv`) : suppression des dates entièrement manquantes et gestion conservatrice des valeurs extrêmes (mais sans supprimer les épisodes de crise).

---

## 3. Exploration des données (EDA)

L’EDA confirme la nature très particulière des données financières :

- Les **boxplots** montrent beaucoup d’outliers (volatilité, VaR, drawdowns, kurtosis, bêta) : ce sont essentiellement des **épisodes de stress** (crises, krachs, rebonds violents), que l’on choisit de **conserver**.
- La **heatmap de corrélation** révèle :
  - un bloc “**risque de marché**” : \(\sigma_{\text{ann}}\), VaR95, drawdowns fortement corrélés ;
  - un bloc “**performance**” : \(\mu_{\text{ann}}\), Sharpe fortement corrélés entre eux ;
  - skew / kurt et bêta apportent une information plus indépendante.
- Les **scatters** mettent en évidence :
  - relation quasi linéaire entre \(\sigma_{\text{ann}}\) et VaR95 → redondance ;
  - relation non linéaire entre \(\sigma_{\text{ann}}\) et Sharpe → les périodes très volatiles ne sont pas forcément les mieux rémunérées ;
  - bêta vs Sharpe : nuage rond, peu de corrélation → bêta mesure surtout l’exposition au marché.

Sur cette base, on retient un sous-ensemble de features **non redondants** pour la suite :
\[
\{\mu_{\text{ann}}, \sigma_{\text{ann}}, Sharpe_{\text{ann}}, \beta_{\text{ewm}},
mdd\_window, skew, kurt\}
\]

---

## 4. Apprentissage non supervisé : profils d’actions

### 4.1. Résumé par action

Pour chaque ticker, on calcule la **moyenne temporelle** des 7 features sur toute la période, ce qui donne un vecteur “**profil moyen**” par action.  
On obtient ainsi une matrice `X_assets` de taille (nb d’actions × 7).

### 4.2. Choix du nombre de clusters

Sur `X_assets`, on applique K-means pour différents \(k\) et on trace :

- la **méthode du coude (Elbow)** sur l’inertie ;
- le **score de silhouette**.

Résultat : pas de “coude” très net sur l’inertie, mais un **maximum clair du score de silhouette pour \(k=2\)**.  
On retient donc **\(k=2\)** : deux grands profils d’actions.

### 4.3. K-means vs clustering spectral

On compare ensuite :

- **K-means (k=2)**
- **Clustering spectral (k=2, affinité RBF)**

Les deux méthodes sont visualisées après **PCA 2D** (PCA uniquement pour la visualisation, clustering fait en 7D).

Constats :

- Les deux méthodes produisent un groupe majoritaire d’actions “standard” et un groupe minoritaire “plus extrême”.
- Le **clustering spectral** semble plus robuste à quelques titres atypiques (volatilité extrême, drawdown très fort), et regroupe mieux entre elles certaines valeurs clairement risquées.

On décide donc de **conserver le clustering spectral** comme segmentation principale.  
Chaque action reçoit un label de cluster `cluster_spectral ∈ {0,1}`.

### 4.4. Interprétation financière

En regardant la **composition des clusters** et les **moyennes par cluster** :

- **Cluster 0** : majorité des titres (profil “standard”)  
  → volatilité et drawdowns plus modérés, Sharpe plus “normal”.

- **Cluster 1** : petit groupe de titres plus extrêmes  
  → volatilité plus élevée, drawdowns plus profonds, souvent Sharpe plus extrême (profils plus risqués / cycliques).

Cette segmentation est utilisée ensuite comme **label de risque** pour l’apprentissage supervisé.

---

## 5. Apprentissage supervisé 1 : prédire le profil de risque (cluster 0/1)

### 5.1. Construction du dataset

- On revient au jeu de données par observation (date, ticker) : `features_scaled_split.csv`.
- Split temporel : **75 % train / 25 % test** (train avant une date de coupure, test après).
- Standardisation recalculée **uniquement sur le train** pour éviter le leakage, puis appliquée au test.
- Label cible :  
  \[
  y_{\text{risque}} = 
  \begin{cases}
  0 & \text{si l’action appartient au cluster “standard”}\\
  1 & \text{sinon (profil risqué)}
  \end{cases}
  \]

### 5.2. Modèles et résultats

On entraîne plusieurs modèles vus en cours :

- **LDA (Linear Discriminant Analysis)**
- **QDA (Quadratic Discriminant Analysis)**
- **k-NN (k=7)**

Tous atteignent une **accuracy élevée (≈ 0.90–0.93)** sur le set de test, avec des nuances :

- LDA : très bonne précision, mais recall plus faible sur la classe 1 (fenêtres risquées).  
- QDA : recall plus élevé sur la classe 1, au prix de quelques erreurs supplémentaires sur la classe 0.  
- k-NN : performance intermédiaire.

Les **matrices de confusion** montrent que la plupart des fenêtres “standard” sont bien classées, et que la détection des fenêtres risquées est correcte, surtout avec QDA.

On trace aussi les **frontières de décision LDA/QDA** dans le plan
\((\sigma_{\text{ann}}, Sharpe_{\text{ann}})\) :  
on visualise clairement que la classification se fait principalement sur la **volatilité** et le **rendement ajusté du risque**, ce qui correspond aux intuitions issues de l’EDA.

**Conclusion de cette partie :**  
Les features conçus à partir des prix suffisent à reproduire correctement la segmentation non supervisée en profils de risque, avec des modèles simples comme LDA / QDA / k-NN.

---

## 6. Apprentissage supervisé 2 : prédire le signe du rendement

On considère maintenant une tâche plus ambitieuse :  
**prédire le signe du rendement du lendemain** à partir des features de risque sur la fenêtre courante.

### 6.1. Cible et modèle

- Label :  
  \(y_{\text{dir}} = 1\) si \(r_{t+1} > 0\), 0 sinon.
- Features : les 7 indicateurs \(\{\mu_{\text{ann}}, \sigma_{\text{ann}}, Sharpe_{\text{ann}}, \beta_{\text{ewm}}, mdd\_window, skew, kurt\}\).
- Split temporel et standardisation comme précédemment.
- Modèle : **régression logistique** (penalty L2 standard).

### 6.2. Résultats

Sur le set de test :

- Accuracy modèle : ~**0.52**  
- Accuracy baseline (classe majoritaire) : ~**0.51**
- AUC ROC : ~**0.51–0.52**

La **courbe ROC** est très proche de la diagonale (modèle aléatoire), et la matrice de confusion montre un comportement assez symétrique sans gain significatif par rapport au hasard.

L’analyse des coefficients indique :

- Effet légèrement **négatif** de \(\mu_{\text{ann}}\) (reversion à la moyenne possible : après de très bons rendements récents, le modèle n’anticipe pas forcément un nouveau jour positif).
- Effet faible mais **positif** de \(mdd\_window\), \(\sigma_{\text{ann}}\) et Sharpe → légère tendance à prévoir un rebond après des phases risquées.

Mais ces effets restent **trop faibles** pour construire un signal de trading robuste : la prévision de \(sign(r_{t+1})\) avec un modèle linéaire simple sur ces features est **quasi impossible**.

---

## 7. Conclusion générale

- L’EDA met en évidence des **régimes de marché** (calmes vs stressés), avec volatilité, drawdowns et kurtosis très asymétriques.
- L’**apprentissage non supervisé** (K-means, clustering spectral) permet de regrouper les actions du CAC 40 en **deux grands profils de risque**, interprétables financièrement.
- Les modèles **supervisés simples** (LDA, QDA, k-NN) sont capables de **retrouver ces profils** à partir des features, avec une bonne précision.
- En revanche, tenter de prédire la **direction du rendement journalier** avec une **régression logistique** ne dépasse pratiquement pas le hasard, ce qui est cohérent avec l’idée d’un marché largement efficient à cet horizon.

Ce projet montre donc à la fois :

1. que des indicateurs simples de risque/retour suffisent pour **segmenter le marché** et identifier des profils d’actions ;
2. mais que **prédire le signe du rendement à court terme** reste extrêmement difficile, même avec des méthodes de machine learning classique.

