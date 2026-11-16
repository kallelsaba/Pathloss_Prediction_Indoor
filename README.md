# 📡 Analyseur de Pathloss Indoor

**Prédiction de l'affaiblissement du signal Wi-Fi et optimisation du placement des points d'accès en environnement intérieur**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io)

## 🎯 Objectif du projet

À partir d'un simple plan d'étage au format PNG, l'application permet de :

- **Prédire le pathloss** (perte de signal, en dB) en tout point du bâtiment, en 2D et en 3D (plusieurs étages) ;
- **Visualiser la couverture** sous forme de cartes de chaleur (2D) et de voxels interactifs (3D) ;
- **Optimiser le placement des points d'accès (AP)** pour atteindre un taux de couverture visé ;
- **Simuler avant installation** : repérer les zones d'ombre et estimer le nombre d'AP nécessaires.

Projet réalisé lors d'un stage en Machine Learning (Telcotec, été 2025).

## 🎬 Démonstration

[![Watch the video on YouTube](https://img.youtube.com/vi/Kkl2KkTb_hc/0.jpg)](https://www.youtube.com/watch?v=Kkl2KkTb_hc)

Lien direct : [YouTube Video](https://www.youtube.com/watch?v=Kkl2KkTb_hc)

## ✨ Fonctionnalités

### 🖼️ Extraction des murs depuis le plan
- Conversion en niveaux de gris, binarisation (seuil 127) et inversion (murs en blanc) ;
- Nettoyage morphologique (fermeture puis ouverture, noyau 3×3) ;
- Détection des contours et filtrage par aire, périmètre et compacité pour éliminer le bruit ;
- Mise à l'échelle pixel → mètre à partir des dimensions réelles saisies par l'utilisateur.

### 🧠 Prédiction hybride du pathloss
- **Modèle ML prioritaire** : XGBoost (un modèle 2D et un modèle 3D) ;
- **Repli théorique** si le modèle est absent ou renvoie une valeur invalide : formule de Friis avec atténuation par mur (6 dB par défaut) et par étage (15 dB par défaut) ;
- **Comptage des murs** entre deux points par le tracé d'une ligne de Bresenham : une entrée dans un mur compte pour un seul mur, quelle que soit son épaisseur.

### 🎨 Visualisation
- Cartes de chaleur 2D (Matplotlib) ;
- Visualisations 3D interactives (Plotly) ;
- Classification par zones de qualité : Excellent / Bon / Faible / Mauvais.

### ⚡ Optimisation du placement des AP
- **K-means** : regroupe les points à couvrir et place un AP au centre de chaque groupe ;
- **GMM + EM** (mélange gaussien) : modélise la distribution spatiale des points à couvrir ;
- **Greedy (glouton)** : ajoute les AP un par un en choisissant celui qui améliore le plus la couverture réelle ;
- **Optimisation automatique** : heuristique géométrique (analyse de l'étalement des récepteurs) combinée à du K-means pour estimer le nombre d'AP et leurs positions.

Dans tous les cas, les positions qui tombent dans un mur sont déplacées vers la zone libre la plus proche, puis la configuration est évaluée avec le modèle de pathloss.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Interface Streamlit (app.py)              │
│                           8 onglets                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│  ImageProcessor          │  Prédiction du pathloss           │
│  ├─ Binarisation         │  ├─ Modèle ML 2D / 3D (XGBoost)   │
│  ├─ Morphologie          │  ├─ Repli théorique (Friis)       │
│  └─ Contours / murs      │  └─ Comptage des murs (Bresenham) │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│  Heatmaps 2D / 3D        │  Optimiseurs de placement         │
│  Visualisations          │  K-means · GMM · Greedy · Auto    │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis
- Python 3.10 ou supérieur (le projet a été développé sous Python 3.12)
- pip

### Mise en route

```bash
git clone https://github.com/kallelsaba/Pathloss_Prediction_Indoor.git
cd Pathloss_Prediction_Indoor
pip install -r requirements.txt
```

### Dépendances

| Paquet | Utilité |
|---|---|
| `streamlit` | Interface web interactive |
| `opencv-python` | Binarisation, morphologie, contours |
| `scikit-image` | Analyse d'image complémentaire |
| `Pillow` | Chargement des images |
| `numpy`, `pandas`, `scipy` | Calcul numérique, données, interpolation |
| `matplotlib` | Cartes 2D |
| `plotly` | Visualisations 3D |
| `scikit-learn` | K-means, GMM, métriques, `GridSearchCV` |
| `xgboost` | Chargement et exécution des modèles XGBoost |
| `joblib` | Chargement du modèle 2D |

> ⚠️ Le fichier `requirements.txt` doit contenir **tous** les paquets ci-dessus. Les modèles sont chargés au format `pickle`/`joblib` : ne chargez que des fichiers dont vous maîtrisez la provenance. Selon la version de XGBoost installée, un avertissement de compatibilité peut apparaître au chargement des modèles sérialisés avec une version plus ancienne.

## 💻 Utilisation

```bash
streamlit run app.py
```

L'application est accessible sur `http://localhost:8501`.

### Format du plan
- Image **PNG**, fond blanc, **murs sombres** (noir de préférence) ;
- Contours nets, pas de texte ni d'annotations parasites, résolution suffisante.

### Les 8 onglets

1. **Calcul Pathloss 2D** : pathloss entre un émetteur et un récepteur ;
2. **Calcul Pathloss 3D** : idem avec plusieurs étages ;
3. **Génération Heatmap 2D** ;
4. **Génération Heatmap 3D** ;
5. **Optimisation Wifi 2D** : K-means, GMM + EM ou Greedy ;
6. **Optimisation Wifi 3D** : K-means, GMM ou Greedy ;
7. **Optimisation Automatique 2D** ;
8. **Optimisation Automatique 3D**.

### Workflow typique

1. Charger le plan PNG ;
2. Saisir les dimensions réelles, la fréquence, la puissance d'émission ;
3. L'application extrait les murs ;
4. Calculer le pathloss ou générer les cartes de chaleur ;
5. Lancer l'optimisation (ou l'optimisation automatique) pour obtenir les positions des AP.

## 🔬 Modèles et algorithmes

### Modèles de prédiction ML

Les métriques ci-dessous proviennent des notebooks et des fichiers de modèle du dépôt. Elles sont mesurées sur des **jeux de test issus de données générées** (voir « Données » plus bas).

| | Modèle 2D | Modèle 3D |
|---|---|---|
| **Type** | XGBoost (`XGBRegressor`) | XGBoost (`XGBRegressor`) |
| **Variables d'entrée** | `num_walls`, `distance`, `frequency` | `distance`, `numwall`, `etage`, `frequence` |
| **Fichier du modèle** | `pathloss_predictor.pkl` | `xgboost_radio_propagation_model.pkl` |
| **Taille du jeu de données** | 5 000 échantillons (4 000 entraînement / 1 000 test) | 200 000 échantillons (160 000 entraînement / 40 000 test) |
| **R² (test)** | 0,8515 | 0,8662 |
| **MAE (test)** | 5,92 dB | 8,99 dB |
| **RMSE (test)** | ≈ 7,5 dB | 16,08 dB |
| **Validation croisée (5 plis)** | R² = 0,847 ± 0,005 | RMSE moyen = 15,93 ± 0,22 dB |
| **Recherche d'hyperparamètres** | `GridSearchCV` : 2 916 combinaisons × 5 plis | `GridSearchCV` |
| **Hyperparamètres retenus** | profondeur 3, 100 arbres, taux d'apprentissage 0,1, `subsample` 0,9, `colsample_bytree` 0,8, `reg_lambda` 2 | profondeur 5, 100 arbres, taux d'apprentissage 0,1, `colsample_bytree` 0,8, `reg_lambda` 1,5 |

**Importance des variables (modèle 3D)** : étage 61 %, murs 32 %, distance 6 %, fréquence 0,6 %.

**Référence linéaire (3D)** : une régression linéaire (`model_3d/model regression/`) atteint R² = 0,829 et RMSE = 18,16 dB. Elle sert de référence interprétable (coefficients lisibles) et n'est pas utilisée par l'application.

### Données

- **2D** : jeu de 5 000 échantillons généré dans le notebook `model_2d/model+data2d.ipynb` avec une formule de propagation inspirée de l'ITU-R P.1238 et un bruit de type shadowing (valeurs limitées à 30–150 dB ; fréquences 900, 1 800, 2 400, 5 000 et 5 800 MHz).
- **3D** : fichier `indoor_pathloss_dataset.csv` (200 000 lignes, colonnes `distance`, `numwall`, `etage`, `frequence`, `pathloss`), fréquences 2 400, 2 450, 5 000, 5 200, 5 500 et 5 800 MHz. Jeu de données généré, non issu de mesures sur le terrain.

L'interface propose les fréquences 2 400 MHz et 5 000 MHz.

### Formules théoriques (repli)

**Friis, espace libre**
```
PL_libre = 20 × log₁₀(d) + 20 × log₁₀(f) + 32,45
```

**Avec obstacles**
```
PL_total = PL_libre + N_murs × A_mur + N_étages × A_étage
```

où `d` est la distance en km, `f` la fréquence en MHz, `A_mur` l'atténuation par mur (6 dB par défaut) et `A_étage` l'atténuation par étage (15 dB par défaut).

### Algorithmes d'optimisation

Le signal reçu en un point est `P_reçue = P_émission − pathloss` (20 dBm d'émission par défaut). Un point est **couvert** si `P_reçue` dépasse le **seuil de signal minimal** (en dBm, réglable). L'objectif est d'atteindre un pourcentage de couverture minimal, avec un nombre maximal d'AP.

#### 1. Greedy (glouton)
1. Générer des positions candidates (zones denses, grille stratégique, centre de chaque étage) et écarter celles situées dans un mur ;
2. À chaque tour, évaluer chaque candidat avec les AP déjà placés et retenir celui qui augmente le plus la couverture ;
3. Écarter les candidats trop proches de l'AP choisi ;
4. S'arrêter quand l'objectif est atteint, que le nombre maximal d'AP est atteint, ou que le gain devient négligeable.

Le score pénalise le nombre d'AP. Le glouton ne garantit pas l'optimum, mais donne une bonne approximation à un coût raisonnable.

#### 2. GMM + EM
Un mélange gaussien est ajusté sur les points à couvrir ; les moyennes des composantes deviennent les positions des AP. Plusieurs nombres de composantes sont essayés jusqu'à atteindre la couverture visée. Contrairement au K-means, l'affectation est probabiliste et les composantes peuvent être allongées.

#### 3. K-means
Les points à couvrir sont regroupés en *k* groupes (initialisation k-means++) ; un AP est placé au centre de chaque groupe.

#### 4. Optimisation automatique
Estime le nombre d'AP à partir de l'étalement géométrique des récepteurs, puis place les AP par K-means. Ce n'est pas une descente de gradient.

> Les centres obtenus par K-means ou GMM sont purement géométriques : ils ne tiennent pas compte de l'atténuation réelle. C'est pourquoi ils sont ajustés (éloignés des murs) puis évalués avec le modèle de pathloss.

## 📊 Qualité du signal

Seuils appliqués au pathloss pour la classification des zones (modifiables dans l'interface) :

| Classe | Pathloss |
|---|---|
| Excellent | ≤ 50 dB |
| Bon | 50 – 70 dB |
| Faible | 70 – 90 dB |
| Mauvais | > 90 dB |


## 🛠️ Structure du projet

```
Pathloss_Prediction_Indoor/
├── app.py                              # Application Streamlit (8 onglets)
├── image_processor.py                  # Extraction des murs depuis le plan
├── pathloss_calculator.py              # Calcul de pathloss 2D
├── pathloss_calculator_3d.py           # Calcul de pathloss 3D
├── ml_pathloss_predictor_2d.py         # Prédicteur ML 2D
├── ml_pathloss_predictor_3d.py         # Prédicteur ML 3D
├── visualization.py                    # Visualisations 2D
├── visualization_3d.py                 # Visualisations 3D
├── heatmap_generator.py                # Cartes de chaleur 2D
├── heatmap_generator_3d.py             # Cartes de chaleur 3D
├── access_point_optimizer.py           # Optimiseur K-means (2D / 3D)
├── access_point_optimizer_2d_fixed.py  # Variante 2D
├── greedy_optimizer.py                 # Optimiseur glouton 2D
├── greedy_optimizer_3d.py              # Optimiseur glouton 3D
├── gmm_optimizer.py                    # Optimiseur GMM 2D
├── gmm_optimizer_3d.py                 # Optimiseur GMM 3D
├── auto_optimizer_2d.py                # Optimisation automatique 2D
├── auto_optimizer_3d.py                # Optimisation automatique 3D
├── auto_optimization_interface.py      # Interface de l'optimisation auto 2D
├── auto_optimization_interface_3d.py   # Interface de l'optimisation auto 3D
├── pathloss_predictor.pkl              # Modèle 2D chargé par l'application
├── xgboost_radio_propagation_model.pkl # Modèle 3D chargé par l'application
├── model_2d/                           # Entraînement du modèle 2D
│   ├── model+data2d.ipynb              #   génération des données + XGBoost
│   ├── 2d_enriched.ipynb               #   essai exploratoire (données enrichies)
│   └── pathloss_predictor.pkl
├── model_3d/
│   ├── model xgboost/                  # Entraînement du modèle 3D
│   │   ├── model_xgboost.ipynb
│   │   ├── indoor_pathloss_dataset.csv
│   │   ├── model_summary.txt
│   │   ├── README_model_info.md
│   │   └── xgboost_radio_propagation_model.pkl
│   └── model regression/               # Référence : régression linéaire
│       ├── regression_pathloss.ipynb
│       ├── indoor_pathloss_dataset.csv
│       └── linear_regression_model.pkl
├── test-model/                         # Notebook et copies de modèles pour tests
├── plan2d/                             # Plan d'exemple (Home 2d.png)
├── rapport_projet_pathloss.tex         # Rapport LaTeX
├── styles.css                          # Style de l'interface
├── requirements.txt
└── README.md
```

## 🔁 Reproduire l'entraînement

1. **Modèle 2D** : ouvrir `model_2d/model+data2d.ipynb` (génération du jeu de 5 000 échantillons, `GridSearchCV`, évaluation, export du `.pkl`).
2. **Modèle 3D** : ouvrir `model_3d/model xgboost/model_xgboost.ipynb` (chargement du CSV de 200 000 lignes, séparation 80/20, recherche d'hyperparamètres, validation croisée, export du `.pkl`).
3. **Référence linéaire** : `model_3d/model regression/regression_pathloss.ipynb`.
4. Copier les fichiers `.pkl` produits à la racine du projet : l'application les y cherche en premier.

## 🧭 Cas d'usage

- Appartements et bureaux : planification Wi-Fi domestique ou professionnelle ;
- Écoles, hôpitaux : repérage des zones d'ombre ;
- Locaux multi-étages : estimation du nombre d'AP avant déploiement ;
- Simulation pré-installation : comparer plusieurs scénarios sans se déplacer.

## 👤 Auteurs

**Saba Kallel** — [GitHub](https://github.com/kallelsaba) · [LinkedIn](https://www.linkedin.com/in/saba-kallel-4219b6276/)

**Yessine Abdelmaksoud** — [GitHub](https://github.com/yessineabdelmaksoud) · [LinkedIn](https://www.linkedin.com/in/yessine-abdelmaksoud-a63b27270/)
