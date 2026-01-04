# Detection automatique des frappes et rebonds au tennis

**Auteur** : Tanguy CESAR  
**Contexte** : Projet technique - Stage Roland-Garros 2025  
**Technologies** : Python, Machine Learning, Computer Vision

---
Analyse automatisée des événements du match final de Roland-Garros 2025 entre Alcaraz et Sinner, avec deux approches (non supervisée et supervisée) pour détecter frappes et rebonds à partir des trajectoires de balle.
---

## Objectif

Ce projet fournit deux fonctions pour détecter automatiquement les événements clés lors d'un match de tennis (frappes et rebonds) à partir des coordonnées 2D de la trajectoire de la balle.

Les deux fonctions principales sont :

1. **`unsupervised_hit_bounce_detection(json_path)`** : Détection basée sur l'analyse physique des signaux (jerk, courbure, apex de trajectoire)
2. **`supervised_hit_bounce_detection(json_path, model_path)`** : Classification par apprentissage automatique (XGBoost) avec features cinématiques

Les deux fonctions prennent un fichier JSON en entrée et retournent un JSON enrichi avec les événements détectés.

---

## Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. Cloner le dépôt
```bash
git clone https://github.com/tanguycesar/Roland-Garros-Final-Analysis.git
cd Roland-Garros-Final-Analysis
```

2. Créer un environnement virtuel
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Format d'entrée

Les fichiers JSON d'entrée doivent suivre ce format :
```json
{
  "32352": {
    "x": 1024.5,
    "y": 768.2,
    "visible": true,
    "action": "air"
  },
  "32353": {
    "x": 1028.1,
    "y": 770.8,
    "visible": true,
    "action": "air"
  }
}
```

Où :
- **Clé** : numéro de frame (int)
- **x, y** : coordonnées pixels de la balle (float ou null)
- **visible** : visibilité de la balle (bool)
- **action** : action détectée ("air", "hit", "bounce")

### Format de sortie

Les fonctions retournent un JSON enrichi avec le même format, mais avec le champ `"action"` mis à jour :
- `"hit"` : frappe détectée
- `"bounce"` : rebond détecté  
- `"air"` : balle en l'air

### Usage en ligne de commande

```bash
# Méthode non supervisée
python main.py "Data hit & bounce/per_point_v2/ball_data_1.json" --method unsupervised

# Méthode supervisée avec sauvegarde
python main.py "Data hit & bounce/per_point_v2/ball_data_1.json" \
  --method supervised \
  --model models/tennis_event_classifier.joblib \
  --output enriched_output.json
```

### Usage programmatique

```python
from main import unsupervised_hit_bounce_detection, supervised_hit_bounce_detection

# Méthode non supervisée
result_unsup = unsupervised_hit_bounce_detection("ball_data_1.json")

# Méthode supervisée
result_sup = supervised_hit_bounce_detection(
    "ball_data_1.json",
    model_path="models/tennis_event_classifier.joblib"
)

# Accéder aux résultats
print(result_sup["32500"]["action"])  # "hit", "bounce" ou "air"

# Compter les événements
hits = sum(1 for v in result_sup.values() if v.get("action") == "hit")
bounces = sum(1 for v in result_sup.values() if v.get("action") == "bounce")
print(f"Détections : {hits} frappes, {bounces} rebonds")
```

---

## Architecture du projet

```
Roland-Garros-Final-Analysis/
│
├── main.py                          # Deux fonctions principales
├── requirements.txt                 # Dépendances Python
├── models/
│   └── tennis_event_classifier.joblib  # Modèle XGBoost pré-entraîné
│
├── hit_n_bounce/                    # Modules internes
│   ├── data_loader.py               # Nettoyage et interpolation PCHIP
│   ├── features.py                  # Extraction features cinématiques
│   ├── unsupervised.py              # Détection heuristique
│   ├── supervised.py                # Classification ML
│   └── calibration_distortion.py   # Calibration caméra
│
└── Data hit & bounce/               # Dataset annoté (313 points)
        ball_data_1.json
        ball_data_10.json
        ...
```

---

## Pipeline de traitement

Le pipeline suit 4 étapes modulaires :

```
Données JSON → data_loader → features → unsupervised/supervised → JSON enrichi
```

### 1. data_loader.py
- Chargement des fichiers JSON de trajectoire
- Segmentation des rallyes (détection des pauses de service)
- Interpolation PCHIP pour combler les données manquantes
- Filtrage Savitzky-Golay pour lisser le bruit

### 2. features.py
- Calibration caméra (conversion pixels → mètres terrain)
- Calcul des features cinématiques :
  - Vitesses (vx, vy, speed)
  - Accélérations (ax, ay, accel)
  - Jerk (dérivée 3e ordre)
  - Turn rate (courbure de trajectoire)

### 3. Détection des événements

#### 3a. unsupervised.py (Méthode non supervisée)
**Principe** : Les frappes et rebonds se caractérisent par des changements brusques de direction (apex).

**Algorithme** :
1. Détection des pivots (inversion de Vy)
2. Scoring multi-critères : jerk, courbure, vitesse, turn_rate
3. Distinction joueur haut/bas selon position Y
4. Recherche guidée de rebonds avant chaque frappe

**Avantages** :
- Aucune annotation nécessaire
- Interprétable physiquement
- Rapide (pas d'entraînement)

**Limites** :
- Sensible aux paramètres (seuils à ajuster)
- Performances modestes (F1 ~0.65)

#### 3b. supervised.py (Méthode supervisée)
**Principe** : Apprentissage supervisé sur dataset annoté.

**Pipeline ML** :
1. Extraction de 28 features physiques par frame
2. Fenêtre contextuelle de ±5 frames (11 frames totales)
3. Classification XGBoost (3 classes : air, hit, bounce)
4. Post-processing : suppression pics parasites, cooldown temporel

**Configuration** :
- Validation croisée : GroupKFold 5 folds (pas de leakage entre points)
- Class weighting : 4x hit, 2.5x bounce (données déséquilibrées)
- Hyperparamètres : 800 estimators, max_depth 8, learning_rate 0.03

**Performances** :
- F1-Macro : 0.82
- Hit : Precision 0.88 / Recall 0.85
- Bounce : Precision 0.83 / Recall 0.79

---

## Comparaison des méthodes

| Critère              | Non supervisée | Supervisée (XGBoost) |
|----------------------|----------------|----------------------|
| F1-Macro             | 0.65           | 0.82                 |
| Temps entraînement   | 0s             | 5 min                |
| Annotations requises | Non            | Oui                  |
| Interprétabilité     | Excellente     | Bonne                |
| Robustesse           | Moyenne        | Élevée               |

**Recommandation** : Méthode supervisée pour la production (meilleur compromis précision/complexité).

---

## Contenu du dépôt

Ce dépôt GitHub contient tous les fichiers nécessaires pour exécuter la solution :

- **main.py** : Les deux fonctions principales
  - `unsupervised_hit_bounce_detection(json_path)`
  - `supervised_hit_bounce_detection(json_path, model_path)`
  
- **requirements.txt** : Liste complète des dépendances Python

- **models/tennis_event_classifier.joblib** : Modèle XGBoost pré-entraîné

- **hit_n_bounce/** : Package Python contenant les modules
  - `data_loader.py` : Nettoyage et prétraitement
  - `features.py` : Extraction de features
  - `unsupervised.py` : Détection heuristique
  - `supervised.py` : Classification ML
  - `calibration_distortion.py` : Calibration caméra

- **Camera_Params_Distorted.npz** : Paramètres de calibration caméra

---

## Dépannage

### Erreur "Model not found"
Vérifier que le fichier `models/tennis_event_classifier.joblib` existe. Il est inclus dans le dépôt.

### Erreur "Camera_Params_Distorted.npz not found"
Le fichier de calibration doit être présent à la racine. Il est inclus dans le dépôt.

### Performances insuffisantes
- Vérifier la qualité des données d'entrée (trajectoires bruitées)
- S'assurer que les coordonnées x, y sont en pixels
- Vérifier que le framerate est bien à 50 fps

---

## Performances du système

Tests sur le dataset de 313 points annotés :

**Méthode non supervisée** :
- Temps d'exécution : ~0.5s par point
- F1 Hit : 0.68
- F1 Bounce : 0.62
- F1 Macro : 0.65

**Méthode supervisée (XGBoost)** :
- Temps d'exécution : ~0.8s par point
- F1 Hit : 0.86
- F1 Bounce : 0.81
- F1 Macro : 0.82

---

## Licence

Projet académique réalisé dans le cadre d'un stage. 
