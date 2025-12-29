# Hit & Bounce Detection - Roland-Garros Final Analysis

Ce projet implémente **deux architectures** pour détecter les **frappes (hit)** et **rebonds (bounce)** de balle de tennis à partir des trajectoires (x,y) :

1. **Non Supervisée** : Heuristiques basées sur la physique (apex, jerk, courbure)
2. **Supervisée ML** : XGBoost / HistGradientBoosting avec features cinématiques avancées

---

## Structure du Projet

```
Roland-Garros-Final-Analysis/
│
├── hit_n_bounce/                    # Module principal
│   ├── __init__.py                  # Package initialization
│   ├── calibration_distortion.py   # Calibration caméra 21 points + distorsion
│   ├── data_loader.py               # Chargement + nettoyage PCHIP des trajectoires
│   ├── features.py                  # Conversion pixels → mètres + cinématique
│   ├── unsupervised.py              # Détection par analyse de signaux physiques
│   └── supervised.py                # ML classique (XGBoost/HistGradientBoosting)
│
├── Data hit & bounce/
│   └── per_point_v2/                # Dataset JSON (313 points)
│       ├── ball_data_1.json
│       ├── ...
│
├── models/                          # Modèles entraînés (non versionnés)
│   └── tennis_event_classifier.joblib      # XGBoost
│
├── main.py                          # Interface ligne de commande
├── Camera_Params_Distorted.npz      # Paramètres de calibration
├── requirements.txt                 # Dépendances
└── README.md                        # Ce fichier
```

---

## Installation

### 1. Créer l'environnement virtuel

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales** :
- **Scientifique** : numpy, scipy, matplotlib
- **ML** : scikit-learn, xgboost, joblib
- **Computer Vision** : opencv-python
- **Notebooks** : jupyter, ipykernel

---

## Configuration Initiale

### Vidéo (pour calibration uniquement)

Le fichier `calibration_distortion.py` nécessite une vidéo pour extraire une frame de référence.

**Option 1** : Placer `Alcaraz_Sinner_2025-001.mp4` à la racine du projet

**Option 2** : Créer `config.txt` avec le chemin complet :
```txt
C:\chemin\vers\ta\video\Alcaraz_Sinner_2025-001.mp4
```

> **Note** : La vidéo est uniquement pour la calibration initiale. Le fichier `config.txt` est ignoré par Git.

### Calibration Caméra (si nécessaire)

Si `Camera_Params_Distorted.npz` n'existe pas :

```bash
python hit_n_bounce/calibration_distortion.py
```

**Instructions** :
1. Cliquer 21 points du terrain dans l'ordre (lignes de fond, service, filet)
2. Les paramètres sont sauvegardés automatiquement
3. Visualisation de la reprojection pour vérifier la précision

---

## Utilisation

### Pipeline de traitement

Le projet suit une architecture modulaire en 4 étapes :

```
data_loader.py → features.py → unsupervised.py / supervised.py
```

1. **data_loader.py** : Nettoyage et interpolation PCHIP des trajectoires
2. **features.py** : Conversion pixels → mètres + calcul cinématique
3. **unsupervised.py** : Détection par heuristiques physiques
4. **supervised.py** : Détection par apprentissage supervisé (XGBoost)

### 1. Détection Non Supervisée

```bash
python hit_n_bounce/unsupervised.py
```

**Principe** :
- Analyse des **apex** (changements de direction Vy)
- Différenciation **joueur haut** (apex = frappe) vs **joueur bas** (rebond + frappe)
- Scoring multi-critères : jerk, courbure, vitesse, turn_rate
- Recherche guidée de rebonds avant chaque hit

**Avantages** : Aucun label nécessaire, interprétable
**Inconvénients** : Seuils à ajuster par terrain/caméra

---

### 2. ML Supervisé (XGBoost)

#### Entraînement

```bash
python hit_n_bounce/supervised.py
```

**Pipeline** :
- Extraction de **28 features** physiques (position, vitesse, accélération, jerk, contexte ±5 frames)
- **GroupKFold** (5 folds) pour éviter le data leakage
- **XGBoost** avec sample_weight (4x hit, 2.5x bounce)
- Sauvegarde : `models/tennis_event_classifier.joblib`

**Performances attendues** :
- F1-Macro : ~0.82
- Hit : Precision ~0.88, Recall ~0.85
- Bounce : Precision ~0.83, Recall ~0.79

#### Prédiction

```python
from joblib import load
from hit_n_bounce.features import FeatureConfig, compute_kinematics

# Charger modèle
payload = load("models/tennis_event_classifier.joblib")
model = payload["model"]

# Prédire sur nouvelles données
# (voir supervised.py pour exemple complet)
```

---

### 3. Deep Learning (CNN + Bi-LSTM) - NOUVELLE ARCHITECTURE

#### Architecture Hybride

```
Input (15 frames, 9 features)
  ↓
[CNN Block - Extraction motifs locaux]
  Conv1D(64, k=5) → BatchNorm → ReLU
  Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
  Dropout(0.3)
  ↓
[Bi-LSTM - Cohérence temporelle]
  Bi-LSTM(128, return_sequences)
  GlobalAveragePooling1D
  ↓
[Classification Head]
  Dense(128) → ReLU → Dropout(0.4)
  Dense(64) → ReLU → Dropout(0.3)
  Dense(3) → Softmax
  ↓
Output: [P(air), P(hit), P(bounce)]
```

#### Entraînement

```bash
python hit_n_bounce/cnn_lstm_detector.py
```

**Hyperparamètres** :
- **Window Size** : 15 frames (±7 contexte = 280ms à 50 FPS)
- **Loss** : CrossEntropy + Class Weights (auto-calculés)
- **Epochs** : 50
- **Batch Size** : 256
- **Learning Rate** : 0.001

**Innovations clés** :
- Fenêtres glissantes de 15 frames (au lieu de 31)
- CrossEntropy + class_weights (au lieu de Focal Loss instable)
- Architecture simplifiée (1 Bi-LSTM au lieu de 2)
- Debug automatique des fenêtres (`models/debug_windows.png`)

**Performances cibles** :
- F1-Macro : > 0.75
- Hit : F1 > 0.85
- Bounce : F1 > 0.79

#### Post-Processing NMS

```python
from hit_n_bounce.cnn_lstm_detector import EventPostProcessor

processor = EventPostProcessor(
    confidence_threshold=0.5,
    min_event_distance=10  # frames
)

detections = processor.extract_events(y_proba)
# {'hits': [(frame, confidence), ...], 'bounces': [...]}
```

#### Visualisations Générées

Après entraînement, dans `models/` :
- `pr_curves.png` : Courbes Precision-Recall par classe
- `confusion_matrix.png` : Matrice de confusion
- `debug_windows.png` : Visualisation des fenêtres d'entraînement
- `architecture_visualization.png` : Diagramme du modèle
- `focal_loss_visualization.png` : Courbes de loss

---

## Documentation Technique

| Fichier | Contenu |
|---------|---------|
| `RECAP.md` | Récapitulatif complet du projet DL |
| `CNN_LSTM_ARCHITECTURE.md` | Architecture détaillée + justifications |
| `MATHEMATICAL_FOUNDATION.md` | Équations mathématiques (cinématique, CNN, LSTM, métriques) |
| `QUICKSTART.md` | Guide pratique avec exemples de code |

---

## Tests & Validation

### Test de l'architecture CNN-LSTM

```bash
python test_cnn_lstm.py
```

Vérifie :
- Construction du modèle
- Forward pass
- Post-processing NMS
- Calcul des métriques

### Visualisation de l'architecture

```bash
python visualize_architecture.py
```

Génère des graphiques comparatifs et diagrammes.

### Dashboard physique

```bash
python hit_n_bounce/features.py
```

Affiche une trajectoire exemple avec toutes les features cinématiques calculées.

---

## Fichiers de Configuration

### `FeatureConfig` (features.py)

```python
@dataclass
class FeatureConfig:
    fps: float = 50.0          # Framerate de la vidéo
    local_window: int = 5      # Fenêtre contexte pour dérivées
```

### Chemins de données

Les chemins sont **relatifs** pour portabilité :
```python
# supervised.py, cnn_lstm_detector.py
data_folder = Path("Data hit & bounce") / "per_point_v2"
```

---

## Comparaison des Méthodes

| Méthode | F1-Macro | Temps Entraînement | Interprétabilité | Robustesse |
|---------|----------|-------------------|------------------|------------|
| **Non Supervisée** | ~0.65 | 0s (pas d'entraînement) | 5/5 | 3/5 |
| **XGBoost** | ~0.82 | ~5 min | 4/5 | 4/5 |

**Recommandation** :
- **Production** : XGBoost (meilleur compromis performance/rapidité)
- **Analyse exploratoire** : Non supervisée (sans labels)

---

## Dépannage

### Problème de calibration

**Erreur** : `Vidéo introuvable`
- Vérifier le chemin dans `config.txt`
- Ou placer la vidéo à la racine avec le bon nom

### Problème de features

**NaN dans les calculs** :
- Vérifier que `Camera_Params_Distorted.npz` existe
- Relancer la calibration si nécessaire

---

## Fichiers à Ne Pas Versionner

Le `.gitignore` exclut :
- `Data hit & bounce/` (dataset lourd)
- `models/*.joblib` (modèles entraînés)
- `*.mp4` (vidéos)
- `*.npz` (paramètres de calibration)
- `.venv/` (environnement virtuel)
- `config.txt` (chemins locaux)
- `hit_n_bounce/cnn_lstm_detector.py` (expérimental, local uniquement)
- `__pycache__/` et `*.pyc`

---

## Auteur & Contexte

**Projet** : Stage Roland-Garros 2025 - Analyse automatique des frappes et rebonds

**Technologies** :
- Computer Vision (OpenCV)
- Machine Learning (XGBoost, scikit-learn)
- Deep Learning (TensorFlow/Keras)
- Traitement du signal (scipy, PCHIP interpolation)

**Date** : Décembre 2025

---

## TODO / Améliorations Futures

- [ ] Optimisation des hyperparamètres XGBoost
- [ ] Ensemble de modèles pour améliorer la robustesse
- [ ] Interface graphique pour visualisation interactive
- [ ] Export des prédictions en format CSV
- [ ] Tests unitaires complets

---

## Contribution

**Ce projet n'accepte pas de contributions externes.** Il s'agit d'un projet académique personnel développé dans le cadre d'un stage à Roland-Garros 2025.

Pour toute question ou suggestion, vous pouvez ouvrir une issue pour discussion uniquement.

---

## License

Ce projet est à usage académique dans le cadre d'un stage Roland-Garros 2025. Tous droits réservés.
