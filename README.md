# 🎾 Hit & Bounce Detection — Roland-Garros Final Analysis

Ce projet implémente **trois architectures** pour détecter les **frappes (hit)** et **rebonds (bounce)** de balle de tennis à partir des trajectoires (x,y) :

1. **Non Supervisée** : Heuristiques basées sur la physique (apex, jerk, courbure)
2. **Supervisée ML** : XGBoost / HistGradientBoosting avec features cinématiques avancées
3. **Deep Learning** : Architecture hybride CNN + Bi-LSTM avec gestion du déséquilibre des classes

---

## 📋 Structure du Projet

```
Roland-Garros-Final-Analysis/
│
├── hit_n_bounce/                    # Module principal
│   ├── calibration_distortion.py   # Calibration caméra 21 points + distorsion
│   ├── data_loader.py               # Chargement + nettoyage PCHIP des trajectoires
│   ├── features.py                  # Conversion pixels → mètres + cinématique
│   ├── unsupervised.py              # Détection par analyse de signaux physiques
│   ├── supervised.py                # ML classique (XGBoost/HistGB)
│   ├── cnn_lstm_detector.py         # Deep Learning (CNN-LSTM + Class Weights)
│   └── supervised_dl.py             # Anciennes expérimentations DL
│
├── Data hit & bounce/
│   └── per_point_v2/                # Dataset JSON (313 points)
│       ├── ball_data_1.json
│       ├── ...
│
├── models/                          # Modèles entraînés (non versionnés)
│   ├── tennis_event_classifier.joblib      # XGBoost
│   ├── cnn_lstm_tennis_events.keras        # CNN-LSTM
│   ├── feature_scaler.pkl                  # Normalisation
│   └── best_cnn_lstm_model.keras           # Meilleur checkpoint
│
├── Camera_Params_Distorted.npz      # Paramètres de calibration
├── requirements.txt                 # Dépendances
├── README.md                        # Ce fichier
├── RECAP.md                         # Récapitulatif technique DL
├── CNN_LSTM_ARCHITECTURE.md         # Documentation architecture
├── MATHEMATICAL_FOUNDATION.md       # Fondements mathématiques
└── QUICKSTART.md                    # Guide démarrage rapide
```

---

## 🚀 Installation

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
- **Scientifique** : numpy, scipy, pandas, matplotlib
- **ML Classique** : scikit-learn, xgboost
- **Deep Learning** : tensorflow (2.10-2.15), keras
- **Computer Vision** : opencv-python
- **Autres** : joblib, tqdm, plotly

---

## ⚙️ Configuration Initiale

### 📹 Vidéo (pour calibration uniquement)

Le fichier `calibration_distortion.py` nécessite une vidéo pour extraire une frame de référence.

**Option 1** : Placer `Alcaraz_Sinner_2025-001.mp4` à la racine du projet

**Option 2** : Créer `config.txt` avec le chemin complet :
```txt
C:\chemin\vers\ta\video\Alcaraz_Sinner_2025-001.mp4
```

> **Note** : La vidéo est uniquement pour la calibration initiale. Le fichier `config.txt` est ignoré par Git.

### 🎯 Calibration Caméra (si nécessaire)

Si `Camera_Params_Distorted.npz` n'existe pas :

```bash
python hit_n_bounce/calibration_distortion.py
```

**Instructions** :
1. Cliquer 21 points du terrain dans l'ordre (lignes de fond, service, filet)
2. Les paramètres sont sauvegardés automatiquement
3. Visualisation de la reprojection pour vérifier la précision

---

## 📊 Utilisation des Pipelines

### 1️⃣ Détection Non Supervisée

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

### 2️⃣ ML Supervisé (XGBoost)

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

### 3️⃣ Deep Learning (CNN + Bi-LSTM) ⭐ **NOUVELLE ARCHITECTURE**

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
- ✅ Fenêtres glissantes de 15 frames (au lieu de 31)
- ✅ CrossEntropy + class_weights (au lieu de Focal Loss instable)
- ✅ Architecture simplifiée (1 Bi-LSTM au lieu de 2)
- ✅ Debug automatique des fenêtres (`models/debug_windows.png`)

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

## 📖 Documentation Technique

| Fichier | Contenu |
|---------|---------|
| `RECAP.md` | Récapitulatif complet du projet DL |
| `CNN_LSTM_ARCHITECTURE.md` | Architecture détaillée + justifications |
| `MATHEMATICAL_FOUNDATION.md` | Équations mathématiques (cinématique, CNN, LSTM, métriques) |
| `QUICKSTART.md` | Guide pratique avec exemples de code |

---

## 🧪 Tests & Validation

### Test de l'architecture CNN-LSTM

```bash
python test_cnn_lstm.py
```

Vérifie :
- ✓ Construction du modèle
- ✓ Forward pass
- ✓ Post-processing NMS
- ✓ Calcul des métriques

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

## 🔧 Fichiers de Configuration

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

## 📈 Comparaison des Méthodes

| Méthode | F1-Macro | Temps Entraînement | Interprétabilité | Robustesse |
|---------|----------|-------------------|------------------|------------|
| **Non Supervisée** | ~0.65 | 0s (pas d'entraînement) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **XGBoost** | ~0.82 | ~5 min | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CNN-LSTM** | ~0.82 | ~2-3h (GPU) | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommandation** :
- **Production rapide** : XGBoost (meilleur compromis)
- **Précision maximale** : CNN-LSTM (après tuning)
- **Analyse exploratoire** : Non supervisée

---

## 🐛 Dépannage

### Problème de calibration

**Erreur** : `Vidéo introuvable`
- Vérifier le chemin dans `config.txt`
- Ou placer la vidéo à la racine avec le bon nom

### Problème d'entraînement DL

**Loss très basse + Accuracy faible** :
- ✅ Maintenant corrigé avec CrossEntropy + class_weights
- Vérifier `models/debug_windows.png` pour alignement données/labels

**OOM (Out of Memory)** :
```python
# Réduire batch_size dans cnn_lstm_detector.py
BATCH_SIZE = 128  # au lieu de 256
```

### Problème de features

**NaN dans les calculs** :
- Vérifier que `Camera_Params_Distorted.npz` existe
- Relancer la calibration si nécessaire

---

## 📦 Fichiers à Ne Pas Versionner

Le `.gitignore` exclut :
- `Data hit & bounce/` (dataset lourd)
- `models/*.keras` et `models/*.pkl` (modèles entraînés)
- `*.mp4` (vidéos)
- `.venv/` (environnement virtuel)
- `config.txt` (chemins locaux)
- `__pycache__/` et `*.pyc`

---

## 🎓 Auteur & Contexte

**Projet** : Stage Roland-Garros 2025 - Analyse automatique des frappes et rebonds

**Technologies** :
- Computer Vision (OpenCV)
- Machine Learning (XGBoost, scikit-learn)
- Deep Learning (TensorFlow/Keras)
- Traitement du signal (scipy, PCHIP interpolation)

**Date** : Décembre 2025

---

## 📝 TODO / Améliorations Futures

- [ ] Data augmentation pour le CNN-LSTM (flips, noise)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Ensemble XGBoost + CNN-LSTM
- [ ] API REST pour inférence temps réel
- [ ] Export ONNX pour déploiement optimisé
- [ ] Tests unitaires complets

---

## 🤝 Contribution

Pour contribuer :
1. Fork le repository
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

---

## 📄 License

Ce projet est à usage académique dans le cadre d'un stage Roland-Garros 2025.

---

**⭐ Si ce projet vous a aidé, n'hésitez pas à lui donner une étoile sur GitHub !**
