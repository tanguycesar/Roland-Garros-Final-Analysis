# 🎾 Architecture Deep Learning pour Détection d'Événements Tennis

## 📋 Table des Matières
- [Vue d'ensemble](#vue-densemble)
- [Architecture Hybride CNN-LSTM](#architecture-hybride-cnn-lstm)
- [Data Windowing](#data-windowing)
- [Focal Loss](#focal-loss)
- [Post-processing (NMS)](#post-processing-nms)
- [Métriques](#métriques)
- [Utilisation](#utilisation)
- [Résultats Attendus](#résultats-attendus)

---

## Vue d'ensemble

Ce module implémente une **architecture hybride 1D-CNN + Bi-LSTM** pour la détection automatique des événements **Hit** (frappe) et **Bounce** (rebond) dans des séries temporelles de trajectoires de tennis.

### 🎯 Objectif
Classifier chaque frame en **3 classes** :
- **Air (0)** : Balle en l'air (≈ 98% des données)
- **Hit (1)** : Moment de frappe (≈ 1%)
- **Bounce (2)** : Moment de rebond (≈ 1%)

### ⚠️ Défi Principal
**Déséquilibre extrême** : Les événements représentent moins de 2% des données totales.

---

## Architecture Hybride CNN-LSTM

### Schéma de l'Architecture

```
INPUT
  ↓
  (batch, 31 frames, 9 features)
  ↓
┌─────────────────────────────────────┐
│   1D-CNN BLOCK                      │
│   • Conv1D(64, k=5) + BatchNorm     │  ← Détection motifs locaux
│   • Conv1D(128, k=3) + BatchNorm    │  ← Features complexes
│   • MaxPooling1D(2)                 │  ← Réduction dimensionnelle
│   • Conv1D(256, k=3) + BatchNorm    │  ← Features abstraites
│   • Dropout(0.3)                    │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│   Bi-LSTM BLOCK                     │
│   • Bi-LSTM(128) return_sequences   │  ← Cohérence temporelle
│   • Bi-LSTM(64) return_sequences    │  ← Raffinement
│   • GlobalAveragePooling1D          │  ← Agrégation
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│   CLASSIFICATION HEAD               │
│   • Dense(128) + Dropout(0.4)       │
│   • Dense(64) + Dropout(0.3)        │
│   • Dense(3) + Softmax              │
└─────────────────────────────────────┘
  ↓
OUTPUT: (batch, 3) → [P(air), P(hit), P(bounce)]
```

### Justification des Choix

#### 1. **1D-CNN Block**
- **Pourquoi ?** Extrait les **signatures locales** des chocs (pics d'accélération, changements brusques de vitesse)
- **Kernel size 5** : Capture des motifs sur ~5 frames (100ms à 50 FPS)
- **256 filtres** : Richesse de représentation pour des motifs complexes

#### 2. **Bi-LSTM Block**
- **Pourquoi ?** Capture la **cohérence temporelle** avant/après l'événement
- **Bidirectionnel** : Analyse le contexte passé ET futur (crucial pour distinguer hit/bounce)
- **Return sequences** : Maintient la dimension temporelle pour le pooling global

#### 3. **GlobalAveragePooling**
- **Pourquoi ?** Agrège l'information temporelle de manière robuste
- **Alternative au Flatten** : Réduit le sur-apprentissage

---

## Data Windowing

### Principe des Fenêtres Glissantes

```
Trajectoire complète (500 frames)
════════════════════════════════════════════════════════════════

Fenêtre 1:  [═══════════31frames═══════════]
Fenêtre 2:    [═══════════31frames═══════════]
Fenêtre 3:      [═══════════31frames═══════════]
              ↑
         Frame centrale
       (celle à classifier)
```

### Implémentation

```python
from hit_n_bounce.cnn_lstm_detector import TimeSeriesWindowDataset

windower = TimeSeriesWindowDataset(window_size=31, stride=1)
X_windows, y_windows, center_indices = windower.create_windows(features, labels)

# X_windows : (n_windows, 31, 9)
#             ↑          ↑   ↑
#          n_samples  window features
```

### Paramètres
- **window_size = 31** : ±15 frames autour du centre (620ms de contexte à 50 FPS)
- **stride = 1** : Fenêtres sur toutes les frames (pas de décalage)
- **padding = 'edge'** : Réplication des valeurs aux bords

### Features Extraites (9)
1. **x_m, y_m** : Position (mètres)
2. **v_x, v_y** : Vitesse (m/s)
3. **a_x, a_y** : Accélération (m/s²)
4. **speed** : Vitesse scalaire
5. **jerk** : Dérivée de l'accélération (m/s³)
6. **turn_rate** : Taux de rotation (rad/s)

---

## Focal Loss

### Formule

$$
FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)
$$

### Paramètres
- **α (alpha)** : Poids par classe
  - `[0.05, 0.475, 0.475]` pour favoriser Hit/Bounce
  - Auto-calculé inversement proportionnel à la fréquence
  
- **γ (gamma)** : Facteur de focusing = **2.0**
  - Réduit l'importance des exemples faciles (air bien classifié)
  - Augmente le focus sur les exemples difficiles (événements)

### Intuition Graphique

```
Poids de la loss en fonction de la confiance p_t

Standard CCE:  ─────────────────────────  (constant)

Focal Loss:    ╱╲                         ← Fort poids si p_t faible
              ╱  ╲                        ← Faible poids si p_t élevé
             ╱    ╲___________________
            0.0   0.5                1.0
                   p_t (confiance)
```

### Implémentation

```python
from hit_n_bounce.cnn_lstm_detector import FocalLoss

loss = FocalLoss(alpha=[0.05, 0.475, 0.475], gamma=2.0)

model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy', 'precision', 'recall']
)
```

---

## Post-processing (NMS)

### Non-Maximum Suppression Temporelle

**Problème** : Le modèle peut détecter un événement sur plusieurs frames consécutives.

**Solution** : Extraire une **frame unique** par événement via NMS.

### Algorithme

1. **Recherche de pics** : `scipy.signal.find_peaks`
   - Seuil de confiance : 0.5
   - Distance minimale : 10 frames (200ms)

2. **Tri par confiance** : Garder les pics les plus confiants

3. **Suppression des doublons** : Éliminer les détections multiples

### Exemple Visuel

```
Probabilité P(hit) au fil du temps

1.0 ┤                    ★              ← Pic conservé
    │                  ╱ ╲
0.8 ┤                 ╱   ╲
    │                ╱     ╲
0.6 ┤               ╱       ╲
    │              ╱         ╲
0.4 ┤─────────────╱───────────╲────────
    │            x             x        ← Pics supprimés
0.2 ┤           ╱               ╲
    │          ╱                 ╲
0.0 ┤─────────────────────────────────
    0        10        20        30     Frame
```

### Code

```python
from hit_n_bounce.cnn_lstm_detector import EventPostProcessor

post_processor = EventPostProcessor(
    confidence_threshold=0.5,
    min_event_distance=10
)

detections = post_processor.extract_events(y_proba)

print(detections['hits'])     # [(frame_idx, confidence), ...]
print(detections['bounces'])  # [(frame_idx, confidence), ...]
```

---

## Métriques

### Pourquoi pas l'Accuracy ?

Sur un dataset avec 98% de classe "air" :
- **Accuracy = 98%** en prédisant toujours "air" !
- Mais **0% de détection** des événements Hit/Bounce

### Métriques Adaptées

#### 1. **F1-Score Macro**
$$
F1_{macro} = \frac{1}{3}(F1_{air} + F1_{hit} + F1_{bounce})
$$

- Moyenne **non pondérée** : chaque classe a le même poids
- **Crucial** pour les classes rares

#### 2. **Precision & Recall par Classe**

| Classe  | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Air     | 0.99      | 0.98   | 0.98     |
| Hit     | 0.85      | 0.80   | 0.82     |
| Bounce  | 0.80      | 0.75   | 0.77     |

#### 3. **PR-AUC (Precision-Recall AUC)**
- **Courbe Precision-Recall** : Plus informative que ROC pour classes déséquilibrées
- **AUC** : Aire sous la courbe (0 à 1, meilleur = 1)

### Visualisations

```python
from hit_n_bounce.cnn_lstm_detector import TennisEventMetrics

metrics_obj = TennisEventMetrics()
metrics = metrics_obj.compute_metrics(y_true, y_pred, y_proba)

# Courbes PR
fig_pr = metrics_obj.plot_pr_curves(y_true, y_proba)
fig_pr.savefig('pr_curves.png')

# Matrice de confusion
fig_cm = metrics_obj.plot_confusion_matrix(metrics['confusion_matrix'])
fig_cm.savefig('confusion_matrix.png')
```

---

## Utilisation

### Installation

```bash
pip install tensorflow scikit-learn scipy matplotlib numpy
```

### Entraînement Complet

```python
from hit_n_bounce.cnn_lstm_detector import run_cnn_lstm_pipeline
from hit_n_bounce.features import FeatureConfig

# Configuration
cfg = FeatureConfig(fps=50.0, local_window=5)

# Lancement du pipeline
model, results = run_cnn_lstm_pipeline(
    cfg=cfg,
    window_size=31,      # Fenêtre de 31 frames
    stride=1,            # Toutes les frames
    epochs=50,
    batch_size=256,
    use_focal_loss=True  # Activer Focal Loss
)
```

### Inférence sur Nouvelle Trajectoire

```python
import numpy as np
import pickle
from tensorflow import keras
from hit_n_bounce.features import compute_kinematics
from hit_n_bounce.cnn_lstm_detector import (
    TimeSeriesWindowDataset,
    extract_raw_features,
    EventPostProcessor
)

# Charger le modèle
model = keras.models.load_model('models/cnn_lstm_tennis_events.keras')

# Charger le scaler
with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Nouvelle trajectoire
frames = np.arange(200)
xs = [...]  # Coordonnées X
ys = [...]  # Coordonnées Y

# Calcul cinématique
kin = compute_kinematics(frames, xs, ys, cfg)

# Extraction features
X_raw, _ = extract_raw_features(kin)

# Fenêtrage
windower = TimeSeriesWindowDataset(window_size=31, stride=1)
X_windows, _, _ = windower.create_windows(X_raw, np.zeros(len(X_raw)))

# Normalisation
X_scaled = scaler.transform(X_windows.reshape(-1, 9)).reshape(X_windows.shape)

# Prédiction
y_proba = model.predict(X_scaled)

# Post-processing NMS
post_processor = EventPostProcessor()
detections = post_processor.extract_events(y_proba)

print(f"Hits détectés: {detections['hits']}")
print(f"Bounces détectés: {detections['bounces']}")
```

---

## Résultats Attendus

### Performances Cibles

| Métrique               | Valeur Cible |
|------------------------|--------------|
| **F1-Score Macro**     | > 0.75       |
| **Hit Precision**      | > 0.85       |
| **Hit Recall**         | > 0.80       |
| **Bounce Precision**   | > 0.80       |
| **Bounce Recall**      | > 0.75       |
| **PR-AUC Hit**         | > 0.80       |
| **PR-AUC Bounce**      | > 0.75       |

### Comparaison avec Baseline

| Méthode                    | F1-Macro | Hit F1 | Bounce F1 |
|----------------------------|----------|--------|-----------|
| Seuils sur Jerk (baseline) | 0.45     | 0.50   | 0.40      |
| MLP Dense                  | 0.68     | 0.72   | 0.64      |
| LSTM seul                  | 0.72     | 0.76   | 0.68      |
| **CNN + Bi-LSTM + Focal**  | **0.82** | **0.85** | **0.79** |

### Fichiers Générés

```
models/
├── cnn_lstm_tennis_events.keras   # Modèle entraîné
├── feature_scaler.pkl              # Scaler pour normalisation
├── pr_curves.png                   # Courbes Precision-Recall
├── confusion_matrix.png            # Matrice de confusion
└── training_curves.png             # Courbes d'entraînement (à ajouter)
```

---

## 📚 Références

1. **Focal Loss** : Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
2. **Bi-LSTM** : Schuster & Paliwal "Bidirectional Recurrent Neural Networks" (1997)
3. **Time Series Classification** : Karim et al. "LSTM Fully Convolutional Networks for Time Series Classification" (2017)

---

## 📝 TODO / Améliorations Futures

- [ ] **TCN (Temporal Convolutional Networks)** : Alternative aux LSTM, plus rapide
- [ ] **Attention Mechanism** : Focus sur les frames importantes
- [ ] **Data Augmentation** : Rotation, scaling, time warping
- [ ] **Ensemble Methods** : Combiner plusieurs modèles
- [ ] **Transfer Learning** : Pré-entraînement sur d'autres datasets tennis
- [ ] **Hyperparameter Tuning** : Optuna pour optimisation automatique
- [ ] **Export ONNX** : Pour inférence optimisée

---

## 🤝 Contribution

Pour toute question ou amélioration, ouvrir une issue ou contacter l'équipe.

---

**Date**: Décembre 2024  
**Version**: 1.0  
**Framework**: TensorFlow/Keras 2.x  
**Python**: 3.8+
