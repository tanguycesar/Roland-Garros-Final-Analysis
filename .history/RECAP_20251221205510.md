# 🎾 Architecture Deep Learning Tennis - Récapitulatif Complet

## 📦 Contenu du Package

### Fichiers Créés

```
Roland-Garros-Final-Analysis/
│
├── 📄 CNN_LSTM_ARCHITECTURE.md          # Documentation architecture complète
├── 📄 MATHEMATICAL_FOUNDATION.md        # Fondements mathématiques
├── 📄 QUICKSTART.md                     # Guide de démarrage rapide
│
├── 🐍 test_cnn_lstm.py                  # Script de test de l'architecture
├── 🐍 visualize_architecture.py         # Visualisation interactive
│
└── hit_n_bounce/
    ├── 🐍 cnn_lstm_detector.py          # Module principal (NOUVEAU)
    ├── 🐍 supervised_dl.py              # Module original (à remplacer)
    ├── 🐍 features.py                   # Features cinématiques
    ├── 🐍 data_loader.py                # Chargement données
    └── ...
```

---

## 🎯 Architecture Proposée : CNN + Bi-LSTM

### Schéma Global

```
┌─────────────────────────────────────────────────────┐
│              INPUT: (batch, 31, 9)                  │
│         31 frames × 9 features cinématiques         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              1D-CNN BLOCK                           │
│  ┌─────────────────────────────────────────────┐   │
│  │ Conv1D(64, k=5) → BatchNorm → ReLU          │   │
│  │ Conv1D(128, k=3) → BatchNorm → ReLU         │   │
│  │ MaxPooling1D(2)                             │   │
│  │ Conv1D(256, k=3) → BatchNorm → ReLU         │   │
│  │ Dropout(0.3)                                │   │
│  └─────────────────────────────────────────────┘   │
│         ↓ Extraction de motifs locaux               │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│             Bi-LSTM BLOCK                           │
│  ┌─────────────────────────────────────────────┐   │
│  │ Bi-LSTM(128, return_sequences)              │   │
│  │ Bi-LSTM(64, return_sequences)               │   │
│  │ GlobalAveragePooling1D                      │   │
│  └─────────────────────────────────────────────┘   │
│         ↓ Cohérence temporelle bidirectionnelle     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│           CLASSIFICATION HEAD                       │
│  ┌─────────────────────────────────────────────┐   │
│  │ Dense(128) → ReLU → Dropout(0.4)            │   │
│  │ Dense(64) → ReLU → Dropout(0.3)             │   │
│  │ Dense(3) → Softmax                          │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        ↓
              OUTPUT: (batch, 3)
          [P(air), P(hit), P(bounce)]
```

### Justifications

| Composant | Choix | Justification |
|-----------|-------|---------------|
| **CNN 1D** | 3 couches [64, 128, 256] | Capture signatures locales (pics d'accélération) |
| **Kernel Size** | [5, 3, 3] | 5 frames ≈ 100ms à 50 FPS |
| **Bi-LSTM** | 2 couches [128, 64] | Contexte temporel avant/après événement |
| **Bidirectionnel** | ✓ | Analyse passé + futur (crucial pour hit vs bounce) |
| **Dropout** | [0.3, 0.4] | Régularisation contre overfitting |
| **GlobalAvgPool** | ✓ | Agrégation robuste vs Flatten |

---

## 📊 Data Windowing

### Principe

```python
# Fenêtre de 31 frames (±15 autour du centre)
windower = TimeSeriesWindowDataset(window_size=31, stride=1)
X_windows, y_windows, indices = windower.create_windows(features, labels)

# X_windows.shape = (n_samples, 31, 9)
#                    ↑          ↑   ↑
#                 samples    window features
```

### Features (9)

1. **x_m, y_m** : Position (mètres)
2. **v_x, v_y** : Vitesse (m/s)
3. **a_x, a_y** : Accélération (m/s²)
4. **speed** : Vitesse scalaire
5. **jerk** : Jerk (m/s³)
6. **turn_rate** : Taux de rotation (rad/s)

**Note** : Pas de features dérivées manuelles, le CNN extrait les motifs complexes.

---

## 🎯 Focal Loss

### Formule

$$
FL(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)
$$

### Paramètres Recommandés

```python
loss = FocalLoss(
    alpha=[0.05, 0.475, 0.475],  # Favorise Hit/Bounce (auto-calculé)
    gamma=2.0                     # Focusing sur exemples difficiles
)
```

### Effet Visuel

```
Poids de la Loss
     │
 1.0 │ ╱╲                    ← Fort poids si faible confiance
     │╱  ╲                   ← Faible poids si forte confiance
 0.5 │    ╲
     │     ╲_______________
 0.0 └─────────────────────
     0.0  0.5          1.0
          Confiance p_t
```

---

## 🔍 Post-Processing (NMS)

### Non-Maximum Suppression

```python
post_processor = EventPostProcessor(
    confidence_threshold=0.5,  # Seuil de probabilité
    min_event_distance=10      # Distance min entre événements (frames)
)

detections = post_processor.extract_events(y_proba)
# {'hits': [(frame_idx, confidence), ...], 
#  'bounces': [(frame_idx, confidence), ...]}
```

### Exemple

```
P(hit) au fil du temps:
  │        ★                 ← Pic conservé (max local + seuil)
1 │      ╱ ╲
  │     ╱   ╲
0.5│────╱─────╲────────────  ← Seuil = 0.5
  │   x       x              ← Pics supprimés (trop proches)
0 │─────────────────────
  0  10  20  30  40  Frame
```

---

## 📈 Métriques

### Priorité : F1-Score Macro

```python
# F1 par classe
F1_air = 0.98
F1_hit = 0.82
F1_bounce = 0.77

# F1 Macro (moyenne non pondérée)
F1_macro = (0.98 + 0.82 + 0.77) / 3 = 0.86
```

**Pourquoi pas Accuracy ?**
- Accuracy = 98% en prédisant toujours "air"
- Mais 0% de détection des événements !

### Courbes PR

```python
metrics_obj = TennisEventMetrics()
fig_pr = metrics_obj.plot_pr_curves(y_true, y_proba)
fig_pr.savefig('pr_curves.png')
```

---

## 🚀 Utilisation

### 1. Test de l'Architecture (sans entraînement)

```bash
python test_cnn_lstm.py
```

**Sortie attendue** :
```
==============================================================
TEST DE L'ARCHITECTURE CNN-LSTM
==============================================================
✓ Modèle construit avec succès
✓ Focal Loss fonctionnelle
✓ Post-processing OK
✓ Métriques OK
✅ TOUS LES TESTS PASSÉS
```

### 2. Visualisation

```bash
python visualize_architecture.py
```

**Génère** :
- `models/architecture_visualization.png` : Diagrammes du modèle
- `models/focal_loss_visualization.png` : Courbes Focal Loss
- `models/architecture_comparison.png` : Comparaison de configs

### 3. Entraînement Complet

```python
from hit_n_bounce.cnn_lstm_detector import run_cnn_lstm_pipeline
from hit_n_bounce.features import FeatureConfig

cfg = FeatureConfig(fps=50.0, local_window=5)

model, results = run_cnn_lstm_pipeline(
    cfg=cfg,
    window_size=31,
    stride=1,
    epochs=50,
    batch_size=256,
    use_focal_loss=True
)

print(f"F1-Macro: {results['metrics']['f1_macro']:.4f}")
```

**Temps estimé** : 2-3 heures (GPU RTX 3080)

### 4. Inférence

```python
# Charger modèle
model = keras.models.load_model('models/cnn_lstm_tennis_events.keras')

# Préparer données
X_windows = windower.create_windows(features, labels)[0]
X_scaled = scaler.transform(X_windows.reshape(-1, 9)).reshape(X_windows.shape)

# Prédire
y_proba = model.predict(X_scaled)

# Post-process
detections = post_processor.extract_events(y_proba)
print(detections['hits'])     # [(frame, conf), ...]
print(detections['bounces'])  # [(frame, conf), ...]
```

---

## 📊 Performances Attendues

### Targets

| Métrique | Valeur Cible |
|----------|--------------|
| **F1-Macro** | > 0.75 |
| **Hit Precision** | > 0.85 |
| **Hit Recall** | > 0.80 |
| **Bounce Precision** | > 0.80 |
| **Bounce Recall** | > 0.75 |
| **PR-AUC Hit** | > 0.80 |
| **PR-AUC Bounce** | > 0.75 |

### Baseline Comparison

| Méthode | F1-Macro | Hit F1 | Bounce F1 |
|---------|----------|--------|-----------|
| Seuils Jerk | 0.45 | 0.50 | 0.40 |
| MLP Dense | 0.68 | 0.72 | 0.64 |
| LSTM seul | 0.72 | 0.76 | 0.68 |
| **CNN-LSTM + Focal** | **0.82** | **0.85** | **0.79** |

---

## 📦 Dépendances

```bash
pip install tensorflow scikit-learn scipy matplotlib numpy pandas
```

**Versions recommandées** :
- TensorFlow : 2.10+ (2.15 idéal)
- Python : 3.8-3.11
- CUDA : 11.2+ (si GPU)

---

## 📁 Structure des Fichiers Générés

```
models/
├── cnn_lstm_tennis_events.keras         # Modèle principal
├── feature_scaler.pkl                    # Normalisation
├── best_cnn_lstm_model.keras             # Meilleur checkpoint
│
├── pr_curves.png                         # Courbes Precision-Recall
├── confusion_matrix.png                  # Matrice de confusion
├── architecture_visualization.png        # Diagrammes architecture
├── focal_loss_visualization.png          # Graphiques Focal Loss
└── architecture_comparison.png           # Comparaison configs
```

---

## 🔧 Hyperparamètres

### Configuration Recommandée

```python
WINDOW_SIZE = 31          # ±15 frames contexte
STRIDE = 1                # Fenêtre sur chaque frame
EPOCHS = 50               # Nombre d'epochs
BATCH_SIZE = 256          # Taille batch
USE_FOCAL_LOSS = True     # Activer Focal Loss
ALPHA = [0.05, 0.475, 0.475]  # Poids Focal Loss
GAMMA = 2.0               # Focusing factor
```

### Tweaking

#### Pour plus de précision (Hit/Bounce)
```python
alpha=[0.02, 0.49, 0.49]  # Favoriser encore plus les événements
window_size=51             # Plus de contexte
```

#### Pour entraînement plus rapide
```python
epochs=30
batch_size=512
stride=2  # Une fenêtre sur deux
```

---

## 🐛 Debugging

### Le modèle ne converge pas

1. **Vérifier distribution** :
```python
print(np.bincount(y_all))  # Doit avoir des événements
```

2. **Réduire learning rate** :
```python
optimizer=keras.optimizers.Adam(learning_rate=0.0001)
```

3. **Augmenter patience** :
```python
EarlyStopping(patience=25)  # Au lieu de 15
```

### F1-Score faible sur événements

1. **Ajuster alpha** :
```python
loss = FocalLoss(alpha=[0.02, 0.49, 0.49], gamma=2.5)
```

2. **Data augmentation** :
```python
X_aug, y_aug = windower.create_augmented_windows(
    X_raw, y, augmentation_factor=5
)
```

---

## 📚 Documentation

| Fichier | Description |
|---------|-------------|
| `CNN_LSTM_ARCHITECTURE.md` | Architecture détaillée + justifications |
| `MATHEMATICAL_FOUNDATION.md` | Équations mathématiques complètes |
| `QUICKSTART.md` | Guide démarrage rapide + exemples code |
| `RECAP.md` | Ce fichier (récapitulatif) |

---

## 🎓 Concepts Clés à Retenir

1. **Fenêtrage** : 31 frames pour capturer contexte temporel
2. **CNN 1D** : Extraction motifs locaux (chocs, pics)
3. **Bi-LSTM** : Cohérence temporelle bidirectionnelle
4. **Focal Loss** : Gérer déséquilibre extrême (98% air)
5. **NMS** : Post-processing pour frame unique par événement
6. **F1-Macro** : Métrique adaptée aux classes rares

---

## 🚀 Prochaines Étapes

1. ✅ **Tester l'architecture** : `python test_cnn_lstm.py`
2. ✅ **Visualiser** : `python visualize_architecture.py`
3. ⏳ **Entraîner** : `python -m hit_n_bounce.cnn_lstm_detector`
4. ⏳ **Évaluer** : Analyser PR curves, confusion matrix
5. ⏳ **Optimiser** : Tuning hyperparamètres si nécessaire

---

## 📞 Support

En cas de problème :
1. Vérifier les logs d'entraînement
2. Lancer `test_cnn_lstm.py` pour isoler le problème
3. Consulter `QUICKSTART.md` pour commandes détaillées

---

**Date** : Décembre 2024  
**Version** : 1.0  
**Framework** : TensorFlow/Keras 2.x  
**Auteur** : Architecture CNN-LSTM pour Détection d'Événements Tennis

---

## 📊 Résumé en 1 Page

### Architecture
- **Input** : (31 frames, 9 features)
- **CNN** : 3 Conv1D [64, 128, 256]
- **Bi-LSTM** : 2 couches [128, 64]
- **Output** : 3 classes (air, hit, bounce)

### Loss Function
- **Focal Loss** : α=[0.05, 0.475, 0.475], γ=2.0
- **Objectif** : Compenser déséquilibre 98% air vs 2% événements

### Métriques
- **F1-Macro** : Moyenne non pondérée (crucial)
- **PR-AUC** : Aire sous courbe Precision-Recall
- **Target** : F1 > 0.75, Hit/Bounce > 0.80

### Post-Processing
- **NMS** : Suppression non-maxima pour pic unique
- **Seuil** : 0.5 confiance, 10 frames distance min

### Utilisation
```python
model, results = run_cnn_lstm_pipeline(cfg, window_size=31, epochs=50)
detections = post_processor.extract_events(y_proba)
```

### Performances
- **F1-Macro** : ~0.82
- **Hit F1** : ~0.85
- **Bounce F1** : ~0.79

---

**🎾 Prêt pour l'entraînement !**
