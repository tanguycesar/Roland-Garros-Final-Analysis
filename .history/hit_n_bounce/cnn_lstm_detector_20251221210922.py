"""
🎾 Architecture Deep Learning Hybride pour Détection d'Événements Tennis
========================================================================

**Auteur**: Architecture 1D-CNN + Bi-LSTM avec Focal Loss
**Objectif**: Classifier chaque frame en 3 classes: Air (0), Hit (1), Bounce (2)

## 📋 Spécifications Techniques

### 1. Data Windowing
- Fenêtres glissantes de **31 frames** (±15 frames autour du centre)
- Features: $x_m, y_m, v_x, v_y, a_x, a_y, speed, jerk, turn\\_rate$
- Padding: Edge mode pour gérer les bords

### 2. Architecture Hybride
```
Input(31, 9) 
  ↓
[Conv1D Block]
  Conv1D(64, k=5) → BatchNorm → ReLU
  Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
  Conv1D(256, k=3) → BatchNorm → ReLU → Dropout(0.3)
  ↓
[Bi-LSTM Block]
  Bi-LSTM(128) → return_sequences
  Bi-LSTM(64) → return_sequences
  GlobalAveragePooling1D
  ↓
[Dense Head]
  Dense(128) → ReLU → Dropout(0.4)
  Dense(64) → ReLU → Dropout(0.3)
  Dense(3) → Softmax
```

### 3. Focal Loss
$$FL(p_t) = -\\alpha_t \\cdot (1 - p_t)^\\gamma \\cdot \\log(p_t)$$

- **α** (alpha): Poids par classe `[0.05, 0.475, 0.475]` (favorise Hit/Bounce)
- **γ** (gamma): Facteur de focusing = 2.0
- **Objectif**: Compenser le déséquilibre extrême (Air ≈ 98%, événements ≈ 2%)

### 4. Post-Processing (NMS)
- Recherche de pics locaux avec `scipy.find_peaks`
- Seuil de confiance: 0.5
- Distance minimale entre événements: 10 frames
- **Output**: Frame unique par événement détecté

### 5. Métriques
- **F1-Score Macro**: Moyenne non pondérée (important pour classes rares)
- **Precision/Recall par classe**
- **PR-AUC** (Precision-Recall Area Under Curve)
- **Confusion Matrix**

## 🚀 Utilisation

```python
from hit_n_bounce.cnn_lstm_detector import run_cnn_lstm_pipeline
from hit_n_bounce.features import FeatureConfig

cfg = FeatureConfig(fps=50.0)

model, results = run_cnn_lstm_pipeline(
    cfg=cfg,
    window_size=31,
    stride=1,
    epochs=50,
    batch_size=256,
    use_focal_loss=True
)
```

## 📊 Performances Attendues
- **F1-Score Macro**: > 0.75
- **Hit Detection**: Precision ~0.85, Recall ~0.80
- **Bounce Detection**: Precision ~0.80, Recall ~0.75

"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, 
    classification_report, 
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import tensorflow.keras.backend as K
    _HAS_TF = True
except Exception:
    _HAS_TF = False
    print("⚠️ TensorFlow non disponible. Installer avec: pip install tensorflow")

# Imports projet (support imports relatifs et absolus)
try:
    from . import data_loader as io_utils
    from .features import FeatureConfig, compute_kinematics
except ImportError:
    import data_loader as io_utils
    from features import FeatureConfig, compute_kinematics

LABELS = ("air", "hit", "bounce")
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {i: k for k, i in LABEL_TO_ID.items()}


# =========================================================================
# 1. DATA WINDOWING - FENÊTRES GLISSANTES
# =========================================================================

class TimeSeriesWindowDataset:
    """
    Génère des fenêtres glissantes centrées pour la classification d'événements.
    
    Paramètres:
    -----------
    window_size : int
        Taille de la fenêtre (ex: 31 frames = ±15 autour du centre)
    stride : int
        Pas de décalage entre fenêtres (1 = toutes les frames)
    
    Exemple:
    --------
    >>> windower = TimeSeriesWindowDataset(window_size=31, stride=1)
    >>> X_windows, y_windows, indices = windower.create_windows(features, labels)
    >>> print(X_windows.shape)  # (n_windows, 31, 9)
    """
    
    def __init__(self, window_size: int = 31, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.half_window = window_size // 2
        
    def create_windows(
        self, 
        features: np.ndarray,  # (n_frames, n_features)
        labels: np.ndarray     # (n_frames,)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crée des fenêtres glissantes centrées.
        
        Returns:
        --------
        X_windows : (n_windows, window_size, n_features)
        y_windows : (n_windows,) - label de la frame centrale
        center_indices : (n_windows,) - index de la frame centrale
        """
        n_frames, n_features = features.shape
        
        # Padding pour gérer les bords
        pad_width = ((self.half_window, self.half_window), (0, 0))
        features_padded = np.pad(features, pad_width, mode='edge')
        
        windows = []
        window_labels = []
        center_indices = []
        
        # Extraction des fenêtres avec stride
        for i in range(0, n_frames, self.stride):
            # Fenêtre centrée sur i (après padding, c'est i + half_window)
            start = i
            end = i + self.window_size
            window = features_padded[start:end, :]
            
            if window.shape[0] == self.window_size:
                windows.append(window)
                window_labels.append(labels[i])  # Label de la frame centrale
                center_indices.append(i)
        
        X_windows = np.array(windows)  # (n_windows, window_size, n_features)
        y_windows = np.array(window_labels)
        center_indices = np.array(center_indices)
        
        return X_windows, y_windows, center_indices


# =========================================================================
# 2. FOCAL LOSS - GESTION DU DÉSÉQUILIBRE
# =========================================================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss pour gérer le déséquilibre extrême des classes.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Référence: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Paramètres:
    -----------
    alpha : List[float]
        Poids par classe (ex: [0.05, 0.475, 0.475] pour favoriser hit/bounce)
    gamma : float
        Facteur de focusing (2.0 par défaut)
    """
    
    def __init__(self, alpha: List[float] = None, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha if alpha is not None else [0.05, 0.475, 0.475]
        self.gamma = gamma
        self.alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
        
    def call(self, y_true, y_pred):
        """Calcul de la Focal Loss."""
        # Conversion en one-hot
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true_int, depth=3)
        
        # Clipping pour stabilité numérique
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Cross-entropy
        ce = -y_true_oh * tf.math.log(y_pred)
        
        # Facteur de modulation: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1, keepdims=True)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Poids alpha par classe
        alpha_factor = tf.reduce_sum(y_true_oh * self.alpha_tensor, axis=-1, keepdims=True)
        
        # Focal Loss
        focal_loss = alpha_factor * modulating_factor * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma": self.gamma})
        return config


# =========================================================================
# 3. ARCHITECTURE HYBRIDE 1D-CNN + Bi-LSTM
# =========================================================================

def build_cnn_lstm_model(
    window_size: int = 31,
    n_features: int = 9,
    n_classes: int = 3
) -> keras.Model:
    """
    Architecture hybride pour détection d'événements tennis.
    
    Architecture:
    -------------
    Input: (batch, window_size, n_features)
    
    [1D-CNN Block] - Extraction de motifs locaux
      Conv1D(64, k=5) → BatchNorm → ReLU
      Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
      Conv1D(256, k=3) → BatchNorm → ReLU → Dropout(0.3)
    
    [Bi-LSTM Block] - Cohérence temporelle
      Bi-LSTM(128, return_sequences)
      Bi-LSTM(64, return_sequences)
      GlobalAveragePooling1D
    
    [Dense Head] - Classification
      Dense(128) → ReLU → Dropout(0.4)
      Dense(64) → ReLU → Dropout(0.3)
      Dense(3) → Softmax
    
    Output: (batch, 3) - probabilités [air, hit, bounce]
    
    Paramètres:
    -----------
    window_size : int
        Nombre de frames dans la fenêtre temporelle
    n_features : int
        Nombre de features par frame
    n_classes : int
        Nombre de classes (3: air, hit, bounce)
    
    Returns:
    --------
    model : keras.Model
        Modèle compilé prêt à l'entraînement
    """
    
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis")
    
    inputs = layers.Input(shape=(window_size, n_features), name="input_window")
    
    # ===== CNN Block: Extraction de features locales =====
    x = layers.Conv1D(
        filters=64, kernel_size=5, padding='same',
        activation='relu', kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(
        filters=128, kernel_size=3, padding='same',
        activation='relu', kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    x = layers.Conv1D(
        filters=256, kernel_size=3, padding='same',
        activation='relu', kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # ===== Bi-LSTM Block: Cohérence temporelle =====
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                   kernel_regularizer=regularizers.l2(0.001)),
        name='bilstm_1'
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2,
                   kernel_regularizer=regularizers.l2(0.001)),
        name='bilstm_2'
    )(x)
    
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # ===== Classification Head =====
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_TennisEvents")
    
    return model


# =========================================================================
# 4. POST-PROCESSING - NON-MAXIMUM SUPPRESSION
# =========================================================================

class EventPostProcessor:
    """
    Post-traitement pour obtenir une frame unique par événement détecté.
    
    Utilise Non-Maximum Suppression (NMS) pour extraire les pics de probabilité
    et éliminer les détections multiples du même événement.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_window: int = 5,
        min_event_distance: int = 10
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_window = nms_window
        self.min_event_distance = min_event_distance
    
    def apply_nms(
        self,
        probabilities: np.ndarray,  # (n_frames, n_classes)
        class_id: int = 1
    ) -> List[int]:
        """Applique NMS pour extraire les pics."""
        class_probs = probabilities[:, class_id]
        
        peaks, _ = find_peaks(
            class_probs,
            height=self.confidence_threshold,
            distance=self.min_event_distance
        )
        
        if len(peaks) > 0:
            peak_confidences = class_probs[peaks]
            sorted_indices = np.argsort(peak_confidences)[::-1]
            peaks = peaks[sorted_indices]
        
        return peaks.tolist()
    
    def extract_events(
        self,
        probabilities: np.ndarray,
        threshold_hit: float = 0.5,
        threshold_bounce: float = 0.5
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Extrait tous les événements avec NMS."""
        results = {'hits': [], 'bounces': []}
        
        # Hits (classe 1)
        hit_peaks = self.apply_nms(probabilities, class_id=1)
        for peak in hit_peaks:
            conf = probabilities[peak, 1]
            if conf >= threshold_hit:
                results['hits'].append((peak, conf))
        
        # Bounces (classe 2)
        bounce_peaks = self.apply_nms(probabilities, class_id=2)
        for peak in bounce_peaks:
            conf = probabilities[peak, 2]
            if conf >= threshold_bounce:
                results['bounces'].append((peak, conf))
        
        return results


# =========================================================================
# 5. MÉTRIQUES AVANCÉES
# =========================================================================

class TennisEventMetrics:
    """Calcul de métriques adaptées aux événements rares."""
    
    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calcule F1, Precision, Recall, CM, PR-AUC."""
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        cm = confusion_matrix(y_true, y_pred)
        
        report = classification_report(
            y_true, y_pred,
            target_names=LABELS,
            digits=4,
            output_dict=True
        )
        
        metrics = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': {LABELS[i]: f1_per_class[i] for i in range(len(LABELS))},
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # PR-AUC
        if y_proba is not None:
            pr_auc = {}
            for class_id, class_name in enumerate(LABELS):
                y_true_binary = (y_true == class_id).astype(int)
                y_score = y_proba[:, class_id]
                
                if len(np.unique(y_true_binary)) > 1:
                    auc = average_precision_score(y_true_binary, y_score)
                    pr_auc[class_name] = auc
            
            metrics['pr_auc'] = pr_auc
        
        return metrics
    
    @staticmethod
    def plot_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, figsize=(15, 5)):
        """Trace les courbes Precision-Recall."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for class_id, (ax, class_name) in enumerate(zip(axes, LABELS)):
            y_true_binary = (y_true == class_id).astype(int)
            y_score = y_proba[:, class_id]
            
            if len(np.unique(y_true_binary)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
                auc = average_precision_score(y_true_binary, y_score)
                
                ax.plot(recall, precision, linewidth=2, label=f'PR-AUC = {auc:.4f}')
                ax.fill_between(recall, precision, alpha=0.2)
            else:
                ax.text(0.5, 0.5, 'Pas d\'exemples positifs', ha='center', va='center')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve - {class_name.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, figsize=(8, 6)):
        """Visualise la matrice de confusion."""
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel='Predicted',
            ylabel='True',
            title='Confusion Matrix'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig


# =========================================================================
# 6. FEATURE EXTRACTION
# =========================================================================

def extract_raw_features(kin: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Extrait les features brutes (sans fenêtrage temporel).
    
    Features (9):
    -------------
    1. x_m, y_m : Position (mètres)
    2. v_x, v_y : Vitesse (m/s)
    3. a_x, a_y : Accélération (m/s²)
    4. speed : Vitesse scalaire (m/s)
    5. jerk : Jerk (m/s³)
    6. turn_rate : Turn rate (rad/s)
    """
    features = np.stack([
        np.nan_to_num(kin["xm"], nan=0.0),
        np.nan_to_num(kin["ym"], nan=0.0),
        np.nan_to_num(kin["vx"], nan=0.0),
        np.nan_to_num(kin["vy"], nan=0.0),
        np.nan_to_num(kin["ax"], nan=0.0),
        np.nan_to_num(kin["ay"], nan=0.0),
        np.nan_to_num(kin["speed"], nan=0.0),
        np.nan_to_num(kin["jerk"], nan=0.0),
        np.nan_to_num(kin["turn_rate"], nan=0.0)
    ], axis=1)
    
    feature_names = ["x_m", "y_m", "v_x", "v_y", "a_x", "a_y", "speed", "jerk", "turn_rate"]
    
    return features, feature_names


# =========================================================================
# 7. ENTRAÎNEMENT
# =========================================================================

def train_cnn_lstm_model(
    X_windows: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    window_size: int = 31,
    n_features: int = 9,
    epochs: int = 50,
    batch_size: int = 256,
    validation_split: float = 0.15,
    focal_loss: bool = True,
    alpha: List[float] = None
) -> Tuple[keras.Model, Dict[str, Any]]:
    """Entraîne le modèle CNN-LSTM avec Focal Loss."""
    
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis")
    
    print(f"\n{'='*70}")
    print(f"🏗️  ENTRAÎNEMENT: Modèle Hybride 1D-CNN + Bi-LSTM")
    print(f"{'='*70}")
    print(f"📊 Dataset: {X_windows.shape[0]:,} samples | Window: {window_size} frames | Features: {n_features}")
    
    class_counts = np.bincount(y)
    for i, label in enumerate(LABELS):
        print(f"   {label:8s}: {class_counts[i]:,} ({class_counts[i]/len(y)*100:.2f}%)")
    
    # Normalisation
    print(f"\n🔧 Normalisation des features...")
    scaler = StandardScaler()
    n_samples, ws, nf = X_windows.shape
    X_flat = X_windows.reshape(-1, nf)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, ws, nf)
    
    # Split
    n_val = int(len(X_scaled) * validation_split)
    indices = np.arange(len(X_scaled))
    np.random.shuffle(indices)
    
    X_train = X_scaled[indices[n_val:]]
    y_train = y[indices[n_val:]]
    X_val = X_scaled[indices[:n_val]]
    y_val = y[indices[:n_val]]
    
    print(f"📈 Train: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # Modèle
    print(f"\n🏗️  Construction du modèle...")
    model = build_cnn_lstm_model(window_size, n_features, n_classes=3)
    
    # Loss
    if focal_loss:
        if alpha is None:
            alpha = [len(y)/(3*c) if c>0 else 1.0 for c in class_counts]
            alpha = [a/sum(alpha) for a in alpha]
        print(f"🎯 Focal Loss - Alpha: {[f'{a:.3f}' for a in alpha]}")
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        loss_fn = 'sparse_categorical_crossentropy'
    
    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint('models/best_cnn_lstm_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # Entraînement
    print(f"\n🚀 Début de l'entraînement...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation
    print(f"\n📊 Évaluation...")
    y_proba = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    metrics_obj = TennisEventMetrics()
    metrics = metrics_obj.compute_metrics(y, y_pred, y_proba)
    
    print(f"\n{'='*70}")
    print(f"✅ RÉSULTATS FINAUX")
    print(f"{'='*70}")
    print(f"F1-Score Macro: {metrics['f1_macro']:.4f}")
    print(f"F1-Score Weighted: {metrics['f1_weighted']:.4f}")
    print(f"\nF1 par classe:")
    for label, f1 in metrics['f1_per_class'].items():
        print(f"  {label:8s}: {f1:.4f}")
    
    if 'pr_auc' in metrics:
        print(f"\nPR-AUC:")
        for label, auc in metrics['pr_auc'].items():
            print(f"  {label:8s}: {auc:.4f}")
    
    print(f"\n{classification_report(y, y_pred, target_names=LABELS, digits=4)}")
    
    results = {
        "scaler": scaler,
        "history": history.history,
        "metrics": metrics,
        "window_size": window_size,
        "n_features": n_features
    }
    
    return model, results


# =========================================================================
# 8. PIPELINE COMPLET
# =========================================================================

def run_cnn_lstm_pipeline(
    cfg: FeatureConfig,
    window_size: int = 31,
    stride: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    use_focal_loss: bool = True
):
    """Pipeline complet de bout en bout."""
    
    import json
    
    print(f"\n{'='*70}")
    print(f"🎾 PIPELINE DEEP LEARNING - DÉTECTION D'ÉVÉNEMENTS TENNIS")
    print(f"{'='*70}\n")
    
    # Chargement
    print(f"📁 Chargement des données...")
    data_folder = Path("Data hit & bounce/per_point_v2")
    json_files = sorted(data_folder.glob("ball_data_*.json"))
    print(f"   Fichiers trouvés: {len(json_files)}")
    
    all_X_raw, all_y, all_groups = [], [], []
    
    for idx, json_path in enumerate(json_files):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        frames, xs, ys, vis, actions = io_utils.extract_series(raw_data)
        
        if len(frames) < window_size:
            continue
        
        kin = compute_kinematics(frames, xs, ys, cfg)
        X_raw, _ = extract_raw_features(kin)
        y_point = np.array([LABEL_TO_ID.get(a, 0) for a in actions])
        
        all_X_raw.append(X_raw)
        all_y.append(y_point)
        all_groups.extend([idx] * len(actions))
        
        if (idx + 1) % 50 == 0:
            print(f"   Traité: {idx + 1}/{len(json_files)}")
    
    X_raw_all = np.vstack(all_X_raw)
    y_all = np.concatenate(all_y)
    groups_all = np.array(all_groups)
    
    print(f"\n✅ Dataset brut: {len(X_raw_all):,} frames | Features: {X_raw_all.shape[1]}")
    print(f"   Distribution: {dict(zip(LABELS, np.bincount(y_all)))}")
    
    # Fenêtrage
    print(f"\n🪟 Création des fenêtres glissantes (size={window_size}, stride={stride})...")
    windower = TimeSeriesWindowDataset(window_size=window_size, stride=stride)
    X_windows, y_windows, _ = windower.create_windows(X_raw_all, y_all)
    
    print(f"   Fenêtres créées: {len(X_windows):,} | Shape: {X_windows.shape}")
    
    # Entraînement
    model, results = train_cnn_lstm_model(
        X_windows, y_windows, groups_all,
        window_size=window_size, n_features=X_raw_all.shape[1],
        epochs=epochs, batch_size=batch_size, focal_loss=use_focal_loss
    )
    
    # Sauvegarde
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model.save(model_dir / "cnn_lstm_tennis_events.keras")
    print(f"\n💾 Modèle sauvegardé: models/cnn_lstm_tennis_events.keras")
    
    import pickle
    with open(model_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)
    print(f"💾 Scaler sauvegardé: models/feature_scaler.pkl")
    
    # Visualisations
    print(f"\n📊 Génération des visualisations...")
    
    X_scaled = results["scaler"].transform(X_windows.reshape(-1, X_raw_all.shape[1])).reshape(X_windows.shape)
    y_proba = model.predict(X_scaled, verbose=0)
    
    metrics_obj = TennisEventMetrics()
    
    fig_pr = metrics_obj.plot_pr_curves(y_windows, y_proba)
    fig_pr.savefig(model_dir / "pr_curves.png", dpi=150, bbox_inches='tight')
    
    fig_cm = metrics_obj.plot_confusion_matrix(results["metrics"]["confusion_matrix"])
    fig_cm.savefig(model_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"   ✓ Courbes PR: pr_curves.png")
    print(f"   ✓ Matrice de confusion: confusion_matrix.png")
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print(f"{'='*70}\n")
    
    return model, results


# =========================================================================
# 9. MAIN
# =========================================================================

if __name__ == "__main__":
    if not _HAS_TF:
        print("❌ TensorFlow non disponible. Installer: pip install tensorflow")
        exit(1)
    
    cfg = FeatureConfig(fps=50.0, local_window=5)
    
    # Hyperparamètres
    WINDOW_SIZE = 31
    STRIDE = 1
    EPOCHS = 50
    BATCH_SIZE = 256
    USE_FOCAL_LOSS = True
    
    print(f"Configuration:")
    print(f"  Window Size: {WINDOW_SIZE} frames")
    print(f"  Stride: {STRIDE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Loss: {'Focal Loss' if USE_FOCAL_LOSS else 'CrossEntropy'}")
    
    model, results = run_cnn_lstm_pipeline(
        cfg=cfg,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_focal_loss=USE_FOCAL_LOSS
    )
    
    print("\n🎾 Modèle prêt pour l'inférence!")
