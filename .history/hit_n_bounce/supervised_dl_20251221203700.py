"""
Architecture de Deep Learning hybride pour la détection d'événements tennis.
Implémentation: 1D-CNN + Bi-LSTM + Focal Loss + Post-processing NMS.

Architecture:
- Data Windowing: Fenêtres glissantes de 31 frames centrées sur chaque instant
- Feature Extraction: Conv1D pour capturer les signatures de chocs locaux
- Temporal Modeling: Bi-LSTM pour la cohérence temporelle avant/après
- Loss Function: Focal Loss pour gérer le déséquilibre extrême (< 2% événements)
- Post-processing: Non-Maximum Suppression pour obtenir une frame unique par événement
- Metrics: F1-Score, Precision, Recall, PR-AUC
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
    print("TensorFlow non disponible. Installer avec: pip install tensorflow")

# Imports projet
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
    
    def create_augmented_windows(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmentation_factor: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée des fenêtres avec augmentation de données pour les événements rares.
        """
        X_base, y_base, _ = self.create_windows(features, labels)
        
        # Identifier les fenêtres contenant des événements
        event_mask = (y_base == 1) | (y_base == 2)  # Hit ou Bounce
        
        X_events = X_base[event_mask]
        y_events = y_base[event_mask]
        
        # Augmentation par bruit gaussien léger
        X_augmented = [X_base, y_base]
        
        for _ in range(augmentation_factor - 1):
            noise = np.random.normal(0, 0.02, X_events.shape)
            X_aug = X_events + noise
            X_augmented.append((X_aug, y_events))
        
        # Combinaison
        X_final = np.vstack([x for x, _ in X_augmented])
        y_final = np.concatenate([y for _, y in X_augmented])
        
        return X_final, y_final


# =========================================================================
# 2. FOCAL LOSS - GESTION DU DÉSÉQUILIBRE
# =========================================================================

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss pour gérer le déséquilibre extrême des classes.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
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
        """
        Calcul de la Focal Loss.
        
        Args:
            y_true: (batch_size,) - indices de classe
            y_pred: (batch_size, n_classes) - probabilités softmax
        """
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
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma
        })
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
    
    [1D-CNN Block]
      ↓ Conv1D(64) + ReLU + BatchNorm - Détection de motifs locaux
      ↓ Conv1D(128) + ReLU + BatchNorm - Features complexes
      ↓ MaxPooling1D - Réduction dimensionnelle
      ↓ Conv1D(256) + ReLU + BatchNorm - Features abstraites
      ↓ Dropout(0.3)
    
    [Bi-LSTM Block]
      ↓ Bi-LSTM(128) return_sequences - Cohérence temporelle bidirectionnelle
      ↓ Bi-LSTM(64) return_sequences - Raffinement
      ↓ Global Average Pooling - Agrégation
    
    [Classification Head]
      ↓ Dense(128) + ReLU + Dropout
      ↓ Dense(64) + ReLU
      ↓ Dense(3) + Softmax
    
    Output: (batch, 3) - probabilités [air, hit, bounce]
    """
    
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis")
    
    inputs = layers.Input(shape=(window_size, n_features), name="input_window")
    
    # ===== CNN Block: Extraction de features locales =====
    # Première couche: détection de motifs simples (chocs, changements de vitesse)
    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    
    # Deuxième couche: features plus complexes
    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)
    
    # Troisième couche: features abstraites
    x = layers.Conv1D(
        filters=256,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='conv1d_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # ===== Bi-LSTM Block: Cohérence temporelle =====
    # Capture les dépendances long terme avant/après l'événement
    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.001)
        ),
        name='bilstm_1'
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(0.001)
        ),
        name='bilstm_2'
    )(x)
    
    # Agrégation temporelle
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # ===== Classification Head =====
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_1'
    )(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_2'
    )(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(
        n_classes,
        activation='softmax',
        name='output'
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_TennisEvents")
    
    return model
def make_frame_features(kin: Dict[str, np.ndarray], cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Crée des features basées sur la physique réelle du court avec contexte étendu.
    """
    ym = np.asarray(kin["ym"], float)
    xm = np.asarray(kin["xm"], float)
    vy = np.asarray(kin["vy"], float)
    vx = np.asarray(kin["vx"], float)
    ay = np.asarray(kin["ay"], float)
    ax = np.asarray(kin["ax"], float)
    jerk = np.asarray(kin["jerk"], float)
    turn = np.asarray(kin["turn_rate"], float)
    speed = np.asarray(kin["speed"], float)
    accel = np.asarray(kin["accel"], float)

    dist_baseline = np.abs(np.abs(ym) - 11.88)
    dist_net = np.abs(ym)
    dist_sideline = np.abs(xm) - 4.115
    
    flip_vy = np.zeros_like(vy)
    flip_vy[1:] = (np.sign(vy[1:]) != np.sign(vy[:-1])).astype(float)
    flip_vx = np.zeros_like(vx)
    flip_vx[1:] = (np.sign(vx[1:]) != np.sign(vx[:-1])).astype(float)
    
    speed_safe = np.where(speed > 0.1, speed, 0.1)
    accel_ratio = accel / speed_safe
    jerk_ratio = jerk / (accel + 1e-6)
    
    djerk = np.gradient(np.nan_to_num(jerk, nan=0.0))
    dturn = np.gradient(np.nan_to_num(turn, nan=0.0))

    signals = {
        "ym": ym, "xm": xm, "vy": vy, "vx": vx, 
        "ay": ay, "ax": ax, "jk": jerk, "tn": turn, 
        "sp": speed, "ac": accel,
        "db": dist_baseline, "dn": dist_net, "ds": dist_sideline,
        "fvy": flip_vy, "fvx": flip_vx,
        "ar": accel_ratio, "jr": jerk_ratio,
        "dj": djerk, "dt": dturn
    }

    feature_arrays, feature_names = [], []
    w = 5  # Fenêtre étendue : 11 frames au total
    
    for name, arr in signals.items():
        arr_clean = np.nan_to_num(arr, nan=0.0)
        for shift in range(-w, w + 1):
            col = np.roll(arr_clean, shift)
            if shift > 0: col[:shift] = 0.0
            elif shift < 0: col[shift:] = 0.0
            feature_arrays.append(col)
            feature_names.append(f"{name}_{shift:+d}")

    return np.stack(feature_arrays, axis=1), feature_names


# -------------------------------------------------------------------------
# Construction du modèle Deep Learning
# -------------------------------------------------------------------------
def build_lstm_model(input_dim: int, sequence_length: int = 11) -> keras.Model:
    """
    Construit un modèle LSTM bidirectionnel pour la classification.
    
    Architecture:
    - Reshape des features temporelles en séquences
    - Bidirectional LSTM pour capturer les patterns avant/après
    - Global Average Pooling pour agréger la séquence
    - Dense layers avec dropout pour régularisation
    - Softmax pour classification 3 classes
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis pour ce modèle")
    
    # Nombre de signaux (features par frame)
    n_signals = input_dim // sequence_length
    
    # Input: (batch, input_dim)
    inputs = layers.Input(shape=(input_dim,), name="features")
    
    # Reshape en séquence temporelle: (batch, sequence_length, n_signals)
    x = layers.Reshape((sequence_length, n_signals))(inputs)
    
    # Bidirectional LSTM - capture les patterns temporels avant/après
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name="bilstm_1"
    )(x)
    
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name="bilstm_2"
    )(x)
    
    # Global pooling pour agréger la séquence
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers avec régularisation
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(3, activation='softmax', name="classification")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="TennisEventLSTM")
    
    return model


def build_dense_model(input_dim: int) -> keras.Model:
    """
    Construit un modèle Dense (MLP) profond pour la classification.
    Plus simple et rapide que LSTM, mais sans structure temporelle explicite.
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis pour ce modèle")
    
    inputs = layers.Input(shape=(input_dim,), name="features")
    
    # Normalisation batch
    x = layers.BatchNormalization()(inputs)
    
    # Architecture profonde avec skip connections
    x1 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x1 = layers.Dropout(0.4)(x1)
    x1 = layers.BatchNormalization()(x1)
    
    x2 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.BatchNormalization()(x2)
    
    x3 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x2)
    x3 = layers.Dropout(0.3)(x3)
    x3 = layers.BatchNormalization()(x3)
    
    x4 = layers.Dense(64, activation='relu')(x3)
    x4 = layers.Dropout(0.2)(x4)
    
    # Output
    outputs = layers.Dense(3, activation='softmax', name="classification")(x4)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="TennisEventDense")
    
    return model


# -------------------------------------------------------------------------
# Entraînement et Évaluation
# -------------------------------------------------------------------------
def train_deep_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str = "lstm",  # "lstm" ou "dense"
    epochs: int = 30,
    batch_size: int = 128,
    validation_split: float = 0.15
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Entraîne un modèle deep learning avec validation croisée.
    """
    if not _HAS_TF:
        raise RuntimeError("TensorFlow requis")
    
    print(f"\n{'='*60}")
    print(f"Entraînement du modèle Deep Learning ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Classes: {np.bincount(y)}")
    
    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcul des poids de classe pour équilibrer
    class_counts = np.bincount(y)
    class_weights = {i: len(y) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    print(f"Class weights: {class_weights}")
    
    # Split validation
    n_val = int(len(X_scaled) * validation_split)
    indices = np.arange(len(X_scaled))
    np.random.shuffle(indices)
    
    X_train = X_scaled[indices[n_val:]]
    y_train = y[indices[n_val:]]
    X_val = X_scaled[indices[:n_val]]
    y_val = y[indices[:n_val]]
    
    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    # Construction du modèle
    if model_type.lower() == "lstm":
        model = build_lstm_model(X.shape[1])
    else:
        model = build_dense_model(X.shape[1])
    
    # Compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']  # Simplifié pour éviter les problèmes de shape
    )
    
    print(f"\n{model.summary()}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_model_dl.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Évaluation finale
    y_pred = np.argmax(model.predict(X_scaled, verbose=0), axis=1)
    f1_macro = f1_score(y, y_pred, average="macro")
    
    print(f"\n{'='*60}")
    print(f"Résultats finaux")
    print(f"{'='*60}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(classification_report(y, y_pred, target_names=LABELS, digits=4))
    
    metrics = {
        "f1_macro": f1_macro,
        "history": history.history,
        "scaler": scaler,
        "model_type": model_type
    }
    
    return model, metrics


# -------------------------------------------------------------------------
# Pipeline complet
# -------------------------------------------------------------------------
def run_supervised_dl_pipeline(cfg: FeatureConfig, model_type: str = "lstm"):
    """
    Pipeline complet: chargement des données, feature engineering, entraînement DL.
    """
    import json
    
    data_folder = Path("Data hit & bounce/per_point_v2")
    json_files = sorted(data_folder.glob("ball_data_*.json"))
    
    all_X, all_y, all_groups = [], [], []
    
    print("Chargement et extraction des features...")
    for idx, json_path in enumerate(json_files):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        frames, xs, ys, vis, actions = io_utils.extract_series(raw_data)
        
        if len(frames) < 10:
            continue
        
        kin = compute_kinematics(frames, xs, ys, cfg)
        X_point, feat_names = make_frame_features(kin, cfg)
        
        # Labels
        y_point = np.array([LABEL_TO_ID.get(a, 0) for a in actions])
        
        all_X.append(X_point)
        all_y.append(y_point)
        all_groups.extend([idx] * len(actions))
        
        if (idx + 1) % 50 == 0:
            print(f"  Traité: {idx + 1}/{len(json_files)}")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)
    
    print(f"\nDataset total: {len(X)} frames, {X.shape[1]} features")
    print(f"Distribution: {dict(zip(LABELS, np.bincount(y)))}")
    
    # Entraînement
    model, metrics = train_deep_model(X, y, groups, model_type=model_type)
    
    # Sauvegarde
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model.save(f"models/tennis_event_classifier_dl_{model_type}.keras")
    print(f"\nModèle sauvegardé: models/tennis_event_classifier_dl_{model_type}.keras")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['history']['loss'], label='Train Loss')
    plt.plot(metrics['history']['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['history']['accuracy'], label='Train Acc')
    plt.plot(metrics['history']['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if 'precision' in metrics['history']:
        plt.plot(metrics['history']['precision'], label='Train Precision')
        plt.plot(metrics['history']['val_precision'], label='Val Precision')
    if 'recall' in metrics['history']:
        plt.plot(metrics['history']['recall'], label='Train Recall')
        plt.plot(metrics['history']['val_recall'], label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Metrics')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves_dl.png', dpi=150)
    print("Courbes d'entraînement sauvegardées: models/training_curves_dl.png")
    
    return model, metrics


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    if not _HAS_TF:
        print("Erreur: TensorFlow non disponible.")
        print("Installer avec: pip install tensorflow")
        exit(1)
    
    # Configuration
    cfg = FeatureConfig()
    
    # Choix du modèle: "lstm" ou "dense"
    model_type = "lstm"  # Changez en "dense" pour tester le modèle MLP
    
    print(f"Démarrage du pipeline Deep Learning ({model_type.upper()})...")
    model, metrics = run_supervised_dl_pipeline(cfg, model_type=model_type)
    
    print("\n✓ Pipeline terminé avec succès!")
