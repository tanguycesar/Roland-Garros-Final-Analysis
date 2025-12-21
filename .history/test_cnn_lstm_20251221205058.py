"""
Script d'exemple pour tester l'architecture CNN-LSTM sur un point.
"""

import numpy as np
from pathlib import Path
import json

from hit_n_bounce.cnn_lstm_detector import (
    TimeSeriesWindowDataset,
    build_cnn_lstm_model,
    FocalLoss,
    EventPostProcessor,
    TennisEventMetrics,
    extract_raw_features,
    LABELS,
    LABEL_TO_ID
)
from hit_n_bounce.features import FeatureConfig, compute_kinematics
import hit_n_bounce.data_loader as io_utils

def test_architecture():
    """Test rapide de l'architecture sur un point unique."""
    
    print("="*70)
    print("TEST DE L'ARCHITECTURE CNN-LSTM")
    print("="*70)
    
    # Configuration
    cfg = FeatureConfig(fps=50.0, local_window=5)
    window_size = 31
    
    # Charger un point test
    data_folder = Path("Data hit & bounce/per_point_v2")
    test_file = list(data_folder.glob("ball_data_*.json"))[0]
    
    print(f"\n📁 Chargement: {test_file.name}")
    
    with open(test_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    frames, xs, ys, vis, actions = io_utils.extract_series(raw_data)
    
    print(f"   Frames: {len(frames)}")
    print(f"   Actions: {set(actions)}")
    
    # Calcul cinématique
    print(f"\n🔬 Calcul cinématique...")
    kin = compute_kinematics(frames, xs, ys, cfg)
    
    # Extraction features
    print(f"🎯 Extraction features...")
    X_raw, feature_names = extract_raw_features(kin)
    y = np.array([LABEL_TO_ID.get(a, 0) for a in actions])
    
    print(f"   Shape features brutes: {X_raw.shape}")
    print(f"   Features: {feature_names}")
    print(f"   Distribution labels: {dict(zip(LABELS, np.bincount(y)))}")
    
    # Fenêtrage
    print(f"\n🪟 Création des fenêtres (window_size={window_size})...")
    windower = TimeSeriesWindowDataset(window_size=window_size, stride=1)
    X_windows, y_windows, center_indices = windower.create_windows(X_raw, y)
    
    print(f"   Fenêtres créées: {X_windows.shape}")
    print(f"   Labels: {y_windows.shape}")
    
    # Construction du modèle
    print(f"\n🏗️  Construction du modèle CNN-LSTM...")
    model = build_cnn_lstm_model(
        window_size=window_size,
        n_features=X_raw.shape[1],
        n_classes=3
    )
    
    print(f"\n📊 Architecture:")
    model.summary()
    
    # Test Focal Loss
    print(f"\n🎯 Test Focal Loss...")
    focal_loss = FocalLoss(alpha=[0.05, 0.475, 0.475], gamma=2.0)
    
    # Compilation
    model.compile(
        optimizer='adam',
        loss=focal_loss,
        metrics=['accuracy']
    )
    
    print(f"   ✓ Focal Loss configurée")
    
    # Prédiction test (sans entraînement, juste pour vérifier la shape)
    print(f"\n🔮 Test de prédiction (modèle non entraîné)...")
    y_proba_test = model.predict(X_windows[:10], verbose=0)
    
    print(f"   Input shape: {X_windows[:10].shape}")
    print(f"   Output shape: {y_proba_test.shape}")
    print(f"   Probabilités exemple:")
    print(f"     Air: {y_proba_test[0, 0]:.4f}")
    print(f"     Hit: {y_proba_test[0, 1]:.4f}")
    print(f"     Bounce: {y_proba_test[0, 2]:.4f}")
    print(f"     Sum: {y_proba_test[0].sum():.4f} (doit être ≈1.0)")
    
    # Test Post-processing
    print(f"\n🎯 Test Post-processing (NMS)...")
    
    # Simuler des probabilités avec pics
    n_frames = len(y_windows)
    y_proba_fake = np.zeros((n_frames, 3))
    y_proba_fake[:, 0] = 0.9  # Majorité Air
    
    # Ajouter des pics de hit
    hit_frames = [50, 100, 150]
    for frame in hit_frames:
        if frame < n_frames:
            y_proba_fake[frame-2:frame+3, 0] = 0.1
            y_proba_fake[frame-2:frame+3, 1] = np.array([0.3, 0.6, 0.9, 0.6, 0.3])
    
    # Ajouter des pics de bounce
    bounce_frames = [75, 125]
    for frame in bounce_frames:
        if frame < n_frames:
            y_proba_fake[frame-2:frame+3, 0] = 0.1
            y_proba_fake[frame-2:frame+3, 2] = np.array([0.3, 0.6, 0.9, 0.6, 0.3])
    
    post_processor = EventPostProcessor(
        confidence_threshold=0.5,
        min_event_distance=10
    )
    
    detections = post_processor.extract_events(y_proba_fake)
    
    print(f"   Hits détectés: {len(detections['hits'])}")
    for frame, conf in detections['hits']:
        print(f"     Frame {frame}: confidence {conf:.4f}")
    
    print(f"   Bounces détectés: {len(detections['bounces'])}")
    for frame, conf in detections['bounces']:
        print(f"     Frame {frame}: confidence {conf:.4f}")
    
    # Test Métriques
    print(f"\n📈 Test Métriques...")
    
    # Simuler des prédictions
    y_pred = np.argmax(y_proba_fake, axis=1)
    
    metrics_obj = TennisEventMetrics()
    metrics = metrics_obj.compute_metrics(y_windows, y_pred, y_proba_fake)
    
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f}")
    print(f"   F1 par classe:")
    for label, f1 in metrics['f1_per_class'].items():
        print(f"     {label:8s}: {f1:.4f}")
    
    if 'pr_auc' in metrics:
        print(f"   PR-AUC:")
        for label, auc in metrics['pr_auc'].items():
            print(f"     {label:8s}: {auc:.4f}")
    
    print(f"\n   Matrice de confusion:")
    print(metrics['confusion_matrix'])
    
    print(f"\n{'='*70}")
    print(f"✅ TOUS LES TESTS PASSÉS AVEC SUCCÈS")
    print(f"{'='*70}")
    print(f"\n💡 L'architecture est prête pour l'entraînement.")
    print(f"   Lancer: python -m hit_n_bounce.cnn_lstm_detector")


if __name__ == "__main__":
    try:
        test_architecture()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
