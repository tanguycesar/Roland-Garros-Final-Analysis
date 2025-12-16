from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from joblib import dump, load
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # <-- L'arme secrète

# Importation de tes modules locaux
from features import FeatureConfig, compute_kinematics
import data_loader as io_utils 

LABELS = ("air", "hit", "bounce")
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {i: k for k, i in LABEL_TO_ID.items()}

# -------------------------------------------------------------------------
# Feature Engineering (Fenêtre glissante)
# -------------------------------------------------------------------------
def make_frame_features(kin: Dict[str, np.ndarray], cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Crée un vecteur de features pour chaque frame en utilisant une fenêtre glissante.
    """
    vx = kin["vx"]
    vy = kin["vy"]
    ax = kin["ax"]
    ay = kin["ay"]
    speed = kin["speed"]
    turn = kin["turn_rate"]
    y_pos = kin["ys"]
    ground_dist = kin["ground_y"] - y_pos

    n_frames = len(vx)
    # Fenêtre un peu plus large pour le Random Forest
    w = 2 
    
    feature_arrays = []
    feature_names = []
    
    signals = {
        "vy": vy, "ay": ay, "sp": speed, "tn": turn, "gd": ground_dist
    }

    for name, arr in signals.items():
        arr_clean = np.nan_to_num(arr, nan=0.0)
        for shift in range(-w, w + 1):
            col = np.roll(arr_clean, shift)
            if shift > 0: col[:shift] = 0
            elif shift < 0: col[shift:] = 0
            feature_arrays.append(col)
            suffix = f"_{shift}" if shift < 0 else f"+{shift}"
            feature_names.append(f"{name}{suffix}")

    X = np.stack(feature_arrays, axis=1)
    return X, feature_names

# -------------------------------------------------------------------------
# Création du Dataset
# -------------------------------------------------------------------------
def _make_dataset(points_dir: str | Path, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    files = io_utils.iter_point_files(points_dir)
    Xs, ys, groups = [], [], []
    feat_names = None

    print(f"Chargement et nettoyage de {len(files)} fichiers pour l'entraînement...")

    for gi, fp in enumerate(files):
        data = io_utils.load_ball_json(fp)
        frames, xs, ys_, vis, acts = io_utils.extract_series(data)
        kin = compute_kinematics(frames, xs, ys_, vis, cfg)
        X, names = make_frame_features(kin, cfg)
        
        if feat_names is None: feat_names = names

        mask = np.asarray(vis, dtype=bool)
        X_subset = X[mask]
        y_subset = np.array([LABEL_TO_ID.get(a, 0) for (a, m) in zip(acts, mask) if m], dtype=int)

        if len(y_subset) > 0:
            Xs.append(X_subset)
            ys.append(y_subset)
            groups.append(np.full_like(y_subset, gi, dtype=int))

    if not Xs: raise ValueError("Aucune donnée valide.")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    g_all = np.concatenate(groups, axis=0)
    
    print(f"Dataset : {X_all.shape[0]} frames, {X_all.shape[1]} features.")
    return X_all, y_all, g_all, feat_names or []

# -------------------------------------------------------------------------
# Post-Processing (Nettoyage des prédictions)
# -------------------------------------------------------------------------
def clean_predictions(pred_labels: List[str]) -> List[str]:
    """
    Transforme une suite de frames bruyantes en événements propres.
    Ex: [air, hit, hit, hit, air] -> [air, air, hit, air, air]
    On ne garde que le centre de l'événement.
    """
    n = len(pred_labels)
    cleaned = ["air"] * n
    
    # On itère pour trouver les séquences consécutives de 'hit' ou 'bounce'
    i = 0
    while i < n:
        label = pred_labels[i]
        if label in ["hit", "bounce"]:
            start = i
            while i < n and pred_labels[i] == label:
                i += 1
            end = i
            
            # On a une séquence de 'start' à 'end'
            # On place l'événement unique au milieu ou au pic
            center = (start + end) // 2
            cleaned[center] = label
        else:
            i += 1
            
    return cleaned

# -------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------
def train_supervised(points_dir: str | Path,
                     model_path: str | Path,
                     cfg: FeatureConfig | None = None,
                     n_splits: int = 5) -> Dict[str, Any]:
    if cfg is None: cfg = FeatureConfig()

    X, y, groups, feat_names = _make_dataset(points_dir, cfg)

    # Random Forest est beaucoup plus robuste que Logistic Regression
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()), 
        ("rf", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced", # Gère le déséquilibre air/hit
            max_depth=15,            # Évite le sur-apprentissage
            n_jobs=-1,               # Utilise tous les coeurs CPU
            random_state=42
        ))
    ])

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    f1s = []
    
    print("Début de la validation croisée (Random Forest)...")
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        score = f1_score(y[te], pred, average="macro")
        f1s.append(score)
        print(f"  Fold {fold+1}: F1-Macro = {score:.4f}")
        
    cv_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"Score F1-Macro Moyen : {cv_f1:.4f}")

    print("Entraînement final...")
    clf.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "model": clf,
        "feature_config": asdict(cfg),
        "feature_names": feat_names,
        "labels": list(LABELS),
    }
    dump(payload, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    counts = np.bincount(y, minlength=len(LABELS))
    return {
        "cv_macro_f1": cv_f1,
        "n_samples": int(X.shape[0]),
        "class_counts": {LABELS[i]: int(counts[i]) for i in range(len(LABELS))},
        "model_path": str(model_path),
    }

def load_model(model_path: str | Path) -> Dict[str, Any]:
    return load(model_path)

# -------------------------------------------------------------------------
# Inférence
# -------------------------------------------------------------------------
def supervised_hit_bounce_detection(ball_data: Dict[str, Any],
                                    model_path: str | Path,
                                    cfg_override: Optional[FeatureConfig] = None) -> Dict[str, Any]:
    try:
        payload = load_model(model_path)
    except FileNotFoundError:
        print(f"Erreur : Modèle {model_path} introuvable.")
        return ball_data

    labels = payload["labels"]
    model = payload["model"]
    cfg = FeatureConfig(**payload["feature_config"])
    if cfg_override: cfg = cfg_override

    frames, xs, ys_, vis, _ = io_utils.extract_series(ball_data)
    kin = compute_kinematics(frames, xs, ys_, vis, cfg)
    X, _ = make_frame_features(kin, cfg)

    # 1. Prédiction brute (par frame)
    pred_ids = model.predict(X)
    raw_labels = [labels[int(i)] for i in pred_ids]
    
    # 2. Post-Processing (Regroupement des événements)
    # C'est ce qui va transformer "200 hits" en "5 hits"
    cleaned_labels = clean_predictions(raw_labels)
    
    # On remet "air" si non visible
    final_labels = []
    mask_vis = np.asarray(vis, dtype=bool)
    for i, label in enumerate(cleaned_labels):
        if not mask_vis[i]:
            final_labels.append("air")
        else:
            final_labels.append(label)

    out = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])
        d["pred_action"] = final_labels[i]
        d["x_clean"] = xs[i]
        d["y_clean"] = ys_[i]
        out[str(fr)] = d
        
    return out

# -------------------------------------------------------------------------
# Main : Test Rapide
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    TRAIN_FOLDER = "Data hit & bounce/per_point_v2" 
    MODEL_FILE = "models/tennis_event_classifier.joblib"
    
    if os.path.exists(TRAIN_FOLDER):
        print("=== MODE ENTRAÎNEMENT ===")
        train_supervised(TRAIN_FOLDER, MODEL_FILE)
        
        print("\n=== MODE TEST (sur un fichier) ===")
        test_file = os.path.join(TRAIN_FOLDER, "ball_data_230.json")
        if os.path.exists(test_file):
            data = io_utils.load_ball_json(test_file)
            res = supervised_hit_bounce_detection(data, MODEL_FILE)
            
            hits = [k for k, v in res.items() if v['pred_action'] == 'hit']
            bounces = [k for k, v in res.items() if v['pred_action'] == 'bounce']
            print(f"Fichier {test_file} :")
            print(f"  -> {len(hits)} Hits (événements uniques)")
            print(f"  -> {len(bounces)} Bounces (événements uniques)")