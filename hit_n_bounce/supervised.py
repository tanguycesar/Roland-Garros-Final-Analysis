from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Modèle : XGBoost ou HistGradientBoosting
# =========================
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    XGBClassifier = None 

from sklearn.ensemble import HistGradientBoostingClassifier

# =========================
# Imports projet
# =========================
import data_loader as io_utils
from features import FeatureConfig, compute_kinematics

LABELS = ("air", "hit", "bounce")
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {i: k for k, i in LABEL_TO_ID.items()}


# -------------------------------------------------------------------------
# Feature Engineering (Version Physique Métrique)
# -------------------------------------------------------------------------
def make_frame_features(kin: Dict[str, np.ndarray], cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Crée des features basées sur la physique réelle du court.
   
    """
    ym = np.asarray(kin["ym"], float)      # Profondeur (m)
    vy = np.asarray(kin["vy"], float)      # Vitesse verticale (m/s)
    ay = np.asarray(kin["ay"], float)      # Accélération (m/s²)
    jerk = np.asarray(kin["jerk"], float)  # Jerk (m/s³)
    turn = np.asarray(kin["turn_rate"], float)
    speed = np.asarray(kin["speed"], float)

    # Contexte spatial et directionnel
    dist_baseline = np.abs(np.abs(ym) - 11.88)
    flip_vy = np.zeros_like(vy)
    flip_vy[1:] = (np.sign(vy[1:]) != np.sign(vy[:-1])).astype(float)

    signals = {
        "ym": ym, "vy": vy, "ay": ay, "jk": jerk,
        "tn": turn, "sp": speed, "db": dist_baseline, "fv": flip_vy
    }

    feature_arrays, feature_names = [], []
    w = 3 # Fenêtre de 7 frames au total
    for name, arr in signals.items():
        arr_clean = np.nan_to_num(arr, nan=0.0)
        for shift in range(-w, w + 1):
            col = np.roll(arr_clean, shift)
            if shift > 0: col[:shift] = 0.0
            elif shift < 0: col[shift:] = 0.0
            feature_arrays.append(col)
            feature_names.append(f"{name}_{shift}")

    return np.stack(feature_arrays, axis=1), feature_names


# -------------------------------------------------------------------------
# Dataset & Modèle
# -------------------------------------------------------------------------
def _make_dataset(points_dir: str | Path, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    files = io_utils.iter_point_files(points_dir)
    Xs, ys, groups = [], [], []
    feat_names = None

    print(f"Chargement + feature engineering sur {len(files)} fichiers...")

    for gi, fp in enumerate(files):
        data = io_utils.load_ball_json(fp)
        frames, xs, ys_px, vis, acts = io_utils.extract_series(data)
        
        # Calcul cinématique en mètres
        kin = compute_kinematics(frames, np.asarray(xs, float), np.asarray(ys_px, float), cfg)
        X_all, names = make_frame_features(kin, cfg)
        
        if feat_names is None: feat_names = names

        mask = np.asarray(vis, dtype=bool)
        if mask.sum() == 0: continue

        Xs.append(X_all[mask])
        ys.append(np.array([LABEL_TO_ID.get(a, 0) for (a, m) in zip(acts, mask) if m], dtype=int))
        groups.append(np.full(mask.sum(), gi, dtype=int))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(groups, axis=0)

    print(f"Dataset: {X.shape[0]} frames, {len(np.unique(g))} points uniques.")
    return X, y, g, feat_names or []

def _build_model() -> Pipeline:
    if _HAS_XGB:
        clf = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, 
                            objective="multi:softprob", num_class=3, tree_method="hist", random_state=42)
    else:
        clf = HistGradientBoostingClassifier(max_iter=400, random_state=42)
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", clf)])


# -------------------------------------------------------------------------
# Post-processing & Visualisation
# -------------------------------------------------------------------------
def _events_from_probs(probs: np.ndarray, fps: float) -> List[str]:
    n = probs.shape[0]
    out = ["air"] * n
    p_hit, p_bnc = probs[:, LABEL_TO_ID["hit"]], probs[:, LABEL_TO_ID["bounce"]]
    
    # Seuils basés sur les percentiles pour la robustesse
    thr_hit = float(np.nanpercentile(p_hit, 96))
    thr_bnc = float(np.nanpercentile(p_bnc, 93))

    last_idx, cooldown = -100, int(0.3 * fps)
    for i in range(n):
        if p_hit[i] > thr_hit and p_hit[i] == np.max(p_hit[max(0, i-2):i+3]):
            if (i - last_idx) > cooldown:
                out[i], last_idx = "hit", i
        elif p_bnc[i] > thr_bnc and p_bnc[i] == np.max(p_bnc[max(0, i-2):i+3]):
            if (i - last_idx) > cooldown:
                out[i], last_idx = "bounce", i
    return out

def visualize_dashboard(frames, kin, probs, final_actions):
    plt.close('all')
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    ym = kin["ym"]
    axes[0].plot(frames, ym, color='black', alpha=0.4)
    axes[0].axhline(11.88, color='red', ls=':', alpha=0.5); axes[0].axhline(-11.88, color='red', ls=':', alpha=0.5)
    
    # Affichage des prédictions
    for i, a in enumerate(final_actions):
        if a == "hit": axes[0].scatter(frames[i], ym[i], color='green', marker='*', s=200)
        if a == "bounce": axes[0].scatter(frames[i], ym[i], color='orange', marker='o', s=100)
    
    axes[0].set_ylabel("Y (mètres)"); axes[0].invert_yaxis(); axes[0].set_title("Trajectoire & Détections")
    
    # Courbes de confiance
    axes[1].plot(frames, probs[:, LABEL_TO_ID["hit"]], color='green', label="Prob Hit")
    axes[1].plot(frames, probs[:, LABEL_TO_ID["bounce"]], color='orange', label="Prob Bounce")
    axes[1].legend(); axes[1].set_title("Confiance de l'IA")
    
    # Jerk métrique
    axes[2].plot(frames, kin["jerk"], color='red', lw=1)
    axes[2].set_ylim(0, np.nanpercentile(kin["jerk"], 98)*3); axes[2].set_title("Signal de Choc (Jerk)")
    
    plt.tight_layout(); plt.show()


# -------------------------------------------------------------------------
# Entraînement Final
# -------------------------------------------------------------------------
def train_supervised(points_dir: str | Path, model_path: str | Path, cfg: FeatureConfig):
    X, y, groups, feat_names = _make_dataset(points_dir, cfg)
    model = _build_model()

    # Pondération pour gérer le déséquilibre
    sw = np.ones_like(y, dtype=float)
    sw[y == LABEL_TO_ID["hit"]] = 4.0
    sw[y == LABEL_TO_ID["bounce"]] = 2.5

    gkf = GroupKFold(n_splits=5)
    f1s = []

    print("\n--- Validation Croisée (GroupKFold) ---")
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        model.fit(X[tr], y[tr], clf__sample_weight=sw[tr] if _HAS_XGB else None)
        y_pred = model.predict(X[te])
        score = f1_score(y[te], y_pred, average="macro")
        f1s.append(score)
        print(f"Pli {fold}: F1-Macro = {score:.4f}")

    print(f"\nScore F1-Macro moyen: {np.mean(f1s):.4f}")
    
    print("\n--- Entraînement Final ---")
    model.fit(X, y, clf__sample_weight=sw if _HAS_XGB else None)
    print(classification_report(y, model.predict(X), target_names=LABELS))

    dump({"model": model, "feature_config": asdict(cfg), "labels": list(LABELS)}, model_path)
    print(f"Modèle sauvegardé : {model_path}")


if __name__ == "__main__":
    TRAIN_DIR = r"c:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2"
    MODEL_FILE = "models/tennis_event_classifier.joblib"
    
    cfg = FeatureConfig()
    
    # 1. Entraîner
    train_supervised(TRAIN_DIR, MODEL_FILE, cfg)

    # 2. Tester et Visualiser un fichier
    test_path = Path(TRAIN_DIR) / "ball_data_230.json"
    if test_path.exists():
        payload = load(MODEL_FILE)
        model = payload["model"]
        
        with open(test_path, "r") as f: data = io_utils.load_ball_json(test_path)
        frames, xs, ys, vis, _ = io_utils.extract_series(data)
        kin = compute_kinematics(frames, np.asarray(xs, float), np.asarray(ys, float), cfg)
        X, _ = make_frame_features(kin, cfg)
        
        probs = model.predict_proba(X)
        final_actions = _events_from_probs(probs, cfg.fps)
        
        visualize_dashboard(frames, kin, probs, final_actions)