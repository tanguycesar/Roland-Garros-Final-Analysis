from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from joblib import dump, load

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Modèle "plus complexe"
# =========================
# Recommandé : XGBoost (souvent meilleur que RF sur ce problème)
# Si tu n'as pas xgboost installé : pip install xgboost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    XGBClassifier = None  # type: ignore

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
# Feature Engineering (fenêtre glissante + features "physiques")
# -------------------------------------------------------------------------
def make_frame_features(kin: Dict[str, np.ndarray], cfg: FeatureConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Features par frame avec fenêtre glissante, robustes pour modèles tabulaires.

    Signaux clés:
      - vy, ay (vertical) -> rebond/hit
      - speed, turn_rate -> hit (changement direction + énergie)
      - ground_dist -> rebond (proche du sol)
      - jerk_y (d(ay)/dt) -> impacts brutaux
      - sign_vy + flip_vy -> inversions (signature d'événement)
    """
    vy = np.asarray(kin["vy"], float)
    ay = np.asarray(kin["ay"], float)
    speed = np.asarray(kin["speed"], float)
    turn = np.asarray(kin["turn_rate"], float)
    y_pos = np.asarray(kin["ys"], float)
    ground_y = float(np.asarray(kin["ground_y"]).ravel()[0])

    ground_dist = ground_y - y_pos  # petit => proche sol
    jerk_y = np.gradient(np.nan_to_num(ay, nan=0.0))

    sign_vy = np.sign(np.nan_to_num(vy, nan=0.0))
    flip_vy = np.zeros_like(sign_vy)
    flip_vy[1:] = (sign_vy[1:] != sign_vy[:-1]).astype(float)

    # Fenêtre glissante
    w = 3  # un peu plus large que ton RF (2) => mieux pour XGB
    signals = {
        "vy": vy,
        "ay": ay,
        "sp": speed,
        "tn": turn,
        "gd": ground_dist,
        "jy": jerk_y,
        "sv": sign_vy,
        "fv": flip_vy,
    }

    feature_arrays: List[np.ndarray] = []
    feature_names: List[str] = []

    for name, arr in signals.items():
        arr_clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        for shift in range(-w, w + 1):
            col = np.roll(arr_clean, shift)
            if shift > 0:
                col[:shift] = 0.0
            elif shift < 0:
                col[shift:] = 0.0
            feature_arrays.append(col)
            suffix = f"{shift}" if shift < 0 else f"+{shift}"
            feature_names.append(f"{name}_{suffix}")

    X = np.stack(feature_arrays, axis=1)
    return X, feature_names


# -------------------------------------------------------------------------
# Création Dataset (frames visibles uniquement)
# -------------------------------------------------------------------------
def _make_dataset(points_dir: str | Path, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    files = io_utils.iter_point_files(points_dir)

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    feat_names: Optional[List[str]] = None

    print(f"Chargement + feature engineering sur {len(files)} fichiers...")

    for gi, fp in enumerate(files):
        data = io_utils.load_ball_json(fp)
        frames, xs, ys_clean, vis, acts = io_utils.extract_series(data)

        # compute_kinematics(frames, xs, ys, cfg)  (signature attendue)
        kin = compute_kinematics(frames, np.asarray(xs, float), np.asarray(ys_clean, float), cfg)

        X_all, names = make_frame_features(kin, cfg)
        if feat_names is None:
            feat_names = names

        mask = np.asarray(vis, dtype=bool)
        if mask.sum() == 0:
            continue

        X_subset = X_all[mask]
        y_subset = np.array([LABEL_TO_ID.get(a, 0) for (a, m) in zip(acts, mask) if m], dtype=int)

        if len(y_subset) == 0:
            continue

        Xs.append(X_subset)
        ys.append(y_subset)
        groups.append(np.full_like(y_subset, gi, dtype=int))

    if not Xs:
        raise ValueError("Aucune donnée valide trouvée dans le dossier (après filtre visible).")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(groups, axis=0)

    print(f"Dataset final: {X.shape[0]} frames, {X.shape[1]} features.")
    return X, y, g, feat_names or []


# -------------------------------------------------------------------------
# Modèle
# -------------------------------------------------------------------------
def _build_model() -> Pipeline:
    """
    Modèle conseillé :
      - XGBClassifier (si dispo) sinon HistGradientBoostingClassifier
    """
    if _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="multi:softprob",
            num_class=len(LABELS),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
    else:
        # fallback solide (souvent moins bon que XGB, mais correct)
        clf = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_depth=8,
            max_iter=500,
            random_state=42,
        )

    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# -------------------------------------------------------------------------
# Post-processing : transformer frames -> événements plausibles
# -------------------------------------------------------------------------
def _events_from_probs(
    probs: np.ndarray,
    vis_mask: np.ndarray,
    fps: float,
    min_gap_bounce: int = 10,
    min_gap_hit: int = 6,
    hit_after_bounce_window: int = 12,
) -> List[str]:
    """
    On part des probabilités frame-level (N, 3).
    Objectif : sortir une suite propre avec des événements ponctuels.

    Règles :
      - un événement = 1 frame (on prend le pic de proba)
      - pas 2 bounces trop proches
      - si bounce, on cherche un hit rapidement après (souvent), mais on autorise volées (hit sans bounce)
      - hits pas trop proches entre eux
      - si non visible => air
    """
    n = probs.shape[0]
    out = ["air"] * n

    # scores
    p_air = probs[:, LABEL_TO_ID["air"]]
    p_hit = probs[:, LABEL_TO_ID["hit"]]
    p_bnc = probs[:, LABEL_TO_ID["bounce"]]

    # candidats par seuils adaptatifs (percentiles => robustes au point)
    hit_thr = float(np.nanpercentile(p_hit, 92))
    bnc_thr = float(np.nanpercentile(p_bnc, 90))

    hit_cand = np.where((p_hit >= hit_thr) & vis_mask)[0].tolist()
    bnc_cand = np.where((p_bnc >= bnc_thr) & vis_mask)[0].tolist()

    # NMS simple 1D
    def nms(peaks: List[int], score: np.ndarray, radius: int) -> List[int]:
        if not peaks:
            return []
        peaks_sorted = sorted(peaks, key=lambda i: float(score[i]), reverse=True)
        picked: List[int] = []
        blocked = np.zeros(n, dtype=bool)
        for i in peaks_sorted:
            if blocked[i]:
                continue
            picked.append(i)
            lo = max(0, i - radius)
            hi = min(n, i + radius + 1)
            blocked[lo:hi] = True
        return sorted(picked)

    bounces = nms(bnc_cand, p_bnc, radius=max(2, min_gap_bounce // 2))
    hits = nms(hit_cand, p_hit, radius=max(2, min_gap_hit // 2))

    # Filtre : pas 2 bounces trop proches
    b_final: List[int] = []
    for b in bounces:
        if not b_final or (b - b_final[-1]) >= min_gap_bounce:
            b_final.append(b)

    # Filtre hits trop proches
    h_final: List[int] = []
    for h in hits:
        if not h_final or (h - h_final[-1]) >= min_gap_hit:
            h_final.append(h)

    # Règle tennis : après un bounce, généralement un hit proche
    # -> on "associe" au mieux un hit après bounce (si existe), sinon on garde bounce seul (fin de point)
    used_hits = set()
    for b in b_final:
        out[b] = "bounce"
        # cherche le hit le plus probable dans une fenêtre après le bounce
        lo = b + 1
        hi = min(n, b + hit_after_bounce_window + 1)
        if lo >= hi:
            continue
        window_hits = [h for h in h_final if lo <= h < hi and h not in used_hits]
        if window_hits:
            # prend celui avec proba hit max
            best = max(window_hits, key=lambda idx: float(p_hit[idx]))
            out[best] = "hit"
            used_hits.add(best)

    # Volées : hits non utilisés (hit sans bounce) => garder seulement les plus sûrs
    # (sinon tu vas sur-détecter)
    volley_thr = float(np.nanpercentile(p_hit, 96))
    for h in h_final:
        if h in used_hits:
            continue
        if p_hit[h] >= volley_thr:
            out[h] = "hit"

    # Non visible => air
    for i in range(n):
        if not vis_mask[i]:
            out[i] = "air"

    return out


# -------------------------------------------------------------------------
# Entraînement
# -------------------------------------------------------------------------
def train_supervised(
    points_dir: str | Path,
    model_path: str | Path,
    cfg: FeatureConfig | None = None,
    n_splits: int = 5,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = FeatureConfig()

    X, y, groups, feat_names = _make_dataset(points_dir, cfg)
    model = _build_model()

    # pondération simple : on pousse les classes rares
    # (air domine très largement)
    sw = np.ones_like(y, dtype=float)
    sw[y == LABEL_TO_ID["hit"]] = 3.0
    sw[y == LABEL_TO_ID["bounce"]] = 2.0

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    f1s: List[float] = []

    print("Validation croisée (GroupKFold)...")
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        model.fit(X[tr], y[tr], clf__sample_weight=sw[tr] if _HAS_XGB else None)  # type: ignore

        pred = model.predict(X[te])
        score = f1_score(y[te], pred, average="macro")
        f1s.append(float(score))
        print(f"  Fold {fold}: F1-Macro = {score:.4f}")

    cv_f1 = float(np.mean(f1s)) if f1s else 0.0
    print(f"Score F1-Macro moyen : {cv_f1:.4f}")

    print("Entraînement final...")
    model.fit(X, y, clf__sample_weight=sw if _HAS_XGB else None)  # type: ignore

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_config": asdict(cfg),
        "feature_names": feat_names,
        "labels": list(LABELS),
        "has_xgb": _HAS_XGB,
    }
    dump(payload, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    counts = np.bincount(y, minlength=len(LABELS))
    return {
        "cv_macro_f1": cv_f1,
        "n_samples": int(X.shape[0]),
        "class_counts": {LABELS[i]: int(counts[i]) for i in range(len(LABELS))},
        "model_path": str(model_path),
        "model_type": "XGBClassifier" if _HAS_XGB else "HistGradientBoostingClassifier",
    }


def load_model(model_path: str | Path) -> Dict[str, Any]:
    return load(model_path)


# -------------------------------------------------------------------------
# Inférence
# -------------------------------------------------------------------------
def supervised_hit_bounce_detection(
    ball_data: Dict[str, Any],
    model_path: str | Path,
    cfg_override: Optional[FeatureConfig] = None,
) -> Dict[str, Any]:
    payload = load_model(model_path)
    labels: List[str] = payload["labels"]
    model: Pipeline = payload["model"]
    cfg = FeatureConfig(**payload["feature_config"])
    if cfg_override is not None:
        cfg = cfg_override

    frames, xs, ys_clean, vis, _ = io_utils.extract_series(ball_data)
    kin = compute_kinematics(frames, np.asarray(xs, float), np.asarray(ys_clean, float), cfg)
    X, _ = make_frame_features(kin, cfg)

    vis_mask = np.asarray(vis, dtype=bool)

    # Probabilités
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        # fallback (rare) : one-hot des prédictions
        pred = model.predict(X)
        probs = np.zeros((len(pred), len(LABELS)), dtype=float)
        probs[np.arange(len(pred)), pred.astype(int)] = 1.0

    # Post-processing "physique" / cohérence tennis
    final_labels = _events_from_probs(
        probs=probs,
        vis_mask=vis_mask,
        fps=cfg.fps,
        min_gap_bounce=int(0.35 * cfg.fps),  # ~9 frames
        min_gap_hit=int(0.25 * cfg.fps),     # ~6 frames
        hit_after_bounce_window=int(0.45 * cfg.fps),  # ~11 frames
    )

    out: Dict[str, Any] = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])
        d["pred_action"] = final_labels[i]
        d["x_clean"] = float(xs[i]) if xs[i] is not None else np.nan
        d["y_clean"] = float(ys_clean[i]) if ys_clean[i] is not None else np.nan
        out[str(fr)] = d

    return out


# -------------------------------------------------------------------------
# Main : train + test rapide
# -------------------------------------------------------------------------
if __name__ == "__main__":
    TRAIN_FOLDER = "Data hit & bounce/per_point_v2"
    MODEL_FILE = "models/tennis_event_classifier.joblib"

    if not Path(TRAIN_FOLDER).exists():
        print(f"Dossier introuvable: {TRAIN_FOLDER}")
        raise SystemExit(1)

    print("=== MODE ENTRAÎNEMENT ===")
    info = train_supervised(TRAIN_FOLDER, MODEL_FILE, cfg=FeatureConfig(), n_splits=5)
    print("\nRésumé entraînement:", info)

    print("\n=== MODE TEST (sur un fichier) ===")
    test_file = Path(TRAIN_FOLDER) / "ball_data_230.json"
    if test_file.exists():
        data = io_utils.load_ball_json(test_file)
        res = supervised_hit_bounce_detection(data, MODEL_FILE)

        hits = [k for k, v in res.items() if v.get("pred_action") == "hit"]
        bounces = [k for k, v in res.items() if v.get("pred_action") == "bounce"]

        print(f"Fichier {test_file.name}:")
        print(f"  -> {len(hits)} hits (événements)")
        print(f"  -> {len(bounces)} bounces (événements)")
    else:
        print(f"Fichier test introuvable: {test_file}")
