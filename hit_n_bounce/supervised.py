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
from sklearn.linear_model import LogisticRegression

from .features import FeatureConfig, compute_kinematics, make_frame_features
from .io import iter_point_files, load_ball_json


LABELS = ("air", "hit", "bounce")
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}
ID_TO_LABEL = {i: k for k, i in LABEL_TO_ID.items()}


def _make_dataset(points_dir: str | Path, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Build X, y, groups from a folder of point jsons."""
    files = iter_point_files(points_dir)
    Xs, ys, groups = [], [], []
    feat_names = None

    for gi, fp in enumerate(files):
        data = load_ball_json(fp)
        frames = sorted([int(k) for k in data.keys()])
        xs = [float(data[str(fr)].get("x", float("nan"))) for fr in frames]
        ys_ = [float(data[str(fr)].get("y", float("nan"))) for fr in frames]
        vis = [bool(data[str(fr)].get("visible", True)) for fr in frames]
        act = [str(data[str(fr)].get("action", "air")) for fr in frames]

        kin = compute_kinematics(frames, xs, ys_, vis, cfg)
        X, names = make_frame_features(kin, cfg)
        if feat_names is None:
            feat_names = names

        # Keep only visible frames for supervised training (labels are meaningful there)
        mask = np.asarray(vis, dtype=bool)
        X = X[mask]
        y = np.array([LABEL_TO_ID.get(a, 0) for (a, m) in zip(act, mask) if m], dtype=int)

        Xs.append(X)
        ys.append(y)
        groups.append(np.full_like(y, gi, dtype=int))

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    g_all = np.concatenate(groups, axis=0)
    return X_all, y_all, g_all, feat_names or []


def train_supervised(points_dir: str | Path,
                     model_path: str | Path,
                     cfg: FeatureConfig | None = None,
                     n_splits: int = 5) -> Dict[str, Any]:
    """
    Baseline supervised model: Logistic Regression with context-window features.
    - fast
    - robust baseline
    - easy to serialize
    """
    if cfg is None:
        cfg = FeatureConfig()

    X, y, groups, feat_names = _make_dataset(points_dir, cfg)

    # class weights to counter imbalance (hits/bounces rare)
    counts = np.bincount(y, minlength=len(LABELS))
    total = counts.sum()
    class_weight = {i: (total / (len(LABELS) * max(1, counts[i]))) for i in range(len(LABELS))}

    clf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            class_weight=class_weight,
            n_jobs=None
        ))
    ])

    # CV evaluation by point (GroupKFold)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    f1s = []
    for tr, te in gkf.split(X, y, groups=groups):
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred, average="macro"))
    cv_f1 = float(np.mean(f1s)) if f1s else float("nan")

    # fit final model on all data
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

    return {
        "cv_macro_f1": cv_f1,
        "n_samples": int(X.shape[0]),
        "class_counts": {LABELS[i]: int(counts[i]) for i in range(len(LABELS))},
        "model_path": str(model_path),
    }


def load_model(model_path: str | Path) -> Dict[str, Any]:
    return load(model_path)


def supervised_hit_bounce_detection(ball_data: Dict[str, Any],
                                   model_path: str | Path,
                                   cfg_override: Optional[FeatureConfig] = None) -> Dict[str, Any]:
    payload = load_model(model_path)
    labels = payload["labels"]
    model = payload["model"]
    cfg = FeatureConfig(**payload["feature_config"])
    if cfg_override is not None:
        cfg = cfg_override

    frames = sorted([int(k) for k in ball_data.keys()])
    xs = [float(ball_data[str(fr)].get("x", float("nan"))) for fr in frames]
    ys_ = [float(ball_data[str(fr)].get("y", float("nan"))) for fr in frames]
    vis = [bool(ball_data[str(fr)].get("visible", True)) for fr in frames]

    kin = compute_kinematics(frames, xs, ys_, vis, cfg)
    X, _ = make_frame_features(kin, cfg)

    pred_ids = model.predict(X)
    pred_labels = np.array([labels[int(i)] for i in pred_ids], dtype=object)
    # for non-visible frames, set air
    pred_labels[~np.asarray(vis, dtype=bool)] = "air"

    out = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])
        d["pred_action"] = str(pred_labels[i])
        out[str(fr)] = d
    return out
