from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# IMPORTS PROJET
# ======================================================
import data_loader as io_utils   # extract_series()
from features import FeatureConfig


# ======================================================
# DÉTECTION PHYSIQUE Y-ONLY
# ======================================================
def detect_bounce_and_hit_from_Y(
    frames: List[int],
    ys: List[float],
    fps: float = 25.0
) -> np.ndarray:
    """
    Détection non supervisée basée UNIQUEMENT sur Y image

    Bounce :
      - maximum local
      - vy : + -> -
      - ay négatif (courbure)

    Hit :
      - fort pic |ay|
      - après un rebond
    """

    y = np.asarray(ys, dtype=float)
    n = len(y)

    vy = np.gradient(y)
    ay = np.gradient(vy)

    actions = np.array(["air"] * n, dtype=object)

    # ==========================
    # 1. DÉTECTION DES BOUNCES
    # ==========================
    for i in range(3, n - 3):
        if np.isnan(y[i-2:i+3]).any():
            continue

        # MAX local (Y image)
        is_local_max = (
            y[i] > y[i-1] and
            y[i] > y[i+1]
        )

        # Descente -> montée
        vy_flip = (vy[i-1] > 0) and (vy[i+1] < 0)

        # Courbure (rebond arrondi)
        strong_curvature = ay[i] < np.nanpercentile(ay, 20)

        if is_local_max and vy_flip and strong_curvature:
            actions[i] = "bounce"

    bounce_idx = np.where(actions == "bounce")[0]

    # ==========================
    # 2. DÉTECTION DES HITS
    # ==========================
    ay_thr = np.nanpercentile(np.abs(ay), 92)

    for b in bounce_idx:
        start = b + 3
        end = min(b + int(0.25 * fps), n - 2)

        for i in range(start, end):
            if np.isnan(ay[i]):
                continue

            if abs(ay[i]) >= ay_thr:
                actions[i] = "hit"
                break

    return actions

# ======================================================
# PIPELINE COMPLET
# ======================================================
def unsupervised_hit_bounce_detection(
    ball_data: Dict[str, Any],
    cfg: FeatureConfig | None = None
) -> Dict[str, Any]:

    if cfg is None:
        cfg = FeatureConfig()

    # --- extraction + nettoyage (sans labels)
    frames, xs, ys, vis, _ = io_utils.extract_series(ball_data)

    # --- détection Y-only
    actions_pred = detect_bounce_and_hit_from_Y(
        frames=frames,
        ys=ys,
        fps=cfg.fps
    )

    # --- construction de la sortie
    out = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])

        if not bool(d.get("visible", True)):
            d["pred_action"] = "air"
        else:
            d["pred_action"] = actions_pred[i]

        # debug / visu
        d["y_clean"] = ys[i]
        d["x_clean"] = xs[i]

        out[str(fr)] = d

    return out


# ======================================================
# VISUALISATION
# ======================================================
def visualize_results(
    enriched_data: Dict[str, Any],
    title: str = "Unsupervised Y-only Detection"
):
    frames = sorted(int(k) for k in enriched_data.keys())

    ys = np.array([enriched_data[str(f)].get("y_clean", np.nan) for f in frames])
    actions = [enriched_data[str(f)]["pred_action"] for f in frames]

    hit_f, hit_y = [], []
    bounce_f, bounce_y = [], []

    for f, y, a in zip(frames, ys, actions):
        if a == "hit":
            hit_f.append(f)
            hit_y.append(y)
        elif a == "bounce":
            bounce_f.append(f)
            bounce_y.append(y)

    plt.figure(figsize=(16, 6))
    plt.plot(frames, ys, lw=2, label="Y trajectory")
    plt.scatter(hit_f, hit_y, c="green", marker="*", s=120, label="Hit", zorder=5)
    plt.scatter(bounce_f, bounce_y, c="red", marker="^", s=120, label="Bounce", zorder=5)

    plt.gca().invert_yaxis()
    plt.xlabel("Frame")
    plt.ylabel("Pixel Y (image)")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    file_to_test = "Data hit & bounce/per_point_v2/ball_data_236.json"

    if not os.path.exists(file_to_test):
        print("Fichier introuvable.")
        exit()

    print(f"Analyse de {file_to_test}")

    raw_data = io_utils.load_ball_json(file_to_test)

    results = unsupervised_hit_bounce_detection(raw_data)

    n_hits = sum(v["pred_action"] == "hit" for v in results.values())
    n_bounces = sum(v["pred_action"] == "bounce" for v in results.values())

    print(f"Résultats : {n_hits} hits, {n_bounces} bounces")

    visualize_results(results, title=os.path.basename(file_to_test))
