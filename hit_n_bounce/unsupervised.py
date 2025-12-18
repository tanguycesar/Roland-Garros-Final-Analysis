from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import data_loader as io_utils
import features as feat_utils

# ======================================================
# LOGIQUE DE DÉTECTION PHYSIQUE (UNITÉS MÉTRIQUES)
# ======================================================
def detect_tennis_events(
    frames: List[int],
    kin: Dict[str, np.ndarray],
    fps: float
) -> np.ndarray:
    """
    Détection basée sur la physique réelle du court (mètres, m/s^3).
    """
    n = len(frames)
    actions = np.array(["air"] * n, dtype=object)

    # Récupération des features en mètres (via Camera_Params_Distorted.npz)
    ym = kin["ym"]          # Position profondeur (m)
    jerk = kin["jerk"]      # Secousse physique (m/s^3)
    turn_rate = kin["turn_rate"] # Changement de direction (rad/s)
    vy = kin["vy"]          # Vitesse de profondeur (m/s)
    
    # --- SEUILS PHYSIQUES RÉELS ---
    # En m/s^3, ces seuils sont universels (haut ou bas du court)
    hit_jerk_thr = np.nanpercentile(jerk, 97.5) 
    bounce_jerk_thr = np.nanpercentile(jerk, 88) 
    
    # Zones du court (Dimensions Roland-Garros)
    # ym ~ 11.88 m (Bas) | ym ~ -11.88 m (Haut) | ym ~ 0 (Filet)
    baseline_limit = 10.0 # On considère qu'un rebond est près des lignes de fond
    service_line = 6.4

    # État pour l'alternance (Machine à États)
    last_action = "none"
    last_idx = -100
    cooldown = int(0.5 * fps) # 0.5 seconde entre deux coups

    for i in range(2, n - 2):
        if np.isnan(ym[i]): continue

        # 1. DÉTECTION DU HIT (FRARE) - LE PLUS GROS JERK
        # Si le Jerk est extrême et que c'est un maximum local
        if jerk[i] >= hit_jerk_thr and jerk[i] == np.nanmax(jerk[i-2:i+3]):
            
            # Règle d'alternance : Pas deux hits d'affilée sans temps de vol
            if last_action != "hit" or (i - last_idx) > cooldown:
                actions[i] = "hit"
                last_action = "hit"
                last_idx = i
                continue

        # 2. DÉTECTION DU BOUNCE (REBOND)
        # Un rebond sur le terrain en mètres crée aussi une rupture de Jerk
        # et se produit souvent dans des zones spécifiques (baselines / service)
        if last_action == "hit" and (i - last_idx) > (0.2 * fps): # Un rebond vient après une frappe
            
            # Un rebond se détecte par un Jerk significatif 
            # ET souvent une inversion ou déviation de la trajectoire au sol
            if jerk[i] > bounce_jerk_thr and jerk[i] == np.nanmax(jerk[i-1:i+2]):
                
                # Check de zone : est-on dans une zone de rebond probable ?
                is_near_lines = (abs(ym[i]) > 5.0) # Baselines ou lignes de service
                
                if is_near_lines:
                    if (i - last_idx) > (0.3 * fps): # Temps de vol minimal
                        actions[i] = "bounce"
                        last_action = "bounce"
                        last_idx = i

    return actions

# ======================================================
# PIPELINE ET VISUALISATION
# ======================================================
def run_unsupervised_pipeline(json_path: str, cfg: feat_utils.FeatureConfig):
    print(f"--- Analyse : {os.path.basename(json_path)} ---")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_json_data = json.load(f)
    
    # 1. Loader (PCHIP + Calibration Distorsion)
    frames, xs, ys, vis = io_utils.extract_series(raw_json_data)
    
    # 2. Features (Conversion PIXELS -> METRES automatique)
    kin = feat_utils.compute_kinematics(frames, np.array(xs), np.array(ys), cfg)

    # 3. Détection sur les données métriques
    pred_actions = detect_tennis_events(frames, kin, cfg.fps)

    results = {str(fr): {"x": xs[i], "y": ys[i], "x_m": kin["xm"][i], "y_m": kin["ym"][i], "pred_action": pred_actions[i]} 
               for i, fr in enumerate(frames)}

    return frames, results, kin

def visualize_results(frames, kin, results):
    ym = kin["ym"] # On affiche en mètres pour vérifier la physique
    actions = [results[str(f)]["pred_action"] for f in frames]

    plt.figure(figsize=(16, 7))
    plt.plot(frames, ym, color='black', lw=1, alpha=0.5, label="Trajectoire Terrain (m)")
    
    # Repères du court (Lignes de fond et filet)
    plt.axhline(11.88, color='red', ls='--', alpha=0.3, label="Ligne de fond Bas")
    plt.axhline(-11.88, color='red', ls='--', alpha=0.3, label="Ligne de fond Haut")
    plt.axhline(0, color='blue', ls=':', alpha=0.3, label="Filet")

    hit_f = [f for f, a in zip(frames, actions) if a == "hit"]
    hit_y = [y for y, a in zip(ym, actions) if a == "hit"]
    bounce_f = [f for f, a in zip(frames, actions) if a == "bounce"]
    bounce_y = [y for y, a in zip(ym, actions) if a == "bounce"]
    
    plt.scatter(hit_f, hit_y, color='limegreen', marker='*', s=250, label='HIT', zorder=5)
    plt.scatter(bounce_f, bounce_y, color='orange', marker='o', s=120, label='BOUNCE', zorder=5)

    plt.title("Détection Non Supervisée : Physique Réelle (Mètres)")
    plt.ylabel("Profondeur du Court (mètres)")
    plt.xlabel("Frames")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    my_cfg = feat_utils.FeatureConfig(fps=50.0)
    input_file = "Data hit & bounce/per_point_v2/ball_data_230.json"

    if os.path.exists(input_file):
        fr, res, kin = run_unsupervised_pipeline(input_file, my_cfg)
        visualize_results(fr, kin, res)