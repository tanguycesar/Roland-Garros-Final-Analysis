from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import data_loader as io_utils
import features as feat_utils

# ======================================================
# LOGIQUE DE DÉTECTION ASYMÉTRIQUE (TOP vs BOTTOM)
# ======================================================
def detect_tennis_events(
    frames: List[int],
    kin: Dict[str, np.ndarray],
    fps: float
) -> np.ndarray:
    n = len(frames)
    actions = np.array(["air"] * n, dtype=object)

    ym = kin["ym"]         # Profondeur réelle (m)
    vy = kin["vy"]         # Vitesse profondeur (m/s)
    jerk = kin["jerk"]     # Secousse (m/s^3)
    turn = kin["turn_rate"] # Virage (rad/s)
    speed = kin["speed"]    # Vitesse (m/s)
    
    # Seuils statistiques pour confirmer les chocs
    jerk_thr = np.nanpercentile(jerk, 90)
    turn_thr = np.nanpercentile(turn, 90)

    # --- 1. TROUVER LES PIVOTS (Changements de direction Vy) ---
    pivots = []
    for i in range(5, n - 5):
        if np.isnan(vy[i-1]) or np.isnan(vy[i+1]): continue
        if vy[i-1] * vy[i+1] < 0: # Inversion de direction verticale
            pivots.append(i)

    last_action_idx = -100
    cooldown = int(0.5 * fps)

    for idx in pivots:
        # Calcul des durées de vol (stabilité de la direction)
        count_pre = 0
        for j in range(idx - 1, 1, -1):
            if np.isnan(vy[j]) or (vy[j] * vy[idx-1] < 0): break
            count_pre += 1
            
        count_post = 0
        for j in range(idx + 1, n - 1):
            if np.isnan(vy[j]) or (vy[j] * vy[idx+1] < 0): break
            count_post += 1

        # --------------------------------------------------
        # CAS A : JOUEUR DU HAUT (ym < 0) - "Subtilité & Apex"
        # --------------------------------------------------
        if ym[idx] < -1.0:
            # Frappe : Balle monte (-Vy), atteint son point le plus haut/profond, 
            # puis est frappée et redescend brutalement (+Vy pendant longtemps)
            is_top_hit = (vy[idx-1] < 0 and vy[idx+1] > 0) and count_post > 15
            
            if is_top_hit and (jerk[idx] > jerk_thr or turn[idx] > turn_thr):
                if (idx - last_action_idx) > cooldown:
                    actions[idx] = "hit"
                    last_action_idx = idx
                    continue

        # --------------------------------------------------
        # CAS B : JOUEUR DU BAS (ym > 0) - "Rebond puis Frappe"
        # --------------------------------------------------
        else:
            # 1. Rebond (Bounce) : Petite remontée locale (inversion de Vy courte)
            # Souvent : la balle tombe (+Vy), rebondit (-Vy court), puis est frappée
            is_bounce = (vy[idx-1] > 0 and vy[idx+1] < 0) and count_post < 12
            
            if is_bounce and abs(ym[idx]) > 5.0: # Proche fond de court ou service
                actions[idx] = "bounce"
                last_action_idx = idx
                continue

            # 2. Frappe (Hit) : Point le plus bas de la baisse locale
            # La balle redescend après le rebond (+Vy) et change de sens (-Vy long)
            is_bottom_hit = (vy[idx-1] > 0 and vy[idx+1] < 0) and count_post > 15
            
            if is_bottom_hit and (jerk[idx] > jerk_thr or turn[idx] > turn_thr):
                if (idx - last_action_idx) > cooldown:
                    actions[idx] = "hit"
                    last_action_idx = idx

    # --- 3. POST-TRAITEMENT : LOGIQUE DE SÉQUENCE ---
    # Évite les rebonds orphelins ou les successions impossibles
    for i in range(1, n):
        # Si on a détecté un "bounce" trop proche d'un "hit" (même événement), on garde le hit
        if actions[i] == "hit":
            for j in range(max(0, i-6), i):
                if actions[j] == "bounce": actions[j] = "air"

    return actions

# ======================================================
# PIPELINE DE TRAITEMENT COMPLET
# ======================================================
def run_unsupervised_pipeline(json_path: str, cfg: feat_utils.FeatureConfig):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Extraction PCHIP via data_loader
    frames, xs_px, ys_px, vis, actions = io_utils.extract_series(raw_data)
    
    # Features en Mètres (Jerk, TurnRate, Vy...) via features.py
    kin = feat_utils.compute_kinematics(frames, np.array(xs_px), np.array(ys_px), cfg)
    
    # Détection avec la nouvelle logique asymétrique
    pred_actions = detect_tennis_events(frames, kin, cfg.fps)

    results = {str(fr): {"pred_action": pred_actions[i], "y_m": kin["ym"][i]} 
               for i, fr in enumerate(frames)}

    return frames, results, kin

def visualize_results(frames, kin, results):
    plt.close('all')  # Fermer toutes les figures précédentes
    ym = kin["ym"]
    actions = [results[str(f)]["pred_action"] for f in frames]

    plt.figure(figsize=(16, 8))
    plt.plot(frames, ym, color='black', lw=1.5, alpha=0.6, label="Trajectoire Terrain (m)")
    
    # Lignes de court (Perspective TV : 0 = Filet)
    plt.axhline(11.88, color='red', ls='--', alpha=0.3, label="Baseline Bas")
    plt.axhline(-11.88, color='red', ls='--', alpha=0.3, label="Baseline Haut")
    plt.axhline(0, color='blue', ls=':', alpha=0.2, label="Filet")

    # Unpacking des détections
    hit_f = [f for f, a in zip(frames, actions) if a == "hit"]
    hit_y = [y for y, a in zip(ym, actions) if a == "hit"]
    bounce_f = [f for f, a in zip(frames, actions) if a == "bounce"]
    bounce_y = [y for y, a in zip(ym, actions) if a == "bounce"]
    
    if hit_f:
        plt.scatter(hit_f, hit_y, color='limegreen', marker='*', s=350, label='FRARE (Hit)', zorder=5)
    if bounce_f:
        plt.scatter(bounce_f, bounce_y, color='orange', marker='o', s=150, label='REBOND (Bounce)', zorder=5)

    plt.gca().invert_yaxis() # Important pour la lecture TV (Haut = Fond de court loin)
    plt.title("Détection Non Supervisée : Signature Physique (Apex Haut vs Rebond/Frappe Bas)")
    plt.ylabel("Profondeur sur le terrain (mètres)")
    plt.xlabel("Frames")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    my_cfg = feat_utils.FeatureConfig(fps=50.0)
    input_file = "Data hit & bounce/per_point_v2/ball_data_230.json"

    if os.path.exists(input_file):
        fr, res, kin = run_unsupervised_pipeline(input_file, my_cfg)
        visualize_results(fr, kin, res)