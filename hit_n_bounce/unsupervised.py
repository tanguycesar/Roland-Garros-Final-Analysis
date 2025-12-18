from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import data_loader as io_utils
import features as feat_utils

# ======================================================
# LOGIQUE DE DÉTECTION HYBRIDE AMÉLIORÉE (SÉQUENCE STRICTE)
# ======================================================
def detect_tennis_events(
    frames: List[int],
    kin: Dict[str, np.ndarray],
    fps: float
) -> np.ndarray:
    n = len(frames)
    actions = np.array(["air"] * n, dtype=object)

    ym = kin["ym"]         
    vy = kin["vy"]         
    jerk = kin["jerk"]     
    turn = kin["turn_rate"]
    ay = kin["ay"] 

    # Seuils statistiques
    jerk_thr = np.nanpercentile(jerk, 90)
    turn_thr = np.nanpercentile(turn, 90)
    ay_thr = np.nanpercentile(np.abs(ay), 85)

    # Variables de contrôle de séquence
    last_event_type = "none"
    last_event_idx = -100
    
    # ---------------------------------------------------------------------
    # ÉTAPE 1 : DÉTECTION DES HITS (LES ANCRES DU RALLYE)
    # ---------------------------------------------------------------------
    # On détecte d'abord les hits car ils servent de pivots pour la séquence
    pivots = []
    for i in range(5, n - 5):
        if np.isnan(vy[i-1]) or np.isnan(vy[i+1]): continue
        if vy[i-1] * vy[i+1] < 0:
            pivots.append(i)

    for idx in pivots:
        count_post = 0
        for j in range(idx + 1, min(idx + 50, n - 1)):
            if np.isnan(vy[j]) or (vy[j] * vy[idx+1] < 0): break
            count_post += 1

        if count_post > 15:
            if (jerk[idx] > jerk_thr * 0.8 or turn[idx] > turn_thr):
                actions[idx] = "hit"

    # ---------------------------------------------------------------------
    # ÉTAPE 2 : DÉTECTION DES REBONDS AVEC RÈGLES DE SÉQUENCE
    # ---------------------------------------------------------------------
    # On repasse pour valider les rebonds en respectant les interdits
    # - Pas de rebond après un hit sans temps de vol
    # - Pas de rebond après un rebond sans hit entre les deux
    can_detect_bounce = True # Autorisé au début du point
    
    for i in range(3, n - 3):
        if np.isnan(ym[i]): continue
        
        # Si on croise un HIT détecté à l'étape 1, on met à jour l'état
        if actions[i] == "hit":
            last_event_type = "hit"
            last_event_idx = i
            can_detect_bounce = True # Un hit autorise à nouveau un rebond
            continue

        # Logique de détection du rebond
        near_ground = abs(ym[i]) > 4.5
        is_choc = (jerk[i] > jerk_thr * 0.8) and (abs(ay[i]) > ay_thr)
        
        if near_ground and is_choc and can_detect_bounce:
            # Vérification du max local
            if jerk[i] == np.nanmax(jerk[i-2:i+3]):
                
                # CONDITION : Pas juste après un hit (min 0.2s de vol)
                if last_event_type == "hit" and (i - last_event_idx) < int(0.2 * fps):
                    continue
                
                actions[i] = "bounce"
                last_event_type = "bounce"
                last_event_idx = i
                can_detect_bounce = False # INTERDIT de détecter un autre rebond avant le prochain HIT

    # ---------------------------------------------------------------------
    # PARTIE 3 : NETTOYAGE FINAL
    # ---------------------------------------------------------------------
    for i in range(1, n - 1):
        if actions[i] == "hit":
            for j in range(max(0, i-6), i):
                if actions[j] == "bounce" and (i - j) < 4:
                    actions[j] = "air"

    return actions

# ======================================================
# PIPELINE ET VISUALISATION
# ======================================================
def run_hybrid_pipeline(json_path: str, cfg: feat_utils.FeatureConfig):
    # Forcer la fermeture pour mettre à jour le graphique
    plt.close('all')
    
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    frames, xs_px, ys_px, vis, _ = io_utils.extract_series(raw_data)
    kin = feat_utils.compute_kinematics(frames, np.array(xs_px), np.array(ys_px), cfg)
    # garder aussi les coordonnées en pixels pour fallback visuel
    kin["xs_px"] = np.array(xs_px)
    kin["ys_px"] = np.array(ys_px)
    
    pred_actions = detect_tennis_events(frames, kin, cfg.fps)

    results = {str(fr): {"pred_action": pred_actions[i], "y_m": kin["ym"][i]} 
               for i, fr in enumerate(frames)}

    return frames, results, kin

def visualize_results(frames, kin, results):
    ym = kin.get("ym", np.array([np.nan] * len(frames)))
    ys_px = kin.get("ys_px", None)
    actions = [results[str(f)]["pred_action"] for f in frames]

    non_nan_idx = np.where(~np.isnan(ym))[0]
    frac_non_nan = len(non_nan_idx) / max(1, len(frames))

    # Fallback vers pixels si peu de valeurs en mètres
    if len(non_nan_idx) == 0 or frac_non_nan < 0.3:
        print(f"⚠️ Peu de valeurs en mètres ({len(non_nan_idx)}/{len(frames)}). Fallback vers pixels.")
        plt.figure(figsize=(16, 8))
        if ys_px is not None:
            plt.plot(frames, ys_px, color='magenta', lw=1.5, alpha=0.8, label='Trajectoire Pixels (px)')
            plt.gca().invert_yaxis()
            plt.title('Trajectoire (fallback pixels)')
            plt.ylabel('Y (px)')
            plt.xlabel('Frame')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            return
        else:
            print("Aucune donnée pixels disponible pour fallback. Affichage en mètres.")

    # Limiter la plage aux frames non-NaN
    start_idx, end_idx = non_nan_idx[0], non_nan_idx[-1]
    sel_frames = frames[start_idx:end_idx + 1]
    sel_ym = ym[start_idx:end_idx + 1]

    plt.figure(figsize=(16, 8))
    plt.plot(sel_frames, sel_ym, color='black', lw=1.5, alpha=0.6, label="Trajectoire Terrain (m)")
    plt.axhline(11.88, color='red', ls='--', alpha=0.3, label="Lignes de fond")
    plt.axhline(-11.88, color='red', ls='--', alpha=0.3)
    plt.axhline(0, color='blue', ls=':', alpha=0.2, label="Filet")

    # Montrer hits/bounces uniquement dans la plage sélectionnée
    actions_sel = actions[start_idx:end_idx + 1]
    hit_f = [f for f, a in zip(sel_frames, actions_sel) if a == "hit"]
    hit_y = [y for y, a in zip(sel_ym, actions_sel) if a == "hit"]
    bounce_f = [f for f, a in zip(sel_frames, actions_sel) if a == "bounce"]
    bounce_y = [y for y, a in zip(sel_ym, actions_sel) if a == "bounce"]

    plt.scatter(hit_f, hit_y, color='limegreen', marker='*', s=350, label='HIT', zorder=5)
    plt.scatter(bounce_f, bounce_y, color='orange', marker='o', s=150, label='BOUNCE', zorder=5)

    plt.gca().invert_yaxis()
    plt.title("Détection Hybride : Séquence Stricte (1 Rebond par Frappe)")
    plt.ylabel("Profondeur Y (mètres)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    POINT_ID = 230 
    my_cfg = feat_utils.FeatureConfig(fps=50.0)
    data_dir = r"c:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2"
    input_file = os.path.join(data_dir, f"ball_data_{POINT_ID}.json")

    if os.path.exists(input_file):
        fr, res, kin = run_hybrid_pipeline(input_file, my_cfg)
        visualize_results(fr, kin, res)