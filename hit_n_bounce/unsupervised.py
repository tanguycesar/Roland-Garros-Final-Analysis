from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import data_loader as io_utils
import features as feat_utils

# ======================================================
# DÉTECTION ENRICHIE AVEC ANALYSE MULTI-SIGNAUX
# ======================================================
def compute_advanced_signals(kin: Dict[str, np.ndarray], fps: float) -> Dict[str, np.ndarray]:
    """Calcule des signaux avancés pour améliorer la détection"""
    xm = kin["xm"]
    ym = kin["ym"]
    vx = kin["vx"]
    vy = kin["vy"]
    ax = kin["ax"]
    ay = kin["ay"]
    jerk = kin["jerk"]
    speed = kin["speed"]
    accel = kin["accel"]
    
    n = len(ym)
    
    # 1. Courbure de trajectoire (changement d'angle)
    curvature = np.full(n, np.nan)
    for i in range(3, n - 3):
        if not (np.isnan(vx[i-1]) or np.isnan(vy[i-1]) or np.isnan(vx[i+1]) or np.isnan(vy[i+1])):
            angle_before = np.arctan2(vy[i-1], vx[i-1])
            angle_after = np.arctan2(vy[i+1], vx[i+1])
            angle_change = np.abs(angle_after - angle_before)
            if angle_change > np.pi:
                angle_change = 2 * np.pi - angle_change
            curvature[i] = angle_change
    
    # 2. Changement relatif de vitesse
    speed_change_rate = np.full(n, np.nan)
    for i in range(1, n):
        if speed[i-1] > 0.1:
            speed_change_rate[i] = (speed[i] - speed[i-1]) / speed[i-1]
    
    # 3. Dérivée du jerk (snap) - changement de troisième ordre
    snap = np.gradient(np.nan_to_num(jerk, nan=0.0))
    
    # 4. Distance au centre du court (indicateur de position)
    dist_from_center = np.sqrt(xm**2 + ym**2)
    
    # 5. Variabilité locale (écart-type glissant)
    window = 5
    jerk_std = np.full(n, np.nan)
    vy_std = np.full(n, np.nan)
    for i in range(window, n - window):
        jerk_std[i] = np.nanstd(jerk[i-window:i+window])
        vy_std[i] = np.nanstd(vy[i-window:i+window])
    
    return {
        "curvature": curvature,
        "speed_change": speed_change_rate,
        "snap": snap,
        "dist_center": dist_from_center,
        "jerk_std": jerk_std,
        "vy_std": vy_std
    }

def detect_tennis_events(
    frames: List[int],
    kin: Dict[str, np.ndarray],
    fps: float
) -> np.ndarray:
    n = len(frames)
    actions = np.array(["air"] * n, dtype=object)

    # Signaux de base
    xm = kin["xm"]
    ym = kin["ym"]
    vx = kin["vx"]
    vy = kin["vy"]
    ax = kin["ax"]
    ay = kin["ay"]
    jerk = kin["jerk"]
    turn = kin["turn_rate"]
    speed = kin["speed"]
    accel = kin["accel"]
    
    # Signaux avancés
    advanced = compute_advanced_signals(kin, fps)
    curvature = advanced["curvature"]
    speed_change = advanced["speed_change"]
    snap = advanced["snap"]
    jerk_std = advanced["jerk_std"]
    
    # Seuils adaptatifs multi-niveaux
    jerk_thr_high = np.nanpercentile(jerk, 92)
    jerk_thr_med = np.nanpercentile(jerk, 85)
    jerk_thr_low = np.nanpercentile(jerk, 75)
    
    turn_thr = np.nanpercentile(turn, 88)
    accel_thr = np.nanpercentile(accel, 85)
    curv_thr = np.nanpercentile(curvature[~np.isnan(curvature)], 80) if np.any(~np.isnan(curvature)) else 0.5
    snap_thr = np.nanpercentile(np.abs(snap), 90)

    # --- 1. DÉTECTION DES PIVOTS AVEC SCORING MULTI-CRITÈRES ---
    pivots = []
    for i in range(5, n - 5):
        if np.isnan(vy[i-1]) or np.isnan(vy[i+1]): continue
        if vy[i-1] * vy[i+1] < 0:  # Inversion de direction verticale
            # Score composite pour qualifier le pivot
            score = 0.0
            
            # Critère 1: Magnitude du jerk
            if not np.isnan(jerk[i]):
                if jerk[i] > jerk_thr_high:
                    score += 3.0
                elif jerk[i] > jerk_thr_med:
                    score += 2.0
                elif jerk[i] > jerk_thr_low:
                    score += 1.0
            
            # Critère 2: Courbure de trajectoire
            if not np.isnan(curvature[i]) and curvature[i] > curv_thr:
                score += 2.0
            
            # Critère 3: Changement de vitesse
            if not np.isnan(turn[i]) and turn[i] > turn_thr:
                score += 1.5
            
            # Critère 4: Snap (dérivée du jerk)
            if not np.isnan(snap[i]) and abs(snap[i]) > snap_thr:
                score += 1.0
            
            # Critère 5: Variabilité locale du jerk
            if not np.isnan(jerk_std[i]) and jerk_std[i] > np.nanpercentile(jerk_std[~np.isnan(jerk_std)], 70):
                score += 1.0
            
            pivots.append((i, score))

    last_action_idx = -100
    cooldown = int(0.4 * fps)  # Réduit pour capturer plus d'événements

    for idx, pivot_score in pivots:
        # Calcul des durées de vol (stabilité de la direction)
        count_pre = 0
        for j in range(idx - 1, max(0, idx - 50), -1):
            if np.isnan(vy[j]) or (vy[j] * vy[idx-1] < 0): break
            count_pre += 1
            
        count_post = 0
        for j in range(idx + 1, min(n, idx + 50)):
            if np.isnan(vy[j]) or (vy[j] * vy[idx+1] < 0): break
            count_post += 1

        # --------------------------------------------------
        # CAS A : JOUEUR DU HAUT (ym < 0) - "Apex & Impact"
        # --------------------------------------------------
        if ym[idx] < -1.0:
            # Hit: Apex suivi d'un changement brusque
            is_top_hit = (vy[idx-1] < 0 and vy[idx+1] > 0) and count_post > 12
            
            # Score amélioré avec distance et position
            hit_confidence = pivot_score
            if abs(ym[idx]) < 3.0:  # Près du filet (volée)
                hit_confidence *= 0.8  # Réduire le seuil
            
            if is_top_hit and hit_confidence >= 3.0:  # Seuil de confiance
                if (idx - last_action_idx) > cooldown:
                    actions[idx] = "hit"
                    last_action_idx = idx
                    continue

        # --------------------------------------------------
        # CAS B : JOUEUR DU BAS (ym > 0) - "Rebond puis Frappe"
        # --------------------------------------------------
        else:
            # Distance du fond de court (adaptative)
            dist_baseline = abs(ym[idx] - 11.88)
            is_near_baseline = dist_baseline < 2.0
            
            # 1. Rebond : Signature courte + près du sol
            is_bounce_candidate = (vy[idx-1] > 0 and vy[idx+1] < 0) and count_post < 15
            
            if is_bounce_candidate:
                bounce_confidence = pivot_score
                
                # Bonus si près du fond de court
                if is_near_baseline:
                    bounce_confidence += 1.5
                
                # Vérifier cohérence spatiale (pas de saut en X)
                spatial_ok = True
                if idx > 0 and not np.isnan(xm[idx-1]):
                    dx = abs(xm[idx] - xm[idx-1])
                    if dx > 1.5:  # Saut > 1.5m
                        spatial_ok = False
                
                if bounce_confidence >= 2.5 and spatial_ok and abs(ym[idx]) > 4.0:
                    actions[idx] = "bounce"
                    last_action_idx = idx
                    continue

            # 2. Frappe : Signature longue + forte
            is_bottom_hit = (vy[idx-1] > 0 and vy[idx+1] < 0) and count_post > 12
            
            if is_bottom_hit:
                hit_confidence = pivot_score
                
                # Ajustement selon position
                if abs(ym[idx]) < 6.0:  # Mi-court
                    hit_confidence *= 1.1
                
                if hit_confidence >= 3.5:
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
    
    # --- 4. RECHERCHE GUIDÉE DE REBONDS AVANT CHAQUE HIT ---
    # Un seul rebond maximum entre deux hits consécutifs
    hit_indices = [i for i in range(n) if actions[i] == "hit"]
    
    for hit_idx in hit_indices:
        # Fenêtre TRES large : entre 0.15s et 6.0s AVANT le hit (rebonds en haut arrivent plus tôt)
        search_start = max(0, hit_idx - int(6.0 * fps))
        search_end = hit_idx - int(0.15 * fps)
        
        if search_start >= search_end:
            continue
        
        # Vérifier s'il y a déjà un rebond dans cette fenêtre
        already_has_bounce = any(actions[i] == "bounce" for i in range(search_start, search_end))
        if already_has_bounce:
            continue  # Un seul rebond maximum avant un hit
        
        # Chercher les PICS DE PERTURBATION les plus forts
        best_bounce_idx = None
        best_bounce_score = 0.0
        
        for i in range(search_start, search_end):
            if actions[i] != "air":
                continue
            
            if np.isnan(ym[i]) or np.isnan(jerk[i]) or np.isnan(ay[i]):
                continue
            
            # Vérifier la distance pour adapter la détection
            distance_from_net = abs(ym[i])
            
            # Comportement de vy au rebond - différent selon la zone
            has_vy_pattern = False
            if i > 1 and i < n-2 and not (np.isnan(vy[i-2]) or np.isnan(vy[i-1]) or np.isnan(vy[i+1]) or np.isnan(vy[i+2])):
                if distance_from_net > 9.0:  # Fond de court - stagnation ou ralentissement puis réaccélération
                    # Minimum local de vy (baisse puis rehausse) ou stagnation
                    is_vy_min_local = vy[i] < vy[i-1] and vy[i] < vy[i+1]
                    is_vy_plateau = abs(vy[i-1] - vy[i]) < 2.0 and abs(vy[i+1] - vy[i]) < 2.0 and vy[i] > -3.0
                    # Ralentissement marqué (décélération forte)
                    is_deceleration = (vy[i-2] - vy[i]) > 1.5 and (vy[i+1] - vy[i]) > 0
                    has_vy_pattern = is_vy_min_local or is_vy_plateau or is_deceleration
                else:  # Près du net - inversion classique
                    has_vy_pattern = (vy[i-1] > 0 and vy[i+1] < 0)
            
            if not has_vy_pattern:
                continue
            
            # Adaptation par zone - TRES PERMISSIF pour le fond de court (signaux faibles)
            
            if distance_from_net > 9.0:  # Fond de court - signaux très faibles, être ULTRA PERMISSIF
                ground_ok = distance_from_net > 1.5  # Plus permissif
                jerk_factor = 0.08  # Très bas pour signaux faibles
                ay_factor = 0.10
                zone_bonus = 8.0  # Bonus très élevé pour compenser signaux faibles
                # Pic dans top 50% locale (très permissif)
                is_strong_peak = jerk[i] > np.nanpercentile(jerk[max(0, i-30):min(n, i+31)], 50)
            elif distance_from_net > 6.0:
                ground_ok = distance_from_net > 2.5
                jerk_factor = 0.20
                ay_factor = 0.25
                zone_bonus = 5.0
                is_strong_peak = jerk[i] > np.nanpercentile(jerk[max(0, i-25):min(n, i+26)], 60)
            elif distance_from_net > 3.0:
                ground_ok = distance_from_net > 3.5
                jerk_factor = 0.40
                ay_factor = 0.45
                zone_bonus = 2.5
                is_strong_peak = jerk[i] > np.nanpercentile(jerk[max(0, i-20):min(n, i+21)], 70)
            else:
                continue
            
            # Utiliser seuil LOW (plus permissif que médian)
            strong_jerk = jerk[i] > jerk_thr_low * jerk_factor 
            strong_ay = abs(ay[i]) > accel_thr * ay_factor
            
            # Max local dans une fenêtre large
            is_local_max = jerk[i] == np.nanmax(jerk[max(0, i-5):min(n, i+6)])
            
            has_curvature = False
            if not np.isnan(curvature[i]):
                # Très permissif pour fond de court
                curv_factor = 0.15 if distance_from_net > 9.0 else (0.30 if distance_from_net > 6.0 else 0.50)
                has_curvature = curvature[i] > curv_thr * curv_factor
            
            spatial_ok = True
            if i > 0 and not np.isnan(xm[i-1]):
                if abs(xm[i] - xm[i-1]) > 3.0:  # Plus permissif
                    spatial_ok = False
            
            # CONDITION assouplie: OR au lieu de AND pour jerk/ay dans le fond
            if distance_from_net > 9.0:
                condition_ok = ground_ok and (strong_jerk or strong_ay) and spatial_ok
            else:
                condition_ok = ground_ok and strong_jerk and strong_ay and is_strong_peak and spatial_ok
            
            if condition_ok:
                score = (
                    jerk[i] * 2.5 +  
                    abs(ay[i]) * 3.5 + 
                    zone_bonus +
                    (5.0 if is_local_max else 0.0) + 
                    (4.0 if has_curvature else 0.0) + 
                    max(0, 20 - distance_from_net) * 0.7  # Bonus plus fort
                )
                
                if score > best_bounce_score:
                    best_bounce_score = score
                    best_bounce_idx = i
        
        # Seuil adaptatif beaucoup plus bas
        if best_bounce_idx is not None:
            distance_from_net = abs(ym[best_bounce_idx])
            if distance_from_net > 9.0:
                min_score = 2.0  # Très bas pour fond
            elif distance_from_net > 6.0:
                min_score = 3.5
            else:
                min_score = 4.5  # Abaissé pour bas de l'écran
            
            if best_bounce_score > min_score:
                actions[best_bounce_idx] = "bounce"

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