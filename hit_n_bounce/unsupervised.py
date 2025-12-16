from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CORRECTION DE L'IMPORT ---
# Assure-toi d'avoir renommé "io.py" en "data_loader.py" !
import data_loader as io_utils  
from features import FeatureConfig, compute_kinematics 

def unsupervised_hit_bounce_detection(ball_data: Dict[str, Any], cfg: FeatureConfig | None = None) -> Dict[str, Any]:
    """
    Détection non-supervisée basée sur la physique :
      - Bounce : Proche du sol, inversion de vy (+ -> -), pic d'accélération ay.
      - Hit : Changement brutal de direction ou de vitesse, loin du sol.
    """
    if cfg is None:
        cfg = FeatureConfig()

    frames = sorted([int(k) for k in ball_data.keys()])
    
    # --- 1. Nettoyage des données via data_loader.py ---
    print("Extraction et nettoyage des données...")
    _, clean_xs, clean_ys, _, _ = io_utils.extract_series(ball_data)
    
    # On crée un masque 'visible' artificiel car extract_series a déjà géré les trous
    vis = [True] * len(frames)

    # --- 2. Calculs Cinématiques ---
    print("Calcul des cinématiques (Vitesse, Accélération, Courbure)...")
    kin = compute_kinematics(frames, clean_xs, clean_ys, vis, cfg)
    y = kin["ys"]
    vy = kin["vy"]
    ay = kin["ay"]
    speed = kin["speed"]
    turn = kin["turn_rate"]
    ground = kin["ground_y"]

    n = len(frames)
    pred = np.array(["air"] * n, dtype=object)

    # --- 3. Heuristiques Physiques ---
    
    # Zone "Proche du sol" (15% de la hauteur max observée)
    yr = np.nanmax(y) - np.nanmin(y)
    yr = yr if np.isfinite(yr) and yr > 1 else 200.0
    near_ground = y > (ground - 0.15 * yr)

    # REBOND (Bounce)
    # Inversion de vitesse verticale (Y augmente vers le bas, donc vy passe de positif à négatif)
    vy_prev = np.r_[vy[0], vy[:-1]]
    sign_flip = (vy_prev > 0) & (vy < 0)
    
    # Seuil dynamique d'accélération verticale (95e percentile)
    ay_thr = np.nanpercentile(np.abs(ay), 95) if n > 10 else np.nanmax(np.abs(ay))
    bounce_cand = near_ground & sign_flip & (np.abs(ay) >= ay_thr)

    # Sélection des meilleurs candidats (Non-Maximum Suppression)
    bounce_score = np.abs(ay) + 0.5 * speed
    bounce_idx = np.where(bounce_cand)[0]
    picked_b = []
    
    for i in bounce_idx:
        # Si un rebond est déjà détecté à +/- 5 frames, on ignore
        if any(abs(i - j) <= 5 for j in picked_b):
            continue
        # On cherche le pic local de score
        lo, hi = max(0, i - 3), min(n, i + 4)
        j = lo + int(np.argmax(bounce_score[lo:hi]))
        if j not in picked_b:
            picked_b.append(j)
            
    for j in picked_b:
        pred[j] = "bounce"

    # FRAPPE (Hit)
    # Changement de direction (turn) ou saut de vitesse (speed_d)
    turn_thr = np.nanpercentile(turn, 95) if n > 10 else np.nanmax(turn)
    speed_d = np.abs(np.r_[0.0, np.diff(speed)])
    speed_thr = np.nanpercentile(speed_d, 95) if n > 10 else np.nanmax(speed_d)
    
    # Un hit n'est généralement pas au sol (sauf demi-volée difficile)
    hit_cand = (~near_ground) & ((turn >= turn_thr) | (speed_d >= speed_thr))

    # On supprime les candidats trop proches des rebonds déjà trouvés
    for j in picked_b:
        lo, hi = max(0, j - 5), min(n, j + 6)
        hit_cand[lo:hi] = False

    # Sélection des meilleurs candidats Hit
    hit_score = turn + 0.5 * speed_d
    hit_idx = np.where(hit_cand)[0]
    picked_h = []
    
    for i in hit_idx:
        if any(abs(i - j) <= 5 for j in picked_h):
            continue
        lo, hi = max(0, i - 3), min(n, i + 4)
        j = lo + int(np.argmax(hit_score[lo:hi]))
        if j not in picked_h:
            picked_h.append(j)
            
    for j in picked_h:
        pred[j] = "hit"

    # --- 4. Construction de la sortie ---
    out = {}
    for i, fr in enumerate(frames):
        d = dict(ball_data[str(fr)])
        # Si la balle n'était pas visible à l'origine, on force "air" (optionnel)
        if not bool(d.get("visible", True)):
            d["pred_action"] = "air"
        else:
            d["pred_action"] = str(pred[i])
        
        # On ajoute les coordonnées nettoyées pour le debug/visu
        d["x_clean"] = clean_xs[i]
        d["y_clean"] = clean_ys[i]
        
        out[str(fr)] = d
        
    return out

# ======================================================
# VISUALISATION DES RÉSULTATS
# ======================================================
def visualize_results(enriched_data: Dict[str, Any], title: str = "Détection Non-Supervisée"):
    frames = sorted([int(k) for k in enriched_data.keys()])
    if not frames:
        print("Aucune donnée à visualiser.")
        return

    # Extraction des données enrichies
    ys = [float(enriched_data[str(f)].get("y_clean", float('nan'))) for f in frames]
    xs = [float(enriched_data[str(f)].get("x_clean", float('nan'))) for f in frames]
    actions = [enriched_data[str(f)]["pred_action"] for f in frames]

    # Préparation des événements pour le plot
    hit_frames = []
    hit_ys = []
    bounce_frames = []
    bounce_ys = []

    for f, y, act in zip(frames, ys, actions):
        if act == "hit":
            hit_frames.append(f)
            hit_ys.append(y)
        elif act == "bounce":
            bounce_frames.append(f)
            bounce_ys.append(y)

    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1. Trajectoire Verticale (Y) - Cruciale pour les rebonds
    ax1.plot(frames, ys, label="Position Y (Verticale)", color='blue', linewidth=1.5)
    ax1.scatter(hit_frames, hit_ys, c='green', s=100, marker='*', label='Hit Détecté', zorder=5)
    ax1.scatter(bounce_frames, bounce_ys, c='red', s=100, marker='^', label='Bounce Détecté', zorder=5)
    
    ax1.set_title(f"{title} - Profil Vertical")
    ax1.set_ylabel("Pixel Y (Inversé: Bas = Sol)")
    ax1.invert_yaxis() # Convention image: 0 en haut
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 2. Trajectoire Horizontale (X) - Pour voir les changements de direction
    ax2.plot(frames, xs, label="Position X (Horizontale)", color='orange', linewidth=1.5)
    
    # On reporte les événements sur X aussi
    # Récupération des X correspondants aux events
    hit_xs = []
    for f in hit_frames:
        val = enriched_data[str(f)].get("x_clean")
        hit_xs.append(float(val) if val is not None else float('nan'))
        
    bounce_xs = []
    for f in bounce_frames:
        val = enriched_data[str(f)].get("x_clean")
        bounce_xs.append(float(val) if val is not None else float('nan'))
    
    ax2.scatter(hit_frames, hit_xs, c='green', s=100, marker='*', zorder=5)
    ax2.scatter(bounce_frames, bounce_xs, c='red', s=100, marker='^', zorder=5)

    ax2.set_title("Profil Horizontal")
    ax2.set_ylabel("Pixel X")
    ax2.set_xlabel("Frame")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os
    
    # Choisis un fichier intéressant (avec des échanges)
    file_to_test = "Data hit & bounce/per_point_v2/ball_data_230.json"
    
    if os.path.exists(file_to_test):
        print(f"Traitement de {file_to_test}...")
        
        # 1. Chargement
        raw_data = io_utils.load_ball_json(file_to_test)
        
        # 2. Détection
        results = unsupervised_hit_bounce_detection(raw_data)
        
        # 3. Stats rapides
        n_hits = sum(1 for v in results.values() if v['pred_action'] == 'hit')
        n_bounces = sum(1 for v in results.values() if v['pred_action'] == 'bounce')
        print(f"Résultats : {n_hits} Hits, {n_bounces} Bounces")
        
        # 4. Visualisation
        visualize_results(results, title=f"Analyse : {os.path.basename(file_to_test)}")
        
    else:
        print(f"Fichier {file_to_test} introuvable.")