from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

# ======================================================
# 1. SEGMENTATION : DÉTECTION DU VRAI DÉPART (RALLYE COMPLET)
# ======================================================

def keep_full_rally_after_service_gap(x: np.ndarray, y: np.ndarray, min_service_gap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Règle : Trouve le dernier grand trou (silence de données) supérieur à min_service_gap.
    Supprime tout ce qui précède ce trou (la 1ère balle ratée).
    Garde TOUT ce qui suit (le service réussi + tout l'échange).
    """
    mask = (~np.isnan(x)).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) <= 1:
        return x, y

    # Identifier les trous entre les segments
    gaps = []
    for i in range(len(starts) - 1):
        gap_size = starts[i+1] - ends[i]
        if gap_size >= min_service_gap:
            gaps.append(starts[i+1]) # On note l'index où le nouveau segment commence après un gros trou

    if not gaps:
        return x, y

    # Le vrai rallye commence après le DERNIER gros trou (le serveur s'est enfin installé)
    rally_start_frame = gaps[-1]
    
    new_x = np.full_like(x, np.nan)
    new_y = np.full_like(y, np.nan)
    
    # On garde TOUT depuis ce point jusqu'à la fin du fichier
    new_x[rally_start_frame:] = x[rally_start_frame:]
    new_y[rally_start_frame:] = y[rally_start_frame:]
    
    return new_x, new_y

# ======================================================
# 2. INTERPOLATION NON-LINÉAIRE (PCHIP)
# ======================================================

def fill_gap_pchip(values: np.ndarray, s_idx: int, e_idx: int) -> np.ndarray:
    context = 5 
    idx_valid = np.where(~np.isnan(values))[0]
    left_support = idx_valid[idx_valid < s_idx][-context:]
    right_support = idx_valid[idx_valid > e_idx][:context]
    support_idx = np.concatenate([left_support, right_support])
    
    if len(support_idx) < 4:
        return np.linspace(values[s_idx-1], values[e_idx+1], (e_idx-s_idx+1)+2)[1:-1]
    try:
        pchip = PchipInterpolator(support_idx, values[support_idx])
        return pchip(np.arange(s_idx, e_idx + 1))
    except:
        return np.linspace(values[s_idx-1], values[e_idx+1], (e_idx-s_idx+1)+2)[1:-1]

def connect_gaps_smart(x: np.ndarray, y: np.ndarray, max_gap_allowed: int = 60):
    nx, ny = x.copy(), y.copy()
    mask_nan = np.isnan(nx).astype(int)
    diff = np.diff(np.concatenate([[0], mask_nan, [0]]))
    starts_nan = np.where(diff == 1)[0]
    ends_nan = np.where(diff == -1)[0] - 1

    for s, e in zip(starts_nan, ends_nan):
        gap_len = e - s + 1
        # On ne bouche que les petits trous internes au rallye
        if s == 0 or e == len(x)-1 or gap_len > max_gap_allowed:
            continue
        nx[s:e+1] = fill_gap_pchip(x, s, e)
        ny[s:e+1] = fill_gap_pchip(y, s, e)
    return nx, ny

# ======================================================
# 3. PIPELINE FINAL
# ======================================================

def process_trajectory(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    x, y = np.asarray(xs, float), np.asarray(ys, float)

    # A. Isolation du rallye COMPLET (après le temps mort du service)
    # On fait ça en premier sur les données brutes
    x, y = keep_full_rally_after_service_gap(x, y, min_service_gap=100)

    # B. Nettoyage des erreurs de mesure (téléportations)
    valid_idx = np.where(~np.isnan(x))[0]
    if len(valid_idx) > 2:
        dists = np.hypot(np.diff(x[valid_idx]), np.diff(y[valid_idx]))
        bad = np.where(dists > 250.0)[0] # Seuil de saut px
        x[valid_idx[bad+1]] = np.nan
        y[valid_idx[bad+1]] = np.nan

    # C. Interpolation des petits trous internes (max 60 frames)
    x, y = connect_gaps_smart(x, y, max_gap_allowed=60)

    # D. Lissage final fidèle aux données (window=7)
    mask = ~np.isnan(x)
    if np.sum(mask) > 10:
        x[mask] = savgol_filter(x[mask], 7, 3)
        y[mask] = savgol_filter(y[mask], 7, 3)

    return x.tolist(), y.tolist()

def extract_series(ball_data: Dict[str, Any]):
    frames = sorted(int(k) for k in ball_data.keys())
    xs_raw = [float(ball_data[str(f)]["x"]) if ball_data[str(f)].get("x") is not None else np.nan for f in frames]
    ys_raw = [float(ball_data[str(f)]["y"]) if ball_data[str(f)].get("y") is not None else np.nan for f in frames]
    
    xs_clean, ys_clean = process_trajectory(xs_raw, ys_raw)
    act = [str(ball_data[str(f)].get("action", "air")) for f in frames]
    return frames, xs_clean, ys_clean, act

# ======================================================
# 4. VISUALISATION
# ======================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Remplacez par votre point 236 ou celui qui posait problème
    data_path = Path(r"C:\Users\tangu\OneDrive\Desktop\Cours\3 - TRIED\STAGE\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2\ball_data_230.json")
    
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames, xs, ys, _ = extract_series(data)
        
        ys_orig = [data[str(f)].get("y") for f in frames]
        
        plt.figure(figsize=(15, 6))
        plt.scatter(frames, ys_orig, color='red', s=8, alpha=0.3, label='Brut (Données initiales)')
        plt.plot(frames, ys, color='blue', linewidth=1.5, label='Rallye complet (service réussi + échange)')
        plt.title("Isolation du Rallye : On garde tout après le silence de la 1ère balle")
        plt.legend()
        plt.show()