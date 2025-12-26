from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

# ======================================================
# 0. CHARGEMENT DES DONNÉES
# ======================================================

def load_ball_json(path: str | Path) -> Dict[str, Any]:
    """Charge un fichier JSON de trajectoire de balle."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def iter_point_files(points_dir: str | Path) -> List[Path]:
    """Retourne la liste des fichiers JSON dans le répertoire."""
    points_dir = Path(points_dir)
    if not points_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {points_dir}")
    return sorted(points_dir.glob("ball_data_*.json"))


# ======================================================
# 1. SEGMENTATION DE TRAJECTOIRE
# ======================================================

def keep_full_rally_after_service_gap(x: np.ndarray, y: np.ndarray, min_service_gap: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isole le rallye principal en détectant le dernier silence de tracking prolongé.
    Conserve uniquement les données après le dernier grand trou (>100 frames).
    """
    mask = (~np.isnan(x)).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) <= 1:
        return x, y

    # Trouve le dernier grand trou
    gaps = [starts[i+1] for i in range(len(starts) - 1) if starts[i+1] - ends[i] >= min_service_gap]
    
    if not gaps:
        return x, y

    # Conserve uniquement après le dernier grand trou
    rally_start = gaps[-1]
    new_x, new_y = np.full_like(x, np.nan), np.full_like(y, np.nan)
    new_x[rally_start:] = x[rally_start:]
    new_y[rally_start:] = y[rally_start:]
    
    return new_x, new_y

# ======================================================
# 2. INTERPOLATION POLYNOMIALE (PCHIP)
# ======================================================

def fill_gap_pchip(values: np.ndarray, s_idx: int, e_idx: int) -> np.ndarray:
    """
    Interpole un segment manquant avec PCHIP (5 points de contexte).
    Utilise interpolation linéaire en fallback si support insuffisant.
    """
    context = 5 
    idx_valid = np.where(~np.isnan(values))[0]
    left = idx_valid[idx_valid < s_idx][-context:]
    right = idx_valid[idx_valid > e_idx][:context]
    support_idx = np.concatenate([left, right])
    
    if len(support_idx) < 4:
        return np.linspace(values[s_idx-1], values[e_idx+1], (e_idx-s_idx+1)+2)[1:-1]
    
    try:
        return PchipInterpolator(support_idx, values[support_idx])(np.arange(s_idx, e_idx + 1))
    except:
        return np.linspace(values[s_idx-1], values[e_idx+1], (e_idx-s_idx+1)+2)[1:-1]

def connect_gaps_smart(x: np.ndarray, y: np.ndarray, max_gap_allowed: int = 60):
    """
    Comble les petits trous (<= 60 frames = 1.2s) dans les trajectoires.
    Ignore les trous aux extrémités et les trous trop larges.
    """
    nx, ny = x.copy(), y.copy()
    mask_nan = np.isnan(nx).astype(int)
    diff = np.diff(np.concatenate([[0], mask_nan, [0]]))
    starts_nan = np.where(diff == 1)[0]
    ends_nan = np.where(diff == -1)[0] - 1

    for s, e in zip(starts_nan, ends_nan):
        if s == 0 or e == len(x)-1 or (e - s + 1) > max_gap_allowed:
            continue
        nx[s:e+1] = fill_gap_pchip(x, s, e)
        ny[s:e+1] = fill_gap_pchip(y, s, e)
    return nx, ny

# ======================================================
# 3. PIPELINE DE NETTOYAGE COMPLET
# ======================================================

def process_trajectory(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    """
    Pipeline : Segmentation → Outliers (>250px) → Interpolation → Lissage Savitzky-Golay.
    """
    x, y = np.asarray(xs, float), np.asarray(ys, float)

    # Isoler le rallye principal
    x, y = keep_full_rally_after_service_gap(x, y)

    # Supprimer les téléportations
    valid_idx = np.where(~np.isnan(x))[0]
    if len(valid_idx) > 2:
        dists = np.hypot(np.diff(x[valid_idx]), np.diff(y[valid_idx]))
        x[valid_idx[np.where(dists > 250.0)[0]+1]] = np.nan
        y[valid_idx[np.where(dists > 250.0)[0]+1]] = np.nan

    # Interpoler les petits trous
    x, y = connect_gaps_smart(x, y)

    # Lissage
    mask = ~np.isnan(x)
    if np.sum(mask) > 10:
        x[mask] = savgol_filter(x[mask], 7, 3)
        y[mask] = savgol_filter(y[mask], 7, 3)

    return x.tolist(), y.tolist()

def extract_series(ball_data: Dict[str, Any]):
    """
    Extrait et nettoie les données depuis le JSON de tracking.
    Retourne : (frames, xs_clean, ys_clean, visibility, actions)
    """
    frames = sorted(int(k) for k in ball_data.keys())
    xs_raw = [float(ball_data[str(f)]["x"]) if ball_data[str(f)].get("x") is not None else np.nan for f in frames]
    ys_raw = [float(ball_data[str(f)]["y"]) if ball_data[str(f)].get("y") is not None else np.nan for f in frames]
    
    xs_clean, ys_clean = process_trajectory(xs_raw, ys_raw)
    act = [str(ball_data[str(f)].get("action", "air")) for f in frames]
    vis = [bool(ball_data[str(f)].get("visible", True)) for f in frames]
    return frames, xs_clean, ys_clean, vis, act


# ======================================================
# 4. VISUALISATION ET TEST
# ======================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    data_path = Path(r"C:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2\ball_data_230.json")
    
    if data_path.exists():
        data = load_ball_json(data_path)
        frames, xs, ys, vis, acts = extract_series(data)
        ys_orig = [data[str(f)].get("y") for f in frames]
        
        plt.figure(figsize=(15, 6))
        plt.scatter(frames, ys_orig, color='red', s=8, alpha=0.3, label='Brut')
        plt.plot(frames, ys, color='blue', linewidth=1.5, label='Nettoyé')
        plt.title("Pipeline de nettoyage - Trajectoire Y")
        plt.xlabel("Frame")
        plt.ylabel("Y (pixels)")
        plt.legend()
        plt.show()
        plt.grid(True, alpha=0.3)
        plt.show()