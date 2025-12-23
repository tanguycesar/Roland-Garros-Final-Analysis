from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

# ======================================================
# 0. UTILITAIRES I/O
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
    
    Détecte les segments de données valides séparés par des trous (NaN).
    Identifie le dernier grand trou (>100 frames) qui marque généralement
    le début du rallye principal après les tentatives de service ratées.
    Conserve uniquement les données à partir de ce point.
    
    Returns:
        Tuple de (x, y) nettoyés avec uniquement le rallye principal
    """
    mask = (~np.isnan(x)).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) <= 1:
        return x, y

    # Calcule la taille des trous entre segments consécutifs
    gaps = []
    for i in range(len(starts) - 1):
        gap_size = starts[i+1] - ends[i]
        if gap_size >= min_service_gap:
            gaps.append(starts[i+1])

    if not gaps:
        return x, y

    # Conserve uniquement les données après le dernier grand trou
    rally_start_frame = gaps[-1]
    
    new_x = np.full_like(x, np.nan)
    new_y = np.full_like(y, np.nan)
    
    new_x[rally_start_frame:] = x[rally_start_frame:]
    new_y[rally_start_frame:] = y[rally_start_frame:]
    
    return new_x, new_y

# ======================================================
# 2. INTERPOLATION POLYNOMIALE (PCHIP)
# ======================================================

def fill_gap_pchip(values: np.ndarray, s_idx: int, e_idx: int) -> np.ndarray:
    """
    Interpole un segment manquant en utilisant PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
    
    Utilise 5 points valides avant et après le trou comme support d'interpolation.
    PCHIP préserve la monotonie et évite les oscillations de Runge contrairement aux splines classiques.
    Si moins de 4 points de support sont disponibles, utilise une interpolation linéaire.
    
    Args:
        values: Tableau contenant les valeurs avec le trou à combler
        s_idx: Index de début du trou
        e_idx: Index de fin du trou
    
    Returns:
        Valeurs interpolées pour combler le trou [s_idx, e_idx]
    """
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
    """
    Comble intelligemment les petits trous dans les trajectoires x et y.
    
    Détecte tous les segments de NaN consécutifs.
    Interpole uniquement les trous internes (pas les bords) de taille <= max_gap_allowed.
    Les trous trop larges (>60 frames = 1.2s à 50 FPS) sont laissés vides car probablement
    dus à une occlusion réelle de la balle.
    
    Args:
        x, y: Coordonnées de la trajectoire
        max_gap_allowed: Taille maximale de trou à interpoler (frames)
    
    Returns:
        Trajectoires (x, y) avec petits trous comblés
    """
    nx, ny = x.copy(), y.copy()
    mask_nan = np.isnan(nx).astype(int)
    diff = np.diff(np.concatenate([[0], mask_nan, [0]]))
    starts_nan = np.where(diff == 1)[0]
    ends_nan = np.where(diff == -1)[0] - 1

    for s, e in zip(starts_nan, ends_nan):
        gap_len = e - s + 1
        # Ignore les trous aux extrémités et les trous trop larges
        if s == 0 or e == len(x)-1 or gap_len > max_gap_allowed:
            continue
        nx[s:e+1] = fill_gap_pchip(x, s, e)
        ny[s:e+1] = fill_gap_pchip(y, s, e)
    return nx, ny

# ======================================================
# 3. PIPELINE DE NETTOYAGE COMPLET
# ======================================================

def process_trajectory(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    """
    Pipeline complet de nettoyage et lissage de trajectoire.
    
    Étapes appliquées dans l'ordre :
    1. Segmentation : Isole le rallye principal (supprime tentatives de service)
    2. Détection d'outliers : Supprime les téléportations (sauts > 250 pixels)
    3. Interpolation : Comble les petits trous (< 60 frames = 1.2s)
    4. Lissage : Applique Savitzky-Golay (fenêtre 7, polynôme 3) pour réduire le bruit
    
    Args:
        xs, ys: Coordonnées brutes en pixels
    
    Returns:
        Trajectoires nettoyées et lissées
    """
    x, y = np.asarray(xs, float), np.asarray(ys, float)

    # Étape 1 : Isoler le rallye principal
    x, y = keep_full_rally_after_service_gap(x, y, min_service_gap=100)

    # Étape 2 : Supprimer les erreurs de tracking (téléportations)
    valid_idx = np.where(~np.isnan(x))[0]
    if len(valid_idx) > 2:
        dists = np.hypot(np.diff(x[valid_idx]), np.diff(y[valid_idx]))
        bad = np.where(dists > 250.0)[0]  # Seuil empirique de saut anormal
        x[valid_idx[bad+1]] = np.nan
        y[valid_idx[bad+1]] = np.nan

    # Étape 3 : Interpoler les petits trous avec PCHIP
    x, y = connect_gaps_smart(x, y, max_gap_allowed=60)

    # Étape 4 : Lissage Savitzky-Golay pour réduire le bruit de mesure
    mask = ~np.isnan(x)
    if np.sum(mask) > 10:
        x[mask] = savgol_filter(x[mask], 7, 3)
        y[mask] = savgol_filter(y[mask], 7, 3)

    return x.tolist(), y.tolist()

def extract_series(ball_data: Dict[str, Any]):
    """
    Extrait et nettoie une série temporelle complète depuis un JSON de tracking.
    
    Parse le dictionnaire JSON où les clés sont des numéros de frame.
    Extrait les coordonnées (x, y), la visibilité et les actions (hit/bounce/air).
    Applique le pipeline de nettoyage complet sur les trajectoires.
    
    Args:
        ball_data: Dictionnaire {frame: {x, y, action, visible}}
    
    Returns:
        Tuple (frames, xs_clean, ys_clean, visibility, actions)
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
    
    # Chemin vers un fichier de test
    data_path = Path(r"C:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2\ball_data_230.json")
    
    if data_path.exists():
        # Charge et traite les données
        data = load_ball_json(data_path)
        frames, xs, ys, vis, acts = extract_series(data)
        
        # Récupère les données brutes pour comparaison
        ys_orig = [data[str(f)].get("y") for f in frames]
        
        # Visualise l'effet du nettoyage
        plt.figure(figsize=(15, 6))
        plt.scatter(frames, ys_orig, color='red', s=8, alpha=0.3, label='Données brutes (avec bruit et trous)')
        plt.plot(frames, ys, color='blue', linewidth=1.5, label='Données nettoyées (segmentation + interpolation + lissage)')
        plt.title("Point 230 - Effet du pipeline de nettoyage sur la trajectoire Y")
        plt.xlabel("Frame")
        plt.ylabel("Position Y (pixels)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()