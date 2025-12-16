from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from scipy.signal import savgol_filter

# Valeurs par défaut pour les données manquantes
DEFAULT_ACTION = "air"
DEFAULT_VISIBLE = True

# ======================================================
# Chargement et sauvegarde des JSON
# ======================================================
def load_ball_json(path: str | Path) -> Dict[str, Any]:
    """Charge un fichier JSON contenant les données d'un point (indexé par frame)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}

def save_ball_json(data: Dict[str, Any], path: str | Path) -> None:
    """Sauvegarde les données de la balle dans un fichier JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def iter_point_files(points_dir: str | Path) -> List[Path]:
    """Trouve tous les fichiers JSON dans un dossier et retourne la liste triée."""
    points_dir = Path(points_dir)
    if not points_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {points_dir}")
    files = sorted(points_dir.rglob("*.json"))
    return files

# ======================================================
# LOGIQUE DE NETTOYAGE AVANCÉE
# ======================================================

def clean_local_spikes(x_arr: np.ndarray, y_arr: np.ndarray, threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supprime les 'pics' locaux : un point qui dévie brutalement de la ligne formée par ses voisins.
    """
    n = len(x_arr)
    if n < 3: return x_arr, y_arr
    
    # On itère pour nettoyer en profondeur
    for _ in range(3):
        valid_idx = np.where(~np.isnan(x_arr))[0]
        if len(valid_idx) < 3: break
        
        to_remove = []
        
        # On regarde chaque point par rapport à ses voisins (i-1 et i+1)
        for k in range(1, len(valid_idx) - 1):
            prev = valid_idx[k-1]
            curr = valid_idx[k]
            next_p = valid_idx[k+1]
            
            # Position prédite (moyenne des voisins)
            pred_x = (x_arr[prev] + x_arr[next_p]) / 2
            pred_y = (y_arr[prev] + y_arr[next_p]) / 2
            
            # Écart réel
            dist = np.sqrt((x_arr[curr] - pred_x)**2 + (y_arr[curr] - pred_y)**2)
            
            if dist > threshold:
                to_remove.append(curr)
        
        if not to_remove:
            break
            
        x_arr[to_remove] = float('nan')
        y_arr[to_remove] = float('nan')
        
    return x_arr, y_arr

def remove_short_segments(x_arr: np.ndarray, y_arr: np.ndarray, min_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supprime les segments isolés < min_length frames.
    """
    mask = ~np.isnan(x_arr)
    if not np.any(mask): return x_arr, y_arr

    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        length = e - s
        if length < min_length:
            x_arr[s:e] = float('nan')
            y_arr[s:e] = float('nan')
            
    return x_arr, y_arr

def clean_velocity_outliers(x_arr: np.ndarray, y_arr: np.ndarray, limit_px: float = 85.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supprime les sauts de vitesse impossibles (téléportations).
    """
    for _ in range(5):
        valid_indices = np.where(~np.isnan(x_arr))[0]
        if len(valid_indices) < 2: break

        dx = np.diff(x_arr[valid_indices])
        dy = np.diff(y_arr[valid_indices])
        dt = np.diff(valid_indices)
        dt[dt == 0] = 1.0 
        
        dist = np.sqrt(dx**2 + dy**2)
        speed = dist / dt 

        bad_jumps = np.where(speed > limit_px)[0]
        
        if len(bad_jumps) == 0: break

        bad_indices = valid_indices[bad_jumps + 1]
        x_arr[bad_indices] = float('nan')
        y_arr[bad_indices] = float('nan')

    return x_arr, y_arr

def interpolate_smart(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Interpole les petits trous uniquement."""
    if np.all(np.isnan(arr)): return arr
    
    n = len(arr)
    valid_mask = ~np.isnan(arr)
    valid_idx = np.where(valid_mask)[0]
    
    out = np.interp(np.arange(n), valid_idx, arr[valid_mask])
    
    gaps = np.diff(valid_idx)
    big_gap_indices = np.where(gaps > max_gap)[0]
    
    for i in big_gap_indices:
        start_gap = valid_idx[i] + 1
        end_gap = valid_idx[i+1]
        out[start_gap:end_gap] = float('nan')
        
    return out

def apply_smoothing(x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lissage Savitzky-Golay."""
    mask = ~np.isnan(x_arr)
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        length = e - s
        if length >= 5:
            wl = min(9, length if length % 2 == 1 else length - 1)
            poly = 2
            try:
                x_arr[s:e] = savgol_filter(x_arr[s:e], wl, poly)
                y_arr[s:e] = savgol_filter(y_arr[s:e], wl, poly)
            except: pass
    return x_arr, y_arr

# ======================================================
# PIPELINE COMPLET
# ======================================================
def process_trajectory(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    # 1. Nettoyage Vitesse (Gros sauts)
    x_arr, y_arr = clean_velocity_outliers(x_arr, y_arr, limit_px=85.0)

    # 2. Nettoyage Pics Locaux (Supprime les zig-zags)
    x_arr, y_arr = clean_local_spikes(x_arr, y_arr, threshold=25.0)

    # 3. Suppression Segments Courts (Bruit)
    x_arr, y_arr = remove_short_segments(x_arr, y_arr, min_length=7)

    # 4. Reconstruction (Interpolation prudente)
    x_arr = interpolate_smart(x_arr, max_gap=5)
    y_arr = interpolate_smart(y_arr, max_gap=5)

    # 5. Lissage Final
    x_arr, y_arr = apply_smoothing(x_arr, y_arr)

    return x_arr.tolist(), y_arr.tolist()

# ======================================================
# Extraction
# ======================================================
def extract_series(ball_data: Dict[str, Any]) -> Tuple[List[int], List[float], List[float], List[bool], List[str]]:
    """Extrait et nettoie les séries temporelles."""
    frames = sorted(int(k) for k in ball_data.keys())
    
    xs, ys, visibles, actions = [], [], [], []

    for frame in frames:
        data = ball_data[str(frame)]
        vx = data.get("x")
        vy = data.get("y")
        xs.append(float(vx) if vx is not None else float("nan"))
        ys.append(float(vy) if vy is not None else float("nan"))
        visibles.append(data.get("visible", DEFAULT_VISIBLE))
        actions.append(data.get("action", DEFAULT_ACTION))
    
    # NETTOYAGE COMPLET (SANS TRICHERIE SUR LES LABELS)
    xs, ys = process_trajectory(xs, ys)
    
    # On renvoie tout, sans couper au premier hit
    return frames, xs, ys, visibles, actions

if __name__ == "__main__":
    # Test simple pour vérifier que ce fichier fonctionne
    print("data_loader.py est prêt !")