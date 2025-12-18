from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

@dataclass
class FeatureConfig:
    fps: float = 50.0 
    local_window: int = 5

# ======================================================
# 1. CONVERSION PIXELS -> MÈTRES
# ======================================================
def load_calibration(path="Camera_Params_Distorted.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} est introuvable.")
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs'], data['rvec'], data['tvec']

def pixels_to_world_meters(xs: np.ndarray, ys: np.ndarray, mtx, dist, rvec, tvec):
    pts_img = np.array([[[x, y]] for x, y in zip(xs, ys)], dtype=np.float32)
    pts_undistorted = cv2.undistortPoints(pts_img, mtx, dist, P=mtx)
    R, _ = cv2.Rodrigues(rvec)
    M = np.hstack((R[:, :2], tvec.reshape(3, 1)))
    H = mtx @ M
    H_inv = np.linalg.inv(H)
    pts_world = []
    for p in pts_undistorted:
        if np.isnan(p[0][0]):
            pts_world.append([np.nan, np.nan])
            continue
        px_homog = np.array([p[0][0], p[0][1], 1.0])
        world_homog = H_inv @ px_homog
        pts_world.append([world_homog[0] / world_homog[2], world_homog[1] / world_homog[2]])
    return np.array(pts_world)

# ======================================================
# 2. CALCUL CINÉMATIQUE ROBUSTE
# ======================================================
def safe_gradient(arr: np.ndarray, dt: float) -> np.ndarray:
    grad = np.full_like(arr, np.nan)
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) < 2: return grad
    grad_values = np.gradient(arr[valid_idx], valid_idx * dt)
    grad[valid_idx] = grad_values
    return grad

def compute_kinematics(frames: List[int], xs_px: np.ndarray, ys_px: np.ndarray, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    mtx, dist, rvec, tvec = load_calibration()
    pts_meters = pixels_to_world_meters(xs_px, ys_px, mtx, dist, rvec, tvec)
    xm, ym = pts_meters[:, 0], pts_meters[:, 1]
    dt = 1.0 / cfg.fps

    vx = safe_gradient(xm, dt)
    vy = safe_gradient(ym, dt)
    speed = np.hypot(vx, vy)
    ax = safe_gradient(vx, dt)
    ay = safe_gradient(vy, dt)
    accel = np.hypot(ax, ay)
    jerk = np.hypot(safe_gradient(ax, dt), safe_gradient(ay, dt))

    angles = np.arctan2(vy, vx)
    valid_idx = np.where(~np.isnan(angles))[0]
    turn_rate = np.full_like(angles, np.nan)
    if len(valid_idx) > 2:
        turn_rate[valid_idx] = np.abs(np.gradient(np.unwrap(angles[valid_idx]), valid_idx * dt))
    
    # Ground estimation (99e percentile de Y en pixels)
    ground_y = np.nanpercentile(ys_px, 99) if len(ys_px) > 0 else 0.0

    return {
        "xm": xm, "ym": ym, "vx": vx, "vy": vy, "ax": ax, "ay": ay,
        "speed": speed, "accel": accel, "jerk": jerk, "turn_rate": turn_rate,
        "xs": xs_px, "ys": ys_px, "ground_y": np.array([ground_y])
    }

# ======================================================
# 3. SORTIE VISUELLE : LE DASHBOARD
# ======================================================
def visualize_dashboard(frames: List[int], kin: Dict[str, np.ndarray], detections: List[str] | None = None):
    """
    Génère une vue complète pour valider la physique et la détection.
    """
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(4, 2, hspace=0.3, wspace=0.2)

    # --- A. Vue 2D du terrain (Top-down) ---
    ax_map = fig.add_subplot(grid[0:2, 0])
    ax_map.plot(kin["xm"], kin["ym"], 'k-', alpha=0.4, label="Trajectoire")
    # Dessin des limites du court (Simple)
    ax_map.plot([-4.11, 4.11, 4.11, -4.11, -4.11], [11.88, 11.88, -11.88, -11.88, 11.88], 'r--', lw=1, label="Court")
    ax_map.axhline(0, color='blue', lw=1, alpha=0.5, label="Filet")
    ax_map.set_title("Vue Terrain (mètres)")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.axis('equal')
    ax_map.legend()

    # --- B. Profondeur vs Temps ---
    ax_y = fig.add_subplot(grid[0, 1])
    ax_y.plot(frames, kin["ym"], color='purple', lw=1.5)
    ax_y.axhline(11.88, color='red', ls=':', alpha=0.5)
    ax_y.axhline(-11.88, color='red', ls=':', alpha=0.5)
    ax_y.set_title("Profondeur sur le terrain (Ym)")
    ax_y.set_ylabel("Mètres")

    # --- C. Vitesse ---
    ax_sp = fig.add_subplot(grid[1, 1], sharex=ax_y)
    ax_sp.plot(frames, kin["speed"] * 3.6, color='blue', lw=1.5) # Conversion km/h
    ax_sp.set_title("Vitesse (km/h)")
    ax_sp.set_ylabel("km/h")

    # --- D. Le JERK (Détection des impacts) ---
    ax_jk = fig.add_subplot(grid[2, :])
    ax_jk.plot(frames, kin["jerk"], color='red', lw=1.2, label="Jerk (m/s³)")
    thr = np.nanpercentile(kin["jerk"], 97)
    ax_jk.axhline(thr, color='black', ls='--', alpha=0.6, label=f"Seuil 97% ({thr:.0f})")
    ax_jk.set_title("Signal de Détection (JERK)")
    ax_jk.set_ylim(0, thr * 4) # On cadre pour voir les pics
    ax_jk.legend()

    # --- E. Turn Rate (Direction) ---
    ax_tr = fig.add_subplot(grid[3, :], sharex=ax_jk)
    ax_tr.plot(frames, kin["turn_rate"], color='green', lw=1.2)
    ax_tr.set_title("Changement de direction (Rad/s)")
    ax_tr.set_xlabel("Frames")

    # Si des détections sont fournies, on les superpose
    if detections is not None:
        for i, act in enumerate(detections):
            if act == "hit":
                ax_jk.axvline(frames[i], color='green', alpha=0.3)
                ax_map.scatter(kin["xm"][i], kin["ym"][i], color='green', marker='*', s=100)
            elif act == "bounce":
                ax_jk.axvline(frames[i], color='orange', alpha=0.3)
                ax_map.scatter(kin["xm"][i], kin["ym"][i], color='orange', marker='o', s=50)

    plt.tight_layout()
    plt.show()

# ======================================================
# EXECUTION DE TEST
# ======================================================
if __name__ == "__main__":
    import json
    from pathlib import Path
    import data_loader as io_utils
    
    test_file = Path(r"c:\Users\tangu\Desktop\Test_Quantum_Tennis\Roland-Garros-Final-Analysis\Data hit & bounce\per_point_v2\ball_data_230.json")
    
    print(f"🔍 Recherche du fichier: {test_file}")
    
    if test_file.exists():
        print("✅ Fichier trouvé, chargement...")
        with open(test_file, 'r') as f:
            ball_data = json.load(f)
        
        frames, xs, ys, actions = io_utils.extract_series(ball_data)
        
        print(f"📊 {len(frames)} frames chargées")
        print(f"Points valides: {sum(~np.isnan(xs))}")
        
        cfg = FeatureConfig()
        print("⚙️ Calcul des features cinématiques...")
        kin = compute_kinematics(frames, np.array(xs), np.array(ys), cfg)
        
        print("📈 Affichage du dashboard...")
        visualize_dashboard(frames, kin, actions)
    else:
        print(f"❌ Fichier introuvable: {test_file}")
        print(f"Répertoire actuel: {os.getcwd()}")