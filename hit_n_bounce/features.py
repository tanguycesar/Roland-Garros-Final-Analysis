from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json

@dataclass
class FeatureConfig:
    fps: float = 50.0  # Vérifie bien ton FPS réel
    local_window: int = 5

# ======================================================
# CONVERSION PIXELS -> MÈTRES (CALIBRATION)
# ======================================================
def load_calibration(path="Camera_Params_Distorted.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError("Il manque le fichier Camera_Params_Distorted.npz")
    data = np.load(path)
    return data['camera_matrix'], data['dist_coeffs'], data['rvec'], data['tvec']

def pixels_to_world_meters(xs: np.ndarray, ys: np.ndarray, mtx, dist, rvec, tvec):
    """
    Transforme les x,y pixels en coordonnées réelles sur le terrain (X, Y mètres).
    On projette sur le plan Z=0 (le sol).
    """
    pts_img = np.array([[[x, y]] for x, y in zip(xs, ys)], dtype=np.float32)

    # 1. Correction optique (Undistort)
    pts_undistorted = cv2.undistortPoints(pts_img, mtx, dist, P=mtx)

    # 2. Matrice de rotation et passage au sol (Z=0)
    R, _ = cv2.Rodrigues(rvec)
    # On crée la matrice d'homographie spécifique au plan Z=0
    M = np.hstack((R[:, :2], tvec.reshape(3, 1)))
    H = mtx @ M
    H_inv = np.linalg.inv(H)

    # 3. Projection
    pts_world = []
    for p in pts_undistorted:
        if np.isnan(p[0][0]):
            pts_world.append([np.nan, np.nan])
            continue
        
        # Coordonnées homogènes
        px_homog = np.array([p[0][0], p[0][1], 1.0])
        world_homog = H_inv @ px_homog
        # Division par la 3ème coordonnée pour revenir en 2D (mètres)
        world_x = world_homog[0] / world_homog[2]
        world_y = world_homog[1] / world_homog[2]
        pts_world.append([world_x, world_y])

    return np.array(pts_world)

# ======================================================
# CALCUL CINÉMATIQUE (EN MÈTRES)
# ======================================================
def compute_kinematics(frames: List[int], xs_px: np.ndarray, ys_px: np.ndarray, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    # --- STEP 1: CONVERSION MONDE RÉEL ---
    mtx, dist, rvec, tvec = load_calibration()
    pts_meters = pixels_to_world_meters(xs_px, ys_px, mtx, dist, rvec, tvec)
    
    xm = pts_meters[:, 0]  # x en mètres
    ym = pts_meters[:, 1]  # y en mètres (profondeur)
    
    dt = 1.0 / cfg.fps

    # --- STEP 2: DÉRIVÉES PHYSIQUES ---
    # Vitesse (m/s)
    vx = np.gradient(xm) / dt
    vy = np.gradient(ym) / dt
    speed = np.hypot(vx, vy)

    # Accélération (m/s²)
    ax = np.gradient(vx) / dt
    ay = np.gradient(vy) / dt
    accel = np.hypot(ax, ay)

    # JERK (m/s³) -> C'est ICI que la détection se joue
    jerk = np.hypot(np.gradient(ax)/dt, np.gradient(ay)/dt)

    # Turn rate (Rad/s)
    angles = np.unwrap(np.arctan2(vy, vx))
    turn_rate = np.abs(np.gradient(angles)) / dt

    return {
        "xm": xm, "ym": ym,           # Positions terrain (m)
        "speed": speed,               # m/s (réel !)
        "accel": accel,               # m/s²
        "jerk": jerk,                 # m/s³ (LA feature pour les hits)
        "turn_rate": turn_rate,       # rad/s
        "vy": vy                      # Vitesse verticale terrain
    }

def build_feature_matrix(frames: List[int], xs: List[float], ys: List[float], cfg: FeatureConfig):
    kin = compute_kinematics(frames, np.array(xs), np.array(ys), cfg)
    
    # On stacke les features en mètres
    X = np.column_stack([
        kin["speed"],
        kin["accel"],
        kin["jerk"],
        kin["turn_rate"],
        kin["vy"],
        kin["ym"] # On garde la profondeur pour aider le modèle à savoir qui tape
    ])
    
    names = ["speed", "accel", "jerk", "turn_rate", "vy", "y_court"]
    return X, names, kin

# ======================================================
# VISUALISATION DES RÉSULTATS
# ======================================================
def visualize_calibration(xs_px, ys_px, pts_meters):
    """
    Compare les coordonnées pixels vs mètres pour vérifier la calibration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Trajectoire en pixels (image caméra)
    ax1.plot(xs_px, ys_px, 'b.-', alpha=0.6, markersize=3)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('Trajectoire Image Caméra (Pixels)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Trajectoire en mètres (terrain réel)
    xm, ym = pts_meters[:, 0], pts_meters[:, 1]
    ax2.plot(xm, ym, 'r.-', alpha=0.6, markersize=3)
    ax2.set_xlabel('X terrain (m)')
    ax2.set_ylabel('Y terrain (m)')
    ax2.set_title('Trajectoire Terrain Réel (Mètres)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Affichage des dimensions du terrain
    x_range = np.nanmax(xm) - np.nanmin(xm)
    y_range = np.nanmax(ym) - np.nanmin(ym)
    ax2.text(0.02, 0.98, f'Largeur: {x_range:.1f}m\nProfondeur: {y_range:.1f}m', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('calibration_check.png', dpi=150, bbox_inches='tight')
    print("✅ Graphique de calibration sauvegardé: calibration_check.png")
    plt.show()

def visualize_kinematics(frames, kin, save_prefix="kinematics"):
    """
    Visualise toutes les features cinématiques calculées
    """
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    
    # 1. Vitesse (m/s)
    axes[0].plot(frames, kin["speed"], 'b-', linewidth=1.5)
    axes[0].set_ylabel('Vitesse (m/s)')
    axes[0].set_title('Vitesse de la balle')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.nanmean(kin["speed"]), color='gray', linestyle='--', alpha=0.5, label='Moyenne')
    axes[0].legend()
    
    # 2. Accélération (m/s²)
    axes[1].plot(frames, kin["accel"], 'orange', linewidth=1.5)
    axes[1].set_ylabel('Accélération (m/s²)')
    axes[1].set_title('Accélération')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Jerk (m/s³) - LA CLEF pour détecter les hits
    axes[2].plot(frames, kin["jerk"], 'red', linewidth=1.5)
    jerk_thr = np.nanpercentile(kin["jerk"], 98)
    axes[2].axhline(y=jerk_thr, color='darkred', linestyle='--', linewidth=2, label=f'Seuil 98% = {jerk_thr:.1f}')
    axes[2].set_ylabel('Jerk (m/s³)')
    axes[2].set_title('Jerk (Détection des Chocs)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 4. Turn Rate (rad/s)
    axes[3].plot(frames, kin["turn_rate"], 'green', linewidth=1.5)
    axes[3].set_ylabel('Virage (rad/s)')
    axes[3].set_title('Vitesse de Changement de Direction')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Position Y terrain (profondeur)
    axes[4].plot(frames, kin["ym"], 'purple', linewidth=1.5)
    axes[4].set_ylabel('Y terrain (m)')
    axes[4].set_xlabel('Frame')
    axes[4].set_title('Profondeur sur le Terrain (Filet ≈ milieu)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_prefix}_features.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphique cinématique sauvegardé: {save_path}")
    plt.show()

def analyze_and_visualize(json_path: str, cfg: FeatureConfig | None = None):
    """
    Pipeline complet: chargement → conversion → calcul → visualisation
    """
    if cfg is None:
        cfg = FeatureConfig()
    
    # Chargement des données
    import data_loader as io_utils
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    frames, xs_px, ys_px, vis = io_utils.extract_series(data)
    
    print(f"📊 Analyse de {os.path.basename(json_path)}")
    # Correction: vis peut contenir des strings, on convertit en booléens
    vis_bool = [bool(v) if isinstance(v, bool) else (v == 'True' or v == True) for v in vis]
    print(f"   Frames: {len(frames)}, Visible: {sum(vis_bool)}")
    
    # Conversion pixels → mètres
    print("🔄 Conversion pixels → mètres...")
    mtx, dist, rvec, tvec = load_calibration()
    pts_meters = pixels_to_world_meters(np.array(xs_px), np.array(ys_px), mtx, dist, rvec, tvec)
    
    # Vérification calibration
    visualize_calibration(xs_px, ys_px, pts_meters)
    
    # Calcul cinématique
    print("⚙️ Calcul des features cinématiques...")
    kin = compute_kinematics(frames, np.array(xs_px), np.array(ys_px), cfg)
    
    # Stats rapides
    print(f"\n📈 Statistiques:")
    print(f"   Vitesse moyenne: {np.nanmean(kin['speed']):.1f} m/s")
    print(f"   Vitesse max: {np.nanmax(kin['speed']):.1f} m/s")
    print(f"   Jerk max: {np.nanmax(kin['jerk']):.1f} m/s³")
    print(f"   Seuil Jerk (98%): {np.nanpercentile(kin['jerk'], 98):.1f} m/s³")
    
    # Visualisation complète
    visualize_kinematics(frames, kin, save_prefix=os.path.basename(json_path).replace('.json', ''))
    
    return frames, xs_px, ys_px, kin

# ======================================================
# TEST
# ======================================================
if __name__ == "__main__":
    test_file = "Data hit & bounce/per_point_v2/ball_data_230.json"
    
    if os.path.exists(test_file):
        try:
            analyze_and_visualize(test_file)
        except FileNotFoundError as e:
            print(f"❌ Erreur: {e}")
            print("   Assure-toi que 'Camera_Params_Distorted.npz' est présent!")
    else:
        print(f"❌ Fichier non trouvé: {test_file}")