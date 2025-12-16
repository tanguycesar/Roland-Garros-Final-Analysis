from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json
import os

# ======================================================
# CONFIGURATION
# ======================================================
@dataclass
class FeatureConfig:
    fps: float = 25.0 # Mettre 50.0 si la vidéo est à 50fps
    smooth_window: int = 5

# ======================================================
# 1. GÉOMÉTRIE & CALIBRATION
# ======================================================
class CameraModel:
    def __init__(self):
        # Calibration avec distorsion (Priorité)
        if os.path.exists("Camera_Params_Distorted.npz"):
            data = np.load("Camera_Params_Distorted.npz")
            self.mtx = data['camera_matrix']
            self.dist = data['dist_coeffs']
            self.rvec = data['rvec']
            self.tvec = data['tvec']
        elif os.path.exists("points_image_12.npy"):
            # Fallback simple
            points_img = np.load("points_image_12.npy")
            points_real = np.array([
                [-4.115, 11.885, 0], [ 4.115, 11.885, 0],
                [-4.115, 6.40, 0],   [ 0.0,   6.40, 0],   [ 4.115, 6.40, 0],
                [-4.115, 0.0, 0],    [ 4.115, 0.0, 0],
                [-4.115, -6.40, 0],  [ 0.0,  -6.40, 0],   [ 4.115, -6.40, 0],
                [-4.115, -11.885, 0],[ 4.115, -11.885, 0]
            ], dtype=np.float32)
            h, w = 1080, 1920
            f = w * 1.5
            self.mtx = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
            self.dist = np.zeros((4,1))
            cv2.solvePnP(points_real, points_img, self.mtx, self.dist, self.rvec, self.tvec) 
            success, self.rvec, self.tvec = cv2.solvePnP(points_real, points_img, self.mtx, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            # Mode dégradé si aucune calibration (pour tests unitaires)
            print("⚠️ AVERTISSEMENT: Aucune calibration trouvée. Mode dummy activé.")
            self.mtx = np.eye(3)
            self.dist = np.zeros((5,1))
            self.rvec = np.zeros((3,1))
            self.tvec = np.zeros((3,1))

        rotM, _ = cv2.Rodrigues(self.rvec)
        self.cam_pos = -rotM.T @ self.tvec
        self.inv_rot = rotM.T
        self.inv_mtx = np.linalg.inv(self.mtx)

    def pixel_to_ground(self, u, v):
        if np.isnan(u) or np.isnan(v): return None
        pts_distorted = np.array([[[u, v]]], dtype=np.float32)
        pts_undistorted = cv2.undistortPoints(pts_distorted, self.mtx, self.dist, P=self.mtx)
        u_corr, v_corr = pts_undistorted[0][0]
        
        uv_homo = np.array([u_corr, v_corr, 1.0])
        ray_world = self.inv_rot @ (self.inv_mtx @ uv_homo)
        
        dz = ray_world[2]
        if abs(dz) < 1e-6: return None
        l = -self.cam_pos[2] / dz
        if l < 0: return None
        
        return (self.cam_pos + l * ray_world.reshape(3,1)).flatten()

# ======================================================
# 2. NETTOYAGE DES DONNÉES (Label-Free)
# ======================================================

def clean_local_spikes(x_arr: np.ndarray, y_arr: np.ndarray, threshold: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Supprime les pics locaux (points qui dévient trop de leurs voisins)."""
    n = len(x_arr)
    if n < 3: return x_arr, y_arr
    for _ in range(3):
        valid_idx = np.where(~np.isnan(x_arr))[0]
        if len(valid_idx) < 3: break
        to_remove = []
        for k in range(1, len(valid_idx) - 1):
            prev, curr, next_p = valid_idx[k-1], valid_idx[k], valid_idx[k+1]
            pred_x = (x_arr[prev] + x_arr[next_p]) / 2
            pred_y = (y_arr[prev] + y_arr[next_p]) / 2
            dist = np.sqrt((x_arr[curr] - pred_x)**2 + (y_arr[curr] - pred_y)**2)
            if dist > threshold: to_remove.append(curr)
        if not to_remove: break
        x_arr[to_remove] = float('nan')
        y_arr[to_remove] = float('nan')
    return x_arr, y_arr

def remove_short_segments(x_arr: np.ndarray, y_arr: np.ndarray, min_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Supprime les segments isolés trop courts."""
    mask = ~np.isnan(x_arr)
    if not np.any(mask): return x_arr, y_arr
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) < min_length:
            x_arr[s:e] = float('nan')
            y_arr[s:e] = float('nan')
    return x_arr, y_arr

def clean_velocity_outliers(x_arr: np.ndarray, y_arr: np.ndarray, limit_px: float = 85.0) -> Tuple[np.ndarray, np.ndarray]:
    """Supprime les sauts de vitesse impossibles."""
    for _ in range(5):
        valid_idx = np.where(~np.isnan(x_arr))[0]
        if len(valid_idx) < 2: break
        dx = np.diff(x_arr[valid_idx])
        dy = np.diff(y_arr[valid_idx])
        dt = np.diff(valid_idx)
        dt[dt == 0] = 1.0
        speed = np.sqrt(dx**2 + dy**2) / dt
        bad_jumps = np.where(speed > limit_px)[0]
        if len(bad_jumps) == 0: break
        bad_indices = valid_idx[bad_jumps + 1]
        x_arr[bad_indices] = float('nan')
        y_arr[bad_indices] = float('nan')
    return x_arr, y_arr

def interpolate_smart(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Interpole les trous <= max_gap uniquement."""
    if np.all(np.isnan(arr)): return arr
    valid_mask = ~np.isnan(arr)
    valid_idx = np.where(valid_mask)[0]
    out = np.interp(np.arange(len(arr)), valid_idx, arr[valid_mask])
    gaps = np.diff(valid_idx)
    big_gaps = np.where(gaps > max_gap)[0]
    for i in big_gaps:
        out[valid_idx[i]+1 : valid_idx[i+1]] = float('nan')
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
            try:
                x_arr[s:e] = savgol_filter(x_arr[s:e], wl, 2)
                y_arr[s:e] = savgol_filter(y_arr[s:e], wl, 2)
            except: pass
    return x_arr, y_arr

def process_trajectory_full(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Pipeline complet : Nettoyage -> Reconstruction -> Lissage."""
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    x_arr, y_arr = clean_velocity_outliers(x_arr, y_arr, limit_px=85.0)
    x_arr, y_arr = clean_local_spikes(x_arr, y_arr, threshold=25.0)
    x_arr, y_arr = remove_short_segments(x_arr, y_arr, min_length=7)
    x_arr = interpolate_smart(x_arr, max_gap=5)
    y_arr = interpolate_smart(y_arr, max_gap=5)
    x_arr, y_arr = apply_smoothing(x_arr, y_arr)
    
    return x_arr, y_arr

# ======================================================
# 3. EXTRACTION DE FEATURES (Kinematics)
# ======================================================
def compute_kinematics(frames: List[int], xs: List[float], ys: List[float], vis: List[bool], cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    """
    Calcule les caractéristiques cinématiques (vitesse, accélération, courbure)
    à partir des trajectoires 2D lissées. Indispensable pour la détection non-supervisée.
    """
    # On s'assure d'avoir des données propres
    if isinstance(xs, list): xs = np.array(xs)
    if isinstance(ys, list): ys = np.array(ys)
    
    # 1. Vitesses (Gradient) -> dx/dt et dy/dt (pixels par frame)
    # On utilise np.gradient qui gère les espacements uniformes
    vx = np.gradient(xs)
    vy = np.gradient(ys)
    
    # 2. Accélérations
    ax = np.gradient(vx)
    ay = np.gradient(vy)
    
    # 3. Features scalaires
    speed = np.sqrt(vx**2 + vy**2)
    
    # Taux de virage (Turn Rate) : variation de l'angle du vecteur vitesse
    # Utile pour détecter les hits (changement brusque de direction)
    angles = np.arctan2(vy, vx)
    turn_rate = np.abs(np.gradient(np.unwrap(angles)))
    
    # Estimation robuste du sol (Ground Y)
    # On prend le max Y (bas de l'image) comme proxy du sol
    ground_y = np.nanpercentile(ys, 99) if np.sum(~np.isnan(ys)) > 0 else 1080.0

    return {
        "xs": xs, "ys": ys,
        "vx": vx, "vy": vy,
        "ax": ax, "ay": ay,
        "speed": speed,
        "turn_rate": turn_rate,
        "ground_y": ground_y
    }

# ======================================================
# 4. PHYSIQUE 3D (Drag + Ancrage)
# ======================================================
def projectile_pos_drag(t, p0, v0, k=0.5, g=9.81):
    t = t[:, None]
    safe_k = max(k, 1e-4)
    exp_kt = np.exp(-safe_k * t)
    factor = (1 - exp_kt) / safe_k
    pos = np.zeros_like(t.repeat(3, axis=1)) 
    pos[:, 0] = p0[0] + v0[0] * factor[:, 0]
    pos[:, 1] = p0[1] + v0[1] * factor[:, 0]
    pos[:, 2] = p0[2] + (v0[2] + g/safe_k) * factor[:, 0] - (g/safe_k) * t[:, 0]
    return pos, safe_k

def cost_function_drag(params, t, obs_pts, cam: CameraModel, anchor_point=None):
    p0, v0, k_drag = params[:3], params[3:6], params[6]
    pos_3d, _ = projectile_pos_drag(t, p0, v0, k_drag)
    proj, _ = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.mtx, cam.dist)
    
    valid = ~np.isnan(obs_pts[:,0])
    if np.sum(valid) == 0: return np.zeros(1)

    err_2d = (proj.reshape(-1, 2)[valid] - obs_pts[valid]).ravel()
    
    if anchor_point is not None:
        diff_3d = pos_3d[-1] - anchor_point
        return np.concatenate([err_2d, diff_3d * 40]) 
        
    return err_2d

def solve_trajectory(frames, xs, ys, cam: CameraModel, fps=50.0, is_bounce_end=False):
    valid_mask = ~np.isnan(xs)
    if np.sum(valid_mask) < 6: return None
    
    t_rel = (frames - frames[0]) / fps 
    obs_pts = np.column_stack((xs, ys)).astype(np.float32)

    anchor = None
    if is_bounce_end:
        last_idx = np.where(valid_mask)[0][-1]
        ground_pt = cam.pixel_to_ground(xs[last_idx], ys[last_idx])
        if ground_pt is not None and abs(ground_pt[1]) < 25:
            anchor = ground_pt

    if anchor is not None:
        dist = anchor - np.array([0, -11, 1.5]) 
        v_avg = dist / t_rel[-1]
        x0 = [*[0, -11, 2], *(v_avg * 1.3), 0.5] 
    else:
        x0 = [0, -11, 2, 0, 40, 5, 0.5]

    bounds_min = [-15, -25, -1, -200, -200, -200, 0.1]
    bounds_max = [ 15,  25, 30,  200,  200,  200, 1.5]

    try:
        res = least_squares(cost_function_drag, x0, 
                            args=(t_rel, obs_pts, cam, anchor),
                            bounds=(bounds_min, bounds_max),
                            method='trf', loss='soft_l1')
    except: return None

    if not res.success: return None

    p0, v0, k_opt = res.x[:3], res.x[3:6], res.x[6]
    pos_3d, _ = projectile_pos_drag(t_rel, p0, v0, k_opt)
    
    exp_kt = np.exp(-k_opt * t_rel)
    vel_3d = np.zeros_like(pos_3d)
    vel_3d[:, 0] = v0[0] * exp_kt
    vel_3d[:, 1] = v0[1] * exp_kt
    vel_3d[:, 2] = (v0[2] + 9.81/k_opt) * exp_kt - 9.81/k_opt
    speed = np.linalg.norm(vel_3d, axis=1) * 3.6 

    proj, _ = cv2.projectPoints(pos_3d, cam.rvec, cam.tvec, cam.mtx, cam.dist)
    
    return {
        "frames": frames, "speed": speed,
        "xs": pos_3d[:,0], "ys": pos_3d[:,1], "zs": pos_3d[:,2],
        "proj_x": proj.reshape(-1,2)[:,0], "proj_y": proj.reshape(-1,2)[:,1]
    }

# ======================================================
# 5. ANALYSE PRINCIPALE & VISUALISATION
# ======================================================
def analyze_rally(json_path, video_path_for_fps=None):
    fps = 50.0 
    if video_path_for_fps and os.path.exists(video_path_for_fps):
        cap = cv2.VideoCapture(video_path_for_fps)
        d_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if d_fps > 10: fps = d_fps
    
    cam = CameraModel() 
    with open(json_path, 'r') as f: data = json.load(f)
    frames = sorted([int(k) for k in data.keys()])
    
    def safe_float(d, key):
        val = d.get(key)
        if val is None: return float('nan')
        return float(val)

    xs_raw = [safe_float(data[str(k)], 'x') for k in frames]
    ys_raw = [safe_float(data[str(k)], 'y') for k in frames]
    vis = np.array([bool(data[str(k)].get('visible', True)) for k in frames])
    actions = [data[str(k)].get('action', 'air') for k in frames]

    # --- NETTOYAGE SANS LABELS ---
    xs, ys = process_trajectory_full(xs_raw, ys_raw)
    
    results = []
    current_seg_idx = []
    
    # Pour la physique, on utilise encore les actions "Oracle" (hit/bounce) 
    # pour segmenter la trajectoire et afficher la vitesse.
    # Pour le mode non-supervisé pur, il faudra utiliser tes propres détections.
    for i in range(len(frames)):
        act = actions[i]
        is_hit = (act == 'hit')
        is_bounce = (act == 'bounce')
        
        if vis[i] and not np.isnan(xs[i]):
            current_seg_idx.append(i)

        is_gap = (i > 0 and frames[i] - frames[i-1] > 5)
        
        if (is_hit or is_bounce or is_gap) and len(current_seg_idx) > 0:
            seg_indices = current_seg_idx if is_bounce else current_seg_idx[:-1]
            if len(seg_indices) > 5:
                f_seg = np.array([frames[k] for k in seg_indices])
                x_seg = xs[seg_indices]
                y_seg = ys[seg_indices]
                
                start_offset = 2 if len(f_seg) > 8 else 0
                res = solve_trajectory(f_seg[start_offset:], x_seg[start_offset:], y_seg[start_offset:], 
                                     cam, fps=fps, is_bounce_end=is_bounce)
                if res: results.append(res)
            
            current_seg_idx = [i] if (is_bounce and not is_gap) else []

    return results, frames, xs, ys

def visualize_dashboard(results, all_frames, all_xs, all_ys):
    if not results: return
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(221); ax1.set_title("Vitesse (km/h) - Données Nettoyées"); ax1.set_ylim(0, 260); ax1.grid(True)
    ax2 = fig.add_subplot(222); ax2.set_title("Vue Dessus"); ax2.set_aspect('equal')
    ax2.plot([-4.1,4.1,4.1,-4.1,-4.1], [11.9,11.9,-11.9,-11.9,11.9], 'k-')
    ax2.set_xlim(-10, 10); ax2.set_ylim(-20, 20)
    ax3 = fig.add_subplot(223); ax3.set_title("Profil Z"); ax3.set_ylim(0, 15); ax3.axhline(0, c='k')
    ax4 = fig.add_subplot(224); ax4.set_title("Reprojection (Gris=Données Lissées)"); ax4.invert_yaxis()
    
    ax4.scatter(all_xs, all_ys, c='gray', alpha=0.2, s=5)

    colors = plt.cm.jet(np.linspace(0, 1, len(results)))
    for i, res in enumerate(results):
        vmax = np.max(res['speed'])
        if vmax > 300: continue 
        c = colors[i]
        ax1.plot(res['frames'], res['speed'], c=c, lw=2)
        ax1.text(res['frames'][len(res['frames'])//2], vmax, f"{int(vmax)}", color=c, fontweight='bold')
        ax2.plot(res['xs'], res['ys'], c=c); ax3.plot(res['ys'], res['zs'], c=c)
        ax4.plot(res['proj_x'], res['proj_y'], c=c, ls='--')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    path = "Data hit & bounce/per_point_v2/ball_data_230.json"
    if os.path.exists(path):
        res, fr, x, y = analyze_rally(path)
        visualize_dashboard(res, fr, x, y)
    else:
        print("Fichier JSON introuvable.")