import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# Mets ici ta vidéo
VIDEO_PATH = r"C:\Users\tangu\OneDrive\Desktop\Cours\3 - TRIED\STAGE\Roland-Garros-Final-Analysis\Alcaraz_Sinner_2025.mp4"
FRAME_TO_USE = 400000 

# --- POINTS DU TERRAIN (Mètres) ---
# On utilise toujours les couloirs de simple (Singles Lines)
OBJECT_POINTS = np.array([
    [-4.115,  11.885, 0], [ 4.115,  11.885, 0], # Fond Haut
    [-4.115,   6.40,  0], [ 0.0,     6.40,  0], [ 4.115,   6.40,  0], # Service Haut
    [-4.115,   0.0,   0], [ 4.115,   0.0,   0], # Filet
    [-4.115,  -6.40,  0], [ 0.0,    -6.40,  0], [ 4.115,  -6.40,  0], # Service Bas
    [-4.115, -11.885, 0], [ 4.115, -11.885, 0]  # Fond Bas
], dtype=np.float32)

POINT_NAMES = [
    "1. Fond Haut GAUCHE", "2. Fond Haut DROITE",
    "3. Service Haut GAUCHE", "4. Service Haut T", "5. Service Haut DROITE",
    "6. Filet GAUCHE", "7. Filet DROITE",
    "8. Service Bas GAUCHE", "9. Service Bas T", "10. Service Bas DROITE",
    "11. Fond Bas GAUCHE", "12. Fond Bas DROITE"
]

clicked_points = []
img_display = None

def click_event(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 12:
            clicked_points.append((x, y))
            idx = len(clicked_points) - 1
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_display, str(idx+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Calibration Avancée", img_display)

def run_advanced_calibration():
    global img_display
    
    if not os.path.exists(VIDEO_PATH):
        print("Vidéo introuvable.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_USE)
    ret, frame = cap.read()
    cap.release()
    if not ret: return

    h, w = frame.shape[:2]
    img_display = frame.copy()
    
    print("--- CALIBRATION AVEC DISTORSION ---")
    print("Clique les 12 points avec précision.")
    
    cv2.namedWindow("Calibration Avancée", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration Avancée", click_event)
    cv2.imshow("Calibration Avancée", img_display)
    
    while len(clicked_points) < 12:
        if cv2.waitKey(100) == 27: return

    img_points = np.array([clicked_points], dtype=np.float32) # Shape (1, 12, 2)
    obj_points = np.array([OBJECT_POINTS], dtype=np.float32)  # Shape (1, 12, 3)
        
    # Flags pour aider l'algo car on n'a qu'une seule image (c'est difficile pour lui)
    # CALIB_FIX_PRINCIPAL_POINT : On suppose que le centre optique est au milieu de l'image
    # CALIB_FIX_ASPECT_RATIO : On suppose que les pixels sont carrés
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
    
    # Guess initial
    camera_matrix_init = np.array([[w*1.5, 0, w/2], [0, w*1.5, h/2], [0, 0, 1]], dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), camera_matrix_init, None, flags=flags
    )
    
    print(f" Calibration terminée (Erreur RMS: {ret:.2f} pixels)")
    print("Matrice Distorsion trouvée :")
    print(dist) # Si ce n'est pas que des zéros, c'est qu'il a trouvé la courbure !

    # Sauvegarde
    rvec = rvecs[0]
    tvec = tvecs[0]
    
    np.savez("Camera_Params_Distorted.npz", 
             camera_matrix=mtx, 
             dist_coeffs=dist, 
             rvec=rvec, 
             tvec=tvec)
    print("💾 Sauvegardé dans Camera_Params_Distorted.npz")

    # --- VERIFICATION ---
    # On reprojette les points AVEC la correction de distorsion
    new_img_points, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, mtx, dist)
    new_img_points = new_img_points.reshape(-1, 2)
    
    viz = frame.copy()
    # Dessin des points reprojetés (Vert) vs Cliqués (Rouge)
    for p_clic, p_proj in zip(clicked_points, new_img_points):
        cv2.circle(viz, (int(p_clic[0]), int(p_clic[1])), 4, (0, 0, 255), -1)
        cv2.circle(viz, (int(p_proj[0]), int(p_proj[1])), 3, (0, 255, 0), -1)
        
    # Dessin du terrain virtuel
    connections = [(0,1), (1,11), (11,10), (10,0), (2,4), (7,9), (3,8), (5,6)]
    for s, e in connections:
        p1 = tuple(new_img_points[s].astype(int))
        p2 = tuple(new_img_points[e].astype(int))
        cv2.line(viz, p1, p2, (0, 255, 255), 2)

    cv2.imshow("Resultat (Jaune = Modele avec Distorsion)", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_advanced_calibration()