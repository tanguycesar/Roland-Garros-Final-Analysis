"""Calibration caméra 21 points avec correction de distorsion.

Génère Camera_Params_Distorted.npz utilisé par features.py pour conversion pixels→mètres.
Lit une frame de vidéo et demande de cliquer 21 points de référence sur le court.
"""
import cv2
import numpy as np
import os

# ======================================================
# CONFIGURATION
# ======================================================
# Cherche la vidéo dans: racine projet, dossier videos/, ou chemin depuis config.txt
VIDEO_FILENAME = "Alcaraz_Sinner_2025-001.mp4"
FRAME_TO_USE = 400000

def find_video_path(filename):
    """Recherche la vidéo dans plusieurs emplacements (racine, videos/, config.txt)."""
    # Chercher dans la racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        os.path.join(project_root, filename),
        os.path.join(project_root, "videos", filename),
        os.path.join(os.getcwd(), filename),
    ]
    
    # Lecture optionnelle de config.txt (exemple fourni dans config.txt.example)
    config_file = os.path.join(project_root, "config.txt")
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    custom_path = line.strip()
                    if os.path.exists(custom_path):
                        return custom_path
    
    # Chercher dans les emplacements par défaut
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

VIDEO_PATH = find_video_path(VIDEO_FILENAME) 

# ======================================================
# POINTS DE RÉFÉRENCE DU COURT (Coordonnées terrain 3D)
# ======================================================
# Dimensions officielles ITF: 10.97m × 23.77m (doubles), 8.23m × 23.77m (simples)
# Origine au centre du filet, Z=0 (plan du sol)
OBJECT_POINTS = np.array([
    # Ligne de fond (+Y) : 5 points de gauche à droite
    [-5.485,  11.885, 0],  # 1. Coin double gauche
    [-4.115,  11.885, 0],  # 2. Coin simple gauche
    [ 0.0,    11.885, 0],  # 3. Centre (marque)
    [ 4.115,  11.885, 0],  # 4. Coin simple droit
    [ 5.485,  11.885, 0],  # 5. Coin double droit
    
    # Ligne de service HAUT (3 points: coins simples + T central)
    [-4.115,   6.40,  0],  # 6. Service Haut GAUCHE (simple)
    [ 0.0,     6.40,  0],  # 7. Service Haut T (centre)
    [ 4.115,   6.40,  0],  # 8. Service Haut DROITE (simple)
    
    # Ligne de FILET (5 points: coin double, coin simple, centre, coin simple, coin double)
    [-5.485,   0.0,   0],  # 9. Filet GAUCHE (coin double)
    [-4.115,   0.0,   0],  # 10. Filet GAUCHE (simple)
    [ 0.0,     0.0,   0],  # 11. Filet CENTRE (marque au milieu)
    [ 4.115,   0.0,   0],  # 12. Filet DROITE (simple)
    [ 5.485,   0.0,   0],  # 13. Filet DROITE (coin double)
    
    # Ligne de service BAS (3 points: coins simples + T central)
    [-4.115,  -6.40,  0],  # 14. Service Bas GAUCHE (simple)
    [ 0.0,    -6.40,  0],  # 15. Service Bas T (centre)
    [ 4.115,  -6.40,  0],  # 16. Service Bas DROITE (simple)
    
    # Ligne de fond BAS (5 points: coin double, coin simple, centre, coin simple, coin double)
    [-5.485, -11.885, 0],  # 17. Fond Bas GAUCHE (coin double)
    [-4.115, -11.885, 0],  # 18. Fond Bas GAUCHE (coin simple)
    [ 0.0,   -11.885, 0],  # 19. Fond Bas CENTRE (petit trait)
    [ 4.115, -11.885, 0],  # 20. Fond Bas DROITE (coin simple)
    [ 5.485, -11.885, 0],  # 21. Fond Bas DROITE (coin double)
], dtype=np.float32)

POINT_NAMES = [
    "1. Fond Haut GAUCHE (coin double)",
    "2. Fond Haut GAUCHE (coin simple)",
    "3. Fond Haut CENTRE (petit trait)",
    "4. Fond Haut DROITE (coin simple)",
    "5. Fond Haut DROITE (coin double)",
    "6. Service Haut GAUCHE (simple)",
    "7. Service Haut T (centre)",
    "8. Service Haut DROITE (simple)",
    "9. Filet GAUCHE (coin double)",
    "10. Filet GAUCHE (ligne simple)",
    "11. Filet CENTRE (marque)",
    "12. Filet DROITE (ligne simple)",
    "13. Filet DROITE (coin double)",
    "14. Service Bas GAUCHE (simple)",
    "15. Service Bas T (centre)",
    "16. Service Bas DROITE (simple)",
    "17. Fond Bas GAUCHE (coin double)",
    "18. Fond Bas GAUCHE (coin simple)",
    "19. Fond Bas CENTRE (petit trait)",
    "20. Fond Bas DROITE (coin simple)",
    "21. Fond Bas DROITE (coin double)"
]

NUM_POINTS = 21

clicked_points = []
img_display = None

def click_event(event, x, y, flags, param):
    """Callback OpenCV : enregistre les clics utilisateur et affiche numéros."""
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < NUM_POINTS:
            clicked_points.append((x, y))
            idx = len(clicked_points) - 1
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_display, str(idx+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Calibration Avancée", img_display)
            
            # Afficher le nom du prochain point à cliquer
            if len(clicked_points) < NUM_POINTS:
                print(f"Point {len(clicked_points)}/{NUM_POINTS} cliqué. Prochain: {POINT_NAMES[len(clicked_points)]}")
            else:
                print(f"Tous les {NUM_POINTS} points ont été cliqués!")

def run_advanced_calibration():
    """Pipeline complet : chargement vidéo, sélection interactive, calibration OpenCV, sauvegarde NPZ."""
    global img_display
    
    if not os.path.exists(VIDEO_PATH):
        print("Vidéo introuvable.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_USE)
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        print("Impossible de lire la frame.")
        return

    h, w = frame.shape[:2]
    img_display = frame.copy()
    
    print("--- CALIBRATION AVEC DISTORSION (21 POINTS) ---")
    print(f"Clique les {NUM_POINTS} points avec précision dans l'ordre:")
    for i, name in enumerate(POINT_NAMES):
        print(f"  {i+1}. {name}")
    print("\nCommence par cliquer: " + POINT_NAMES[0])
    
    cv2.namedWindow("Calibration Avancée", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration Avancée", click_event)
    cv2.imshow("Calibration Avancée", img_display)
    
    while len(clicked_points) < NUM_POINTS:
        if cv2.waitKey(100) == 27: 
            print("Calibration annulée.")
            return

    img_points = np.array([clicked_points], dtype=np.float32)
    obj_points = np.array([OBJECT_POINTS], dtype=np.float32)
        
    # Flags OpenCV : fixe le point principal et l'aspect ratio pour stabiliser l'optimisation
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
    
    # Matrice initiale (focale ~1.5× largeur image)
    camera_matrix_init = np.array([[w*1.5, 0, w/2], [0, w*1.5, h/2], [0, 0, 1]], dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), camera_matrix_init, None, flags=flags
    )
    
    print(f"\n✓ Calibration terminée (Erreur RMS: {ret:.2f} pixels)")
    print("Matrice de distorsion trouvée:")
    print(dist)

    # Sauvegarde matrice caméra, distorsion, rotation, translation
    rvec = rvecs[0]
    tvec = tvecs[0]
    
    np.savez("Camera_Params_Distorted.npz", 
             camera_matrix=mtx, 
             dist_coeffs=dist, 
             rvec=rvec, 
             tvec=tvec)
    print("\n✓ Paramètres sauvegardés dans Camera_Params_Distorted.npz")

    # Vérification : reprojette les points 3D pour comparer avec les clics
    new_img_points, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, mtx, dist)
    new_img_points = new_img_points.reshape(-1, 2)
    
    viz = frame.copy()
    # Points cliqués (rouge) vs reprojetés avec distorsion (vert)
    for i, (p_clic, p_proj) in enumerate(zip(clicked_points, new_img_points)):
        cv2.circle(viz, (int(p_clic[0]), int(p_clic[1])), 4, (0, 0, 255), -1)
        cv2.circle(viz, (int(p_proj[0]), int(p_proj[1])), 3, (0, 255, 0), -1)
        cv2.putText(viz, str(i+1), (int(p_proj[0])+5, int(p_proj[1])-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    # Tracé complet du court (lignes + filet + service)
    connections = [
        # Rectangle extérieur (doubles)
        (0, 4), (4, 20), (20, 16), (16, 0),
        # Rectangle intérieur (simples)
        (1, 3), (3, 19), (19, 17), (17, 1),
        # Marques centrales fond de court
        (2, 2), (18, 18),
        # Lignes de service horizontales
        (5, 7), (13, 15),
        # Ligne centrale (T de service)
        (6, 14),
        # Filet complet
        (8, 9), (9, 10), (10, 11), (11, 12)
    ]
    
    for s, e in connections:
        p1 = tuple(new_img_points[s].astype(int))
        p2 = tuple(new_img_points[e].astype(int))
        cv2.line(viz, p1, p2, (0, 255, 255), 2)

    cv2.imshow("Resultat (Jaune = Modele avec Distorsion)", viz)
    print("\nAppuie sur une touche pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if VIDEO_PATH is None or not os.path.exists(VIDEO_PATH):
        print("Vidéo introuvable!")
        print("\nSolutions:")
        print(f"   1. Placer '{VIDEO_FILENAME}' à la racine du projet")
        print("   2. Créer un dossier 'videos/' et y mettre la vidéo")
        print("   3. Créer un fichier 'config.txt' avec le chemin complet (voir config.txt.example)")
    else:
        run_advanced_calibration()