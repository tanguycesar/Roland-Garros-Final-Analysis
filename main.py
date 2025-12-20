from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# --- IMPORTS DEPUIS LE PACKAGE hit_n_bounce ---
try:
    from hit_n_bounce import data_loader as io_utils
    from hit_n_bounce import features as feat_utils
    from hit_n_bounce.features import FeatureConfig
    from hit_n_bounce import supervised
    from hit_n_bounce import unsupervised
    try:
        from hit_n_bounce import supervised_dl
        _HAS_DL = True
    except ImportError:
        _HAS_DL = False
        print("⚠️  Modèle Deep Learning non disponible (TensorFlow manquant)")
except ImportError as e:
    print(f"❌ Erreur d'import critique : {e}")
    print("Vérifiez que le dossier 'hit_n_bounce' contient bien tous les modules.")
    sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hit & Bounce Detection - Roland-Garros 2025")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- CALIBRATION ---
    p_calib = sub.add_parser("calibrate", help="Run camera calibration tool")
    p_calib.add_argument("--video", help="Path to video file (optional if using config.txt)")
    p_calib.add_argument("--frame", type=int, default=400000, help="Frame number to use for calibration")

    # --- TRAIN SUPERVISED (XGBoost/HistGradientBoosting) ---
    p_train = sub.add_parser("train", help="Train supervised ML model (XGBoost)")
    p_train.add_argument("--points_dir", default="Data hit & bounce/per_point_v2", help="Folder containing JSON data")
    p_train.add_argument("--model_path", default="models/tennis_event_classifier.joblib", help="Output model path")

    # --- TRAIN DEEP LEARNING ---
    p_train_dl = sub.add_parser("train-dl", help="Train deep learning model (LSTM)")
    p_train_dl.add_argument("--points_dir", default="Data hit & bounce/per_point_v2", help="Folder containing JSON data")
    p_train_dl.add_argument("--model_type", choices=["lstm", "dense"], default="lstm", help="DL architecture")
    p_train_dl.add_argument("--epochs", type=int, default=100, help="Max training epochs")

    # --- PREDICT ---
    p_pred = sub.add_parser("predict", help="Predict hits/bounces on data")
    p_pred.add_argument("--method", choices=["unsupervised", "supervised", "supervised-dl"], required=True)
    p_pred.add_argument("--model_path", help="Model path (for supervised methods)")
    p_pred.add_argument("--input", help="Single JSON file path")
    p_pred.add_argument("--input_dir", help="Folder of JSON files")
    p_pred.add_argument("--output", help="Output file path (single file mode)")
    p_pred.add_argument("--output_dir", default="outputs", help="Output folder (batch mode)")
    p_pred.add_argument("--visualize", action="store_true", help="Show visualization")

    # --- VISUALIZE ---
    p_viz = sub.add_parser("visualize", help="Visualize trajectory and detections")
    p_viz.add_argument("--input", required=True, help="JSON file to visualize")
    p_viz.add_argument("--method", choices=["unsupervised", "supervised", "supervised-dl"], default="unsupervised")
    p_viz.add_argument("--model_path", help="Model path (for supervised methods)")

    # --- PROCESS DATA ---
    p_process = sub.add_parser("process-data", help="Process and clean trajectory data")
    p_process.add_argument("--input", required=True, help="Input JSON file")
    p_process.add_argument("--output", help="Output JSON file (optional)")
    p_process.add_argument("--visualize", action="store_true", help="Show before/after")

    return p.parse_args()

# --- LOGIQUE PRINCIPALE ---

def main() -> None:
    args = parse_args()
    cfg = FeatureConfig()

    # ========================================
    # 1. CALIBRATION
    # ========================================
    if args.cmd == "calibrate":
        print("🎯 Lancement de l'outil de calibration caméra...")
        try:
            from hit_n_bounce import calibration_distortion
            calibration_distortion.run_advanced_calibration()
        except Exception as e:
            print(f"❌ Erreur de calibration: {e}")
        return

    # ========================================
    # 2. PROCESS DATA (nettoyage trajectoires)
    # ========================================
    if args.cmd == "process-data":
        print(f"🔧 Traitement des données: {args.input}")
        try:
            import json
            import matplotlib.pyplot as plt
            
            with open(args.input, 'r') as f:
                ball_data = json.load(f)
            
            frames, xs, ys, vis, acts = io_utils.extract_series(ball_data)
            
            if args.visualize:
                ys_orig = [ball_data[str(f)].get("y") for f in frames]
                plt.figure(figsize=(15, 6))
                plt.scatter(frames, ys_orig, color='red', s=8, alpha=0.3, label='Données brutes')
                plt.plot(frames, ys, color='blue', linewidth=1.5, label='Données nettoyées')
                plt.title(f"Trajectoire nettoyée - {Path(args.input).name}")
                plt.legend()
                plt.show()
            
            if args.output:
                # Sauvegarder les données nettoyées
                output_data = {}
                for i, f in enumerate(frames):
                    output_data[str(f)] = {
                        "x": xs[i],
                        "y": ys[i],
                        "visible": vis[i],
                        "action": acts[i]
                    }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"✅ Données nettoyées sauvegardées: {args.output}")
            
            print(f"✅ Traitement terminé: {len(frames)} frames, {sum(~np.isnan(xs))} points valides")
        except Exception as e:
            print(f"❌ Erreur de traitement: {e}")
        return

    # ========================================
    # 3. ENTRAÎNEMENT SUPERVISED (ML)
    # ========================================
    if args.cmd == "train":
        print(f"🎓 Entraînement du modèle supervisé (XGBoost/HistGradientBoosting)")
        print(f"📂 Données: {args.points_dir}")
        try:
            supervised.train_supervised(args.points_dir, args.model_path, cfg)
            print(f"\n✅ Modèle sauvegardé: {args.model_path}")
        except Exception as e:
            print(f"❌ Entraînement échoué: {e}")
            import traceback
            traceback.print_exc()
        return

    # ========================================
    # 4. ENTRAÎNEMENT DEEP LEARNING
    # ========================================
    if args.cmd == "train-dl":
        if not _HAS_DL:
            print("❌ TensorFlow requis. Installer avec: pip install tensorflow")
            return
        
        print(f"🧠 Entraînement du modèle Deep Learning ({args.model_type.upper()})")
        print(f"📂 Données: {args.points_dir}")
        try:
            model, metrics = supervised_dl.run_supervised_dl_pipeline(cfg, model_type=args.model_type)
            print(f"\n✅ Modèle sauvegardé: models/tennis_event_classifier_dl_{args.model_type}.keras")
        except Exception as e:
            print(f"❌ Entraînement échoué: {e}")
            import traceback
            traceback.print_exc()
        return

    # ========================================
    # 5. VISUALISATION
    # ========================================
    if args.cmd == "visualize":
        print(f"📊 Visualisation: {args.input}")
        try:
            import json
            import matplotlib.pyplot as plt
            
            with open(args.input, 'r') as f:
                ball_data = json.load(f)
            
            frames, xs, ys, vis, acts = io_utils.extract_series(ball_data)
            kin = feat_utils.compute_kinematics(frames, np.array(xs), np.array(ys), cfg)
            
            # Détection selon la méthode
            if args.method == "unsupervised":
                pred_actions = unsupervised.detect_tennis_events(frames, kin, cfg.fps)
                results = {str(f): {"pred_action": pred_actions[i]} for i, f in enumerate(frames)}
                unsupervised.visualize_results(frames, kin, results)
            
            elif args.method == "supervised":
                if not args.model_path or not os.path.exists(args.model_path):
                    print("❌ Modèle introuvable. Spécifier --model_path")
                    return
                from joblib import load
                payload = load(args.model_path)
                model = payload["model"]
                X, _ = supervised.make_frame_features(kin, cfg)
                probs = model.predict_proba(X)
                final_actions = supervised._events_from_probs(probs, cfg.fps)
                supervised.visualize_dashboard(frames, kin, probs, final_actions)
            
            elif args.method == "supervised-dl":
                if not _HAS_DL:
                    print("❌ TensorFlow requis")
                    return
                if not args.model_path or not os.path.exists(args.model_path):
                    print("❌ Modèle introuvable. Spécifier --model_path")
                    return
                from tensorflow import keras
                from sklearn.preprocessing import StandardScaler
                model = keras.models.load_model(args.model_path)
                X, _ = supervised_dl.make_frame_features(kin, cfg)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                probs = model.predict(X_scaled, verbose=0)
                final_actions = supervised._events_from_probs(probs, cfg.fps)
                supervised.visualize_dashboard(frames, kin, probs, final_actions)
            
        except Exception as e:
            print(f"❌ Erreur de visualisation: {e}")
            import traceback
            traceback.print_exc()
        return

    # ========================================
    # 6. PRÉDICTION
    # ========================================
    if args.cmd == "predict":
        if not args.input and not args.input_dir:
            sys.exit("❌ Erreur: Spécifier --input ou --input_dir")

        # Mode Fichier Unique
        if args.input:
            inp = Path(args.input)
            out_path = Path(args.output) if args.output else inp.with_name(inp.stem + "_pred.json")
            
            print(f"🔍 Traitement: {inp.name}")
            
            try:
                import json
                with open(inp, 'r') as f:
                    ball_data = json.load(f)
                
                frames, xs, ys, vis, acts = io_utils.extract_series(ball_data)
                kin = feat_utils.compute_kinematics(frames, np.array(xs), np.array(ys), cfg)
                
                if args.method == "unsupervised":
                    pred_actions = unsupervised.detect_tennis_events(frames, kin, cfg.fps)
                
                elif args.method == "supervised":
                    if not args.model_path or not os.path.exists(args.model_path):
                        sys.exit(f"❌ Modèle introuvable: {args.model_path}")
                    from joblib import load
                    payload = load(args.model_path)
                    model = payload["model"]
                    X, _ = supervised.make_frame_features(kin, cfg)
                    probs = model.predict_proba(X)
                    pred_actions = supervised._events_from_probs(probs, cfg.fps)
                
                elif args.method == "supervised-dl":
                    if not _HAS_DL:
                        sys.exit("❌ TensorFlow requis")
                    if not args.model_path or not os.path.exists(args.model_path):
                        sys.exit(f"❌ Modèle introuvable: {args.model_path}")
                    from tensorflow import keras
                    from sklearn.preprocessing import StandardScaler
                    model = keras.models.load_model(args.model_path)
                    X, _ = supervised_dl.make_frame_features(kin, cfg)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    probs = model.predict(X_scaled, verbose=0)
                    pred_actions = supervised._events_from_probs(probs, cfg.fps)
                
                # Sauvegarder les résultats
                output_data = {}
                for i, f in enumerate(frames):
                    output_data[str(f)] = {
                        "x": xs[i] if not np.isnan(xs[i]) else None,
                        "y": ys[i] if not np.isnan(ys[i]) else None,
                        "visible": vis[i],
                        "action": pred_actions[i]
                    }
                
                with open(out_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                # Stats
                hits = sum(1 for a in pred_actions if a == "hit")
                bounces = sum(1 for a in pred_actions if a == "bounce")
                print(f"✅ Détections: {hits} hits, {bounces} bounces")
                print(f"💾 Sauvegardé: {out_path}")
                
                if args.visualize:
                    results = {str(f): {"pred_action": pred_actions[i]} for i, f in enumerate(frames)}
                    if args.method == "unsupervised":
                        unsupervised.visualize_results(frames, kin, results)
                    else:
                        supervised.visualize_dashboard(frames, kin, probs, pred_actions)
                
            except Exception as e:
                print(f"❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
            return

        # Mode Dossier (Batch)
        if args.input_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            files = io_utils.iter_point_files(args.input_dir)
            
            print(f"📦 Traitement de {len(files)} fichiers de {args.input_dir}...")
            
            # Charger le modèle si supervisé
            model = None
            if args.method in ["supervised", "supervised-dl"]:
                if not args.model_path or not os.path.exists(args.model_path):
                    sys.exit(f"❌ Modèle introuvable: {args.model_path}")
                
                if args.method == "supervised":
                    from joblib import load
                    payload = load(args.model_path)
                    model = payload["model"]
                else:
                    if not _HAS_DL:
                        sys.exit("❌ TensorFlow requis")
                    from tensorflow import keras
                    model = keras.models.load_model(args.model_path)
            
            count = 0
            total_hits = 0
            total_bounces = 0
            
            for fp in files:
                try:
                    import json
                    with open(fp, 'r') as f:
                        ball_data = json.load(f)
                    
                    frames, xs, ys, vis, acts = io_utils.extract_series(ball_data)
                    kin = feat_utils.compute_kinematics(frames, np.array(xs), np.array(ys), cfg)
                    
                    if args.method == "unsupervised":
                        pred_actions = unsupervised.detect_tennis_events(frames, kin, cfg.fps)
                    elif args.method == "supervised":
                        X, _ = supervised.make_frame_features(kin, cfg)
                        probs = model.predict_proba(X)
                        pred_actions = supervised._events_from_probs(probs, cfg.fps)
                    else:  # supervised-dl
                        from sklearn.preprocessing import StandardScaler
                        X, _ = supervised_dl.make_frame_features(kin, cfg)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        probs = model.predict(X_scaled, verbose=0)
                        pred_actions = supervised._events_from_probs(probs, cfg.fps)
                    
                    # Sauvegarder
                    output_data = {}
                    for i, f in enumerate(frames):
                        output_data[str(f)] = {
                            "x": xs[i] if not np.isnan(xs[i]) else None,
                            "y": ys[i] if not np.isnan(ys[i]) else None,
                            "visible": vis[i],
                            "action": pred_actions[i]
                        }
                    
                    with open(out_dir / fp.name, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    hits = sum(1 for a in pred_actions if a == "hit")
                    bounces = sum(1 for a in pred_actions if a == "bounce")
                    total_hits += hits
                    total_bounces += bounces
                    count += 1
                    
                    print(f"✓ {fp.name}: {hits} hits, {bounces} bounces", end='\r')
                    
                except Exception as e:
                    print(f"\n⚠️  Ignoré {fp.name}: {e}")

            print(f"\n\n✅ Batch terminé!")
            print(f"   Fichiers traités: {count}/{len(files)}")
            print(f"   Total détections: {total_hits} hits, {total_bounces} bounces")
            print(f"   Dossier de sortie: {out_dir}")
            return

if __name__ == "__main__":
    import numpy as np
    main()