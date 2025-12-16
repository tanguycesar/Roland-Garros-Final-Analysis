from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# --- IMPORTS DEPUIS LE PACKAGE hit_n_bounce ---
# Python trouve automatiquement le dossier s'il est au même niveau que main.py
try:
    from hit_n_bounce.data_loader import load_ball_json, save_ball_json, iter_point_files
    from hit_n_bounce.unsupervised import unsupervised_hit_bounce_detection
    from hit_n_bounce.supervised import train_supervised, supervised_hit_bounce_detection
except ImportError as e:
    print(f" Erreur d'import critique : {e}")
    print("Vérifiez que le dossier 'hit_n_bounce' contient bien data_loader.py, supervised.py, etc.")
    sys.exit(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hit & Bounce detection (Roland-Garros 2025).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- TRAIN ---
    p_train = sub.add_parser("train", help="Train supervised model.")
    p_train.add_argument("--points_dir", required=True, help="Folder containing JSON data.")
    p_train.add_argument("--model_path", default="models/supervised_model.joblib", help="Output model path.")
    p_train.add_argument("--n_splits", type=int, default=5)

    # --- PREDICT ---
    p_pred = sub.add_parser("predict", help="Predict hits/bounces.")
    p_pred.add_argument("--method", choices=["unsupervised", "supervised"], required=True)
    p_pred.add_argument("--model_path", default="models/supervised_model.joblib", help="Model path (supervised only).")
    p_pred.add_argument("--input", help="Single JSON file.")
    p_pred.add_argument("--input_dir", help="Folder of JSON files.")
    p_pred.add_argument("--output", help="Output file path.")
    p_pred.add_argument("--output_dir", default="outputs", help="Output folder.")

    return p.parse_args()

# --- WRAPPERS EXIGÉS PAR L'EXERCICE ---

def unsupervised_hit_bounce_detection_fn(ball_data_json_path: str, output_path: str | None = None) -> dict:
    data = load_ball_json(ball_data_json_path)
    out = unsupervised_hit_bounce_detection(data)
    if output_path:
        save_ball_json(out, output_path)
    return out

def supervized_hit_bounce_detection_fn(ball_data_json_path: str, model_path: str, output_path: str | None = None) -> dict:
    data = load_ball_json(ball_data_json_path)
    out = supervised_hit_bounce_detection(data, model_path=model_path)
    if output_path:
        save_ball_json(out, output_path)
    return out

# --- LOGIQUE PRINCIPALE ---

def main() -> None:
    args = parse_args()

    # 1. ENTRAÎNEMENT
    if args.cmd == "train":
        print(f" Training on: {args.points_dir}")
        try:
            info = train_supervised(points_dir=args.points_dir, model_path=args.model_path, n_splits=args.n_splits)
            print("\n Training Complete!")
            for k, v in info.items():
                print(f"  - {k}: {v}")
        except Exception as e:
            print(f" Training Failed: {e}")
        return

    # 2. PRÉDICTION
    if args.cmd == "predict":
        if not args.input and not args.input_dir:
            sys.exit(" Error: Provide --input or --input_dir")

        # Mode Fichier Unique
        if args.input:
            inp = Path(args.input)
            out_path = Path(args.output) if args.output else inp.with_name(inp.stem + "_pred.json")
            
            print(f"Processing: {inp.name}")
            data = load_ball_json(inp)
            
            if args.method == "unsupervised":
                out = unsupervised_hit_bounce_detection(data)
            else:
                if not os.path.exists(args.model_path):
                    sys.exit(f" Model not found: {args.model_path}")
                out = supervised_hit_bounce_detection(data, model_path=args.model_path)
            
            save_ball_json(out, out_path)
            print(f"💾 Saved to: {out_path}")
            return

        # Mode Dossier (Batch)
        if args.input_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            files = iter_point_files(args.input_dir)
            
            print(f"Processing {len(files)} files from {args.input_dir}...")
            count = 0
            for fp in files:
                try:
                    data = load_ball_json(fp)
                    if args.method == "unsupervised":
                        out = unsupervised_hit_bounce_detection(data)
                    else:
                        out = supervised_hit_bounce_detection(data, model_path=args.model_path)
                    
                    save_ball_json(out, out_dir / fp.name)
                    count += 1
                    print(f"Processed: {fp.name}", end='\r')
                except Exception as e:
                    print(f"\n Skipped {fp.name}: {e}")

            print(f"\n Batch complete! {count} files saved to: {out_dir}")
            return

if __name__ == "__main__":
    main()