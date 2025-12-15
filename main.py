from __future__ import annotations

import argparse
from pathlib import Path

from hit_n_bounce.io import load_ball_json, save_ball_json, iter_point_files
from hit_n_bounce.unsupervised import unsupervised_hit_bounce_detection
from hit_n_bounce.supervised import train_supervised, supervised_hit_bounce_detection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hit & Bounce detection (unsupervised + supervised).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train supervised model on a folder of point jsons.")
    p_train.add_argument("--points_dir", required=True, help="Folder containing point json files (recursively).")
    p_train.add_argument("--model_path", default="models/supervised_model.joblib", help="Where to save model.")
    p_train.add_argument("--n_splits", type=int, default=5)

    p_pred = sub.add_parser("predict", help="Predict on a single json or a folder.")
    p_pred.add_argument("--method", choices=["unsupervised", "supervised"], required=True)
    p_pred.add_argument("--model_path", default="models/supervised_model.joblib",
                        help="Required for supervised.")
    p_pred.add_argument("--input", help="Path to a single point json.")
    p_pred.add_argument("--input_dir", help="Folder containing point json files (recursively).")
    p_pred.add_argument("--output", help="Output json path (single input).")
    p_pred.add_argument("--output_dir", default="outputs", help="Output folder (folder input).")

    return p.parse_args()


# --- REQUIRED BY THE EXERCISE: two public functions ---
def unsupervised_hit_bounce_detection_fn(ball_data_json_path: str, output_path: str | None = None) -> dict:
    ball_data = load_ball_json(ball_data_json_path)
    out = unsupervised_hit_bounce_detection(ball_data)
    if output_path is not None:
        save_ball_json(out, output_path)
    return out


def supervized_hit_bounce_detection_fn(ball_data_json_path: str, model_path: str, output_path: str | None = None) -> dict:
    ball_data = load_ball_json(ball_data_json_path)
    out = supervised_hit_bounce_detection(ball_data, model_path=model_path)
    if output_path is not None:
        save_ball_json(out, output_path)
    return out


def main() -> None:
    args = parse_args()

    if args.cmd == "train":
        info = train_supervised(points_dir=args.points_dir, model_path=args.model_path, n_splits=args.n_splits)
        print("Training done:")
        for k, v in info.items():
            print(f"  - {k}: {v}")
        return

    if args.cmd == "predict":
        if args.input is None and args.input_dir is None:
            raise SystemExit("Provide --input or --input_dir")

        if args.input:
            inp = Path(args.input)
            out_path = Path(args.output) if args.output else inp.with_name(inp.stem + "_pred.json")
            data = load_ball_json(inp)
            if args.method == "unsupervised":
                out = unsupervised_hit_bounce_detection(data)
            else:
                out = supervised_hit_bounce_detection(data, model_path=args.model_path)
            save_ball_json(out, out_path)
            print(f"Saved: {out_path}")
            return

        # folder mode
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = iter_point_files(args.input_dir)
        for fp in files:
            data = load_ball_json(fp)
            if args.method == "unsupervised":
                out = unsupervised_hit_bounce_detection(data)
            else:
                out = supervised_hit_bounce_detection(data, model_path=args.model_path)
            save_ball_json(out, out_dir / fp.name)
        print(f"Saved {len(files)} files to: {out_dir}")
        return


if __name__ == "__main__":
    main()
