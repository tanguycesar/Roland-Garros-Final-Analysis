"""
Hit & Bounce Detection - Roland-Garros
Author: Tanguy CESAR
Date: Janvier 2026

This module provides two main functions for detecting tennis hits and bounces:
1. unsupervised_hit_bounce_detection: Physics-based heuristic detection
2. supervised_hit_bounce_detection: Machine learning-based detection (XGBoost)

Both functions take a JSON file path as input and return an enriched JSON with detected events.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
from joblib import load

from hit_n_bounce import data_loader, features, supervised, unsupervised
from hit_n_bounce.features import FeatureConfig


def unsupervised_hit_bounce_detection(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect tennis hits and bounces using unsupervised physics-based heuristics.
    
    This method analyzes kinematic features (jerk, curvature, velocity changes) to identify
    sudden direction changes characteristic of hits and bounces.
    
    Parameters
    ----------
    json_path : str or Path
        Path to the input JSON file containing ball trajectory data.
        Format: {frame_id: {"x": float, "y": float, "visible": bool, "action": str}}
    
    Returns
    -------
    dict
        Enriched JSON with the same structure as input, but with "action" field updated:
        - "hit" for detected hits
        - "bounce" for detected bounces
        - "air" for frames in the air
        
    Example
    -------
    >>> result = unsupervised_hit_bounce_detection("ball_data_1.json")
    >>> print(result["32500"]["action"])  # "hit" or "bounce" or "air"
    
    Notes
    -----
    - No training required
    - Works well on clean trajectories
    - May require parameter tuning for different camera setups
    """
    json_path = Path(json_path)
    
    # Load and process data
    data = data_loader.load_ball_json(json_path)
    frames, xs, ys, vis, actions = data_loader.extract_series(data)
    
    # Convert to numpy arrays
    frames_arr = np.array(frames)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    # Check if we have valid data
    if len(frames_arr) == 0 or np.all(np.isnan(xs_arr)):
        return data
    
    # Process trajectory
    cfg = FeatureConfig(fps=50.0)
    enriched_data = data.copy()
    
    # Compute kinematic features
    kin = features.compute_kinematics(frames_arr, xs_arr, ys_arr, cfg)
    
    # Run unsupervised detection
    pred_actions = unsupervised.detect_tennis_events(frames_arr, kin, cfg.fps)
    
    # Update enriched data with detected events
    for i, frame_id in enumerate(frames_arr):
        frame_str = str(int(frame_id))
        if frame_str in enriched_data:
            enriched_data[frame_str]["action"] = pred_actions[i]
    
    return enriched_data


def supervised_hit_bounce_detection(
    json_path: Union[str, Path],
    model_path: Union[str, Path] = "models/tennis_event_classifier.joblib"
) -> Dict[str, Any]:
    """
    Detect tennis hits and bounces using supervised machine learning (XGBoost).
    
    This method uses a pre-trained XGBoost classifier that analyzes 28 kinematic features
    over an 11-frame window (±5 frames) to classify each frame as hit, bounce, or air.
    
    Parameters
    ----------
    json_path : str or Path
        Path to the input JSON file containing ball trajectory data.
        Format: {frame_id: {"x": float, "y": float, "visible": bool, "action": str}}
    model_path : str or Path, optional
        Path to the trained model file (.joblib format).
        Default: "models/tennis_event_classifier.joblib"
    
    Returns
    -------
    dict
        Enriched JSON with the same structure as input, but with "action" field updated:
        - "hit" for detected hits
        - "bounce" for detected bounces
        - "air" for frames in the air
    
    Example
    -------
    >>> result = supervised_hit_bounce_detection("ball_data_1.json")
    >>> print(result["32500"]["action"])  # "hit" or "bounce" or "air"
    
    Notes
    -----
    - Requires a pre-trained model (included in repository)
    - Higher accuracy than unsupervised method (F1 ~0.82 vs ~0.65)
    - Robust to noisy trajectories
    
    Raises
    ------
    FileNotFoundError
        If model_path does not exist
    """
    json_path = Path(json_path)
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model_data = load(model_path)
    model = model_data["model"]
    
    # Load and process data
    data = data_loader.load_ball_json(json_path)
    frames, xs, ys, vis, actions = data_loader.extract_series(data)
    
    # Convert to numpy arrays
    frames_arr = np.array(frames)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    
    # Check if we have valid data
    if len(frames_arr) == 0 or np.all(np.isnan(xs_arr)):
        return data
    
    # Process trajectory
    cfg = FeatureConfig(fps=50.0)
    enriched_data = data.copy()
    
    # Compute kinematic features
    kin = features.compute_kinematics(frames_arr, xs_arr, ys_arr, cfg)
    
    # Prepare features for ML model
    X, _ = supervised.make_frame_features(kin, cfg)
    
    # Predict with XGBoost
    probs = model.predict_proba(X)
    
    # Get class predictions (0=air, 1=hit, 2=bounce)
    pred_classes = np.argmax(probs, axis=1)
    class_to_label = {0: "air", 1: "hit", 2: "bounce"}
    
    # Update enriched data with detected events
    for i, frame_id in enumerate(frames_arr):
        frame_str = str(int(frame_id))
        if frame_str in enriched_data and i < len(pred_classes):
            enriched_data[frame_str]["action"] = class_to_label[pred_classes[i]]
    
    return enriched_data


def save_enriched_json(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save enriched JSON data to file.
    
    Parameters
    ----------
    data : dict
        Enriched trajectory data
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tennis Hit & Bounce Detection")
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("--method", choices=["unsupervised", "supervised"], 
                       default="supervised", help="Detection method")
    parser.add_argument("--model", default="models/tennis_event_classifier.joblib",
                       help="Model path for supervised method")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    
    args = parser.parse_args()
    
    print(f"Processing: {args.input}")
    print(f"Method: {args.method}")
    
    # Run detection
    if args.method == "unsupervised":
        result = unsupervised_hit_bounce_detection(args.input)
    else:
        result = supervised_hit_bounce_detection(args.input, args.model)
    
    # Count detected events
    hits = sum(1 for v in result.values() if v.get("action") == "hit")
    bounces = sum(1 for v in result.values() if v.get("action") == "bounce")
    
    print(f"Detected: {hits} hits, {bounces} bounces")
    
    # Save if output path provided
    if args.output:
        save_enriched_json(result, args.output)
    else:
        print("\nSample results (first 5 events):")
        count = 0
        for frame_id, frame_data in result.items():
            if frame_data.get("action") in ["hit", "bounce"]:
                print(f"  Frame {frame_id}: {frame_data['action']}")
                count += 1
                if count >= 5:
                    break
