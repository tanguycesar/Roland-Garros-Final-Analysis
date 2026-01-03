"""
Script to generate predictions on all ball data files using both methods.
This will create enriched JSON files with predicted actions.
"""

import json
from pathlib import Path
from tqdm import tqdm
from main import unsupervised_hit_bounce_detection, supervised_hit_bounce_detection

def generate_all_predictions():
    """Generate predictions for all ball data files using both methods."""
    
    # Input and output directories
    input_dir = Path("Data hit & bounce/per_point_v2")
    output_dir_unsup = Path("predictions/unsupervised")
    output_dir_sup = Path("predictions/supervised")
    
    # Create output directories
    output_dir_unsup.mkdir(parents=True, exist_ok=True)
    output_dir_sup.mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files
    json_files = sorted(input_dir.glob("ball_data_*.json"))
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    # Model path for supervised method
    model_path = "models/tennis_event_classifier.joblib"
    
    # Process each file
    print("=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)
    
    for json_file in tqdm(json_files, desc="Processing files"):
        file_name = json_file.name
        
        try:
            # Unsupervised method
            result_unsup = unsupervised_hit_bounce_detection(json_file)
            output_unsup = output_dir_unsup / file_name
            with open(output_unsup, 'w') as f:
                json.dump(result_unsup, f, indent=2)
            
            # Supervised method
            result_sup = supervised_hit_bounce_detection(json_file, model_path)
            output_sup = output_dir_sup / file_name
            with open(output_sup, 'w') as f:
                json.dump(result_sup, f, indent=2)
                
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("PREDICTIONS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Unsupervised predictions: {output_dir_unsup}/")
    print(f"Supervised predictions: {output_dir_sup}/")
    print(f"Total files processed: {len(json_files)}")

def generate_statistics():
    """Generate statistics on predictions."""
    
    output_dir_unsup = Path("predictions/unsupervised")
    output_dir_sup = Path("predictions/supervised")
    
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    
    # Unsupervised statistics
    total_hits_unsup = 0
    total_bounces_unsup = 0
    total_frames_unsup = 0
    
    for json_file in output_dir_unsup.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        for frame_data in data.values():
            total_frames_unsup += 1
            action = frame_data.get("action", "air")
            if action == "hit":
                total_hits_unsup += 1
            elif action == "bounce":
                total_bounces_unsup += 1
    
    print(f"\nUnsupervised Method:")
    print(f"  Total frames: {total_frames_unsup}")
    print(f"  Hits detected: {total_hits_unsup}")
    print(f"  Bounces detected: {total_bounces_unsup}")
    print(f"  Air frames: {total_frames_unsup - total_hits_unsup - total_bounces_unsup}")
    
    # Supervised statistics
    total_hits_sup = 0
    total_bounces_sup = 0
    total_frames_sup = 0
    
    for json_file in output_dir_sup.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        for frame_data in data.values():
            total_frames_sup += 1
            action = frame_data.get("action", "air")
            if action == "hit":
                total_hits_sup += 1
            elif action == "bounce":
                total_bounces_sup += 1
    
    print(f"\nSupervised Method:")
    print(f"  Total frames: {total_frames_sup}")
    print(f"  Hits detected: {total_hits_sup}")
    print(f"  Bounces detected: {total_bounces_sup}")
    print(f"  Air frames: {total_frames_sup - total_hits_sup - total_bounces_sup}")

if __name__ == "__main__":
    # Install tqdm if not available
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bar...")
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
    
    generate_all_predictions()
    generate_statistics()
    
    print("\n" + "=" * 60)
    print("DONE! Predictions ready for GitHub upload.")
    print("=" * 60)
