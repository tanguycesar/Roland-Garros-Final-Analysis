# Hit & Bounce Detection — Roland-Garros 2025 Final

This repository implements **two pipelines** to detect **tennis ball hits** and **bounces** from the ball (x,y) time series:

- **Unsupervised (physics-inspired heuristics)**: no labels used.
- **Supervised (ML baseline)**: trains on the provided `action` labels and predicts `pred_action`.

The expected output is the original JSON enriched with:

```json
"pred_action": "hit" | "bounce" | "air"
```

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2) Data layout (recommended)

**Do NOT commit the raw dataset/video to Git.**

Place the provided point JSONs under:

```
data/raw/points/
  point_0001.json
  ...
```

(Any nested structure is fine: the code scans recursively.)

## 3) Train the supervised model

```bash
python main.py train --points_dir data/raw/points --model_path models/supervised_model.joblib
```

This saves a `joblib` payload containing the sklearn pipeline + feature config.

## 4) Predict on a single point

Unsupervised:

```bash
python main.py predict --method unsupervised --input data/raw/points/point_0001.json
```

Supervised:

```bash
python main.py predict --method supervised --model_path models/supervised_model.joblib --input data/raw/points/point_0001.json
```

## 5) Predict on a folder

```bash
python main.py predict --method supervised --model_path models/supervised_model.joblib --input_dir data/raw/points --output_dir outputs/supervised_preds
```

## Notes on the methods

### Unsupervised
- Estimates the "ground line" from the 95th percentile of y (image coordinates).
- **Bounce**: near ground + vertical velocity sign change (+ → −) + acceleration spike.
- **Hit**: strong direction change / speed jump away from ground.

### Supervised baseline
- Context-window features: x/y, velocities, accelerations, speed, turn-rate, distance to ground (with ±2 frame context).
- Class-weighted **Logistic Regression** (fast, strong baseline, easy to serialize).
- GroupKFold CV by point to avoid leakage.

## What to submit
- `main.py` (contains both methods)
- `requirements.txt`
- `models/supervised_model.joblib` after training
- Any extra code files needed to run
