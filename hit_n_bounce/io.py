from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List


def load_ball_json(path: str | Path) -> Dict[str, Any]:
    """Load a single point json (frame-indexed dict)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys to strings
    return {str(k): v for k, v in data.items()}


def save_ball_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_point_files(points_dir: str | Path) -> List[Path]:
    points_dir = Path(points_dir)
    if not points_dir.exists():
        raise FileNotFoundError(f"points_dir not found: {points_dir}")
    files = sorted([p for p in points_dir.rglob("*.json") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .json files found under: {points_dir}")
    return files


def extract_series(ball_data: Dict[str, Any]) -> Tuple[List[int], List[float], List[float], List[bool], List[str]]:
    """Return aligned arrays sorted by frame number."""
    frames = sorted([int(k) for k in ball_data.keys()])
    xs, ys, vis, act = [], [], [], []
    for fr in frames:
        d = ball_data[str(fr)]
        xs.append(float(d.get("x", float("nan"))))
        ys.append(float(d.get("y", float("nan"))))
        vis.append(bool(d.get("visible", True)))
        act.append(str(d.get("action", "air")))
    return frames, xs, ys, vis, act
