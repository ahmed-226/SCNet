"""Shared I/O helpers: CSV loading, JSON centroid parsing, subject loading."""

import json
import pandas as pd
import nibabel as nib


def is_valid_path(val):
    """Return True if *val* is a non-empty, non-NaN string."""
    if val is None:
        return False
    if isinstance(val, float):
        return False
    return isinstance(val, str) and val.strip() != ""


def parse_centroid_json(json_path):
    """Parse a VerSe centroid JSON file.

    Returns
    -------
    direction : list[str]
        Axis direction labels, e.g. ``['P', 'I', 'R']``.
    centroids : dict[int, tuple[float, float, float]]
        ``{label: (X, Y, Z)}`` in the original image's coordinate space.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    direction = data[0].get("direction", [])
    centroids = {}
    for entry in data[1:]:
        label = int(entry["label"])
        centroids[label] = (float(entry["X"]), float(entry["Y"]), float(entry["Z"]))
    return direction, centroids


def load_subject(row):
    """Load a single subject from a CSV *row* (a ``pandas.Series``).

    Returns a dict with ``nibabel`` image objects and parsed centroids.
    Raises ``ValueError`` when required path columns are missing.
    """
    required = {
        "image_path": row.get("image_path"),
        "centroid_json_path": row.get("centroid_json_path"),
    }
    for col, val in required.items():
        if not is_valid_path(val):
            raise ValueError(
                f"Column '{col}' is missing or NaN for subject '{row.get('name', '?')}'"
            )

    volume_nib = nib.load(str(row["image_path"]))

    mask_nib = None
    if is_valid_path(row.get("mask_path")):
        mask_nib = nib.load(str(row["mask_path"]))

    direction, centroids = parse_centroid_json(str(row["centroid_json_path"]))
    return {
        "name": row["name"],
        "type": row["type"],
        "volume": volume_nib,
        "mask": mask_nib,
        "direction": direction,
        "centroids": centroids,
    }


def load_split(csv_path, split):
    """Load all subjects for a given split from the CSV.

    Returns a list of subject dicts (see :func:`load_subject`).
    """
    df = pd.read_csv(csv_path)
    df_split = df[df["type"] == split].reset_index(drop=True)

    subjects = []
    for _, row in df_split.iterrows():
        try:
            subjects.append(load_subject(row))
        except (ValueError, FileNotFoundError):
            pass
    return subjects
