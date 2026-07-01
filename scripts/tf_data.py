"""TensorFlow notebook helpers: load preprocessed sequences without PyTorch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paths import find_project_root

DEFAULT_SEQUENCES_NAME = "stocks_universe_sequences_plus5.npz"


def resolve_artifact_names(
    sequences_name: str,
    config_name: str | None = None,
    scaler_name: str | None = None,
) -> tuple[str, str, str]:
    stem = sequences_name.replace(".npz", "")
    if "_sequences" in stem:
        dataset_id, profile_suffix = stem.split("_sequences", 1)
        if dataset_id == "aapl":
            inferred_config = f"pipeline_config{profile_suffix}.json"
            inferred_scaler = f"scaler{profile_suffix}.json"
        else:
            inferred_config = f"pipeline_config_{dataset_id}{profile_suffix}.json"
            inferred_scaler = f"scaler_{dataset_id}{profile_suffix}.json"
    else:
        suffix = stem.replace("aapl_sequences", "")
        inferred_config = f"pipeline_config{suffix}.json"
        inferred_scaler = f"scaler{suffix}.json"
    return sequences_name, config_name or inferred_config, scaler_name or inferred_scaler


@dataclass
class SequenceBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: list[str]
    cfg: dict
    scaler: dict


def load_sequence_bundle(
    processed_dir: Path | None = None,
    sequences_name: str = DEFAULT_SEQUENCES_NAME,
    config_name: str | None = None,
    scaler_name: str | None = None,
) -> SequenceBundle:
    if processed_dir is None:
        processed_dir = find_project_root() / "data" / "processed"
    sequences_name, config_name, scaler_name = resolve_artifact_names(
        sequences_name, config_name, scaler_name
    )
    seq_path = processed_dir / sequences_name
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences not found at {seq_path}")

    data = np.load(seq_path, allow_pickle=True)
    feature_cols = [str(c) for c in data["feature_cols"]]
    with open(processed_dir / config_name) as f:
        cfg = json.load(f)
    with open(processed_dir / scaler_name) as f:
        scaler = json.load(f)

    return SequenceBundle(
        X_train=data["X_train"],
        y_train=data["y_train"].astype(np.float32),
        X_val=data["X_val"],
        y_val=data["y_val"].astype(np.float32),
        X_test=data["X_test"],
        y_test=data["y_test"].astype(np.float32),
        feature_cols=feature_cols,
        cfg=cfg,
        scaler=scaler,
    )
