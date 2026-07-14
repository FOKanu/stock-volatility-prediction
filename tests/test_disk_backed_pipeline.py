from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_preprocessing import StreamingMoments
from tf_data import load_sequence_bundle
from tf_notebook_utils import make_tf_datasets, predict_in_batches


class DiskBackedPipelineTests(unittest.TestCase):
    def test_streaming_moments_match_numpy(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(37, 5, 3)).astype(np.float32)
        moments = StreamingMoments(3)
        moments.update(X[:11])
        moments.update(X[11:29])
        moments.update(X[29:])
        mean, std = moments.finalize()
        flat = X.reshape(-1, 3)
        np.testing.assert_allclose(mean, flat.mean(axis=0), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(std, flat.std(axis=0), rtol=1e-6, atol=1e-6)

    def test_loader_prefers_memmap_over_stale_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = "stocks_universe_sequences_plus5.npz"
            bundle_dir = root / requested.replace(".npz", "")
            bundle_dir.mkdir()
            arrays = {
                "X_train": np.ones((8, 4, 2), dtype=np.float32),
                "y_train": np.arange(8, dtype=np.float32),
                "X_val": np.ones((3, 4, 2), dtype=np.float32),
                "y_val": np.arange(3, dtype=np.float32),
                "X_test": np.ones((2, 4, 2), dtype=np.float32),
                "y_test": np.arange(2, dtype=np.float32),
            }
            for name, values in arrays.items():
                np.save(bundle_dir / f"{name}.npy", values)
            manifest = {
                "format": "memmap_v1",
                "feature_cols": ["a", "b"],
                "lookback_days": 4,
                "n_features": 2,
                "n_symbols": 9,
                "splits": {
                    split: {
                        "X": f"X_{split}.npy",
                        "y": f"y_{split}.npy",
                        "shape": list(arrays[f"X_{split}"].shape),
                        "dtype": "float32",
                    }
                    for split in ("train", "val", "test")
                },
            }
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
            cfg = {"n_symbols": 9, "target_type": "abs_return", "lookback_days": 4}
            scaler = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
            (root / "pipeline_config_stocks_universe_plus5.json").write_text(
                json.dumps(cfg)
            )
            (root / "scaler_stocks_universe_plus5.json").write_text(json.dumps(scaler))
            # A stale NPZ exists, but the complete manifest directory must win.
            np.savez(
                root / requested,
                X_train=np.zeros((1, 4, 2), dtype=np.float32),
                y_train=np.zeros(1, dtype=np.float32),
                X_val=np.zeros((1, 4, 2), dtype=np.float32),
                y_val=np.zeros(1, dtype=np.float32),
                X_test=np.zeros((1, 4, 2), dtype=np.float32),
                y_test=np.zeros(1, dtype=np.float32),
                feature_cols=np.array(["a", "b"]),
            )
            bundle = load_sequence_bundle(root, sequences_name=requested)
            self.assertTrue(bundle.is_disk_backed)
            self.assertIsInstance(bundle.X_train, np.memmap)
            self.assertEqual(bundle.X_train.shape, (8, 4, 2))
            self.assertEqual(bundle.cfg["n_symbols"], 9)

    def test_interrupted_staging_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requested = "stocks_universe_sequences_plus5.npz"
            interrupted = root / ".stocks_universe_sequences_plus5.interrupted"
            interrupted.mkdir()
            np.save(interrupted / "X_train.npy", np.zeros((2, 4, 2), dtype=np.float32))
            arrays = {
                "X_train": np.zeros((1, 4, 2), dtype=np.float32),
                "y_train": np.zeros(1, dtype=np.float32),
                "X_val": np.zeros((1, 4, 2), dtype=np.float32),
                "y_val": np.zeros(1, dtype=np.float32),
                "X_test": np.zeros((1, 4, 2), dtype=np.float32),
                "y_test": np.zeros(1, dtype=np.float32),
                "feature_cols": np.array(["a", "b"]),
            }
            np.savez(root / requested, **arrays)
            (root / "pipeline_config_stocks_universe_plus5.json").write_text(
                json.dumps({"n_symbols": 1})
            )
            (root / "scaler_stocks_universe_plus5.json").write_text(
                json.dumps({"mean": [0.0, 0.0], "std": [1.0, 1.0]})
            )
            bundle = load_sequence_bundle(root, sequences_name=requested)
            self.assertFalse(bundle.is_disk_backed)
            self.assertEqual(bundle.X_train.shape, (1, 4, 2))

    def test_memmap_tf_batches_and_chunked_prediction(self) -> None:
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X.npy"
            y_path = Path(tmp) / "y.npy"
            X_write = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(17, 4, 2)
            )
            X_write[:] = np.arange(17 * 4 * 2, dtype=np.float32).reshape(17, 4, 2)
            X_write.flush()
            y_write = np.lib.format.open_memmap(
                y_path, mode="w+", dtype=np.float32, shape=(17,)
            )
            y_write[:] = np.arange(17, dtype=np.float32)
            y_write.flush()
            X = np.load(path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            train_ds, _ = make_tf_datasets(X, y, X, y, batch_size=5, disk_block_rows=8)
            batches = list(train_ds.as_numpy_iterator())
            self.assertEqual(sum(len(batch_y) for _, batch_y in batches), 17)

            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input((4, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1),
                ]
            )
            predictions = predict_in_batches(model, X, batch_size=5)
            self.assertEqual(predictions.shape, (17, 1))


if __name__ == "__main__":
    unittest.main()
