"""The only place that builds an onnxruntime `InferenceSession`."""

from pathlib import Path

import onnxruntime as ort


def load_session(path: Path, threads: int) -> ort.InferenceSession:
    """A CPU session on `threads` intra-op threads; fails fast when the model file is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file {path} not found: run `make models`, or point CHECKBOX_MODELS_DIR at it"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
