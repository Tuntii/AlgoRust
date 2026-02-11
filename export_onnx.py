#!/usr/bin/env python3
import json
from pathlib import Path

import torch

from train_lstm import LSTMFilter, FEATURE_COLUMNS


def main() -> None:
    models_dir = Path("models")
    meta_path = models_dir / "lstm_meta.json"
    model_path = models_dir / "lstm_filter.pt"
    onnx_path = models_dir / "lstm_filter.onnx"

    if not meta_path.exists() or not model_path.exists():
        raise SystemExit("Missing model files. Run train_lstm.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lookback = int(meta["lookback"])
    feature_count = len(meta.get("feature_columns", FEATURE_COLUMNS))
    hidden_dim = meta.get("hidden_dim", 256)
    num_layers = meta.get("num_layers", 3)
    bidirectional = meta.get("bidirectional", True)
    num_heads = meta.get("num_heads", 4)

    model = LSTMFilter(
        input_dim=feature_count,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,  # No dropout for inference
        bidirectional=bidirectional,
        num_heads=num_heads,
    )

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, lookback, feature_count, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    print(f"Saved ONNX to {onnx_path}")
    print(f"Architecture: input={feature_count}, hidden={hidden_dim}, "
          f"layers={num_layers}, heads={num_heads}, bidir={bidirectional}, "
          f"lookback={lookback}")


if __name__ == "__main__":
    main()
