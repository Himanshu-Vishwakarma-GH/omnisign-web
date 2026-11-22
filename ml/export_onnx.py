"""Train the BiLSTM on synthetic motion data and export ONNX (< 3 MB).

Usage:
    python export_onnx.py [--epochs 30] [--samples 400] [--hidden 64]

If the exported file exceeds 3 MB the script retries with hidden=48.
Verifies the ONNX file with onnx.checker and runs a python onnxruntime
inference pass on a synthetic 'Help' sample.
"""

import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()

    import torch

    from model import NUM_CLASSES, SIGNS, build_model, generate_sequence, normalize, train

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "web", "public", "models", "omnisign_bilstm.onnx"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hidden = args.hidden
    for attempt in range(2):
        model = train(epochs=args.epochs, samples_per_class=args.samples, hidden=hidden)
        model.eval()
        # Checkpoint first so an export failure never loses the trained weights.
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/bilstm.pt")
        print("saved checkpoints/bilstm.pt")
        dummy = torch.randn(1, 30, 48, 3)
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
        )
        size_kb = os.path.getsize(out_path) / 1024
        print(f"exported {out_path}: {size_kb:.1f} KB (hidden={hidden})")
        if size_kb < 3 * 1024:
            break
        print("over 3 MB budget — retrying with hidden=48")
        hidden = 48

    # --- verify with onnx checker + onnxruntime -------------------------
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("onnx.checker: OK")

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    seq = normalize(generate_sequence(0, seed=123))  # class 0 = Help
    logits = sess.run(None, {"input": seq[None, :, :, :]})[0][0]
    probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    pred = int(probs.argmax())
    print(
        f"onnxruntime inference on synthetic 'Help': predicted "
        f"'{SIGNS[pred]}' (idx {pred}) confidence {probs[pred]:.3f}"
    )
    print(f"final size: {os.path.getsize(out_path) / 1024:.1f} KB")
    assert os.path.getsize(out_path) < 3 * 1024 * 1024, "ONNX over 3 MB budget"
    assert pred == 0, f"expected Help (0), got {SIGNS[pred]}"
    print("EXPORT VERIFIED")


if __name__ == "__main__":
    main()
