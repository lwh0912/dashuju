"""Verify imports and a small image-model forward/backward pass."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU checks on teammates' non-GPU computers")
    args = parser.parse_args()
    print("Interpreter:", sys.executable)
    print("Python:", sys.version)
    try:
        import torch
        import torchvision
    except ImportError as exc:
        print("Missing dependency:", exc)
        print("Install requirements-cu128.txt, then requirements.txt, in your project environment.")
        return 1
    print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
    print("CUDA runtime:", torch.version.cuda, "available:", torch.cuda.is_available())
    print("cuDNN:", torch.backends.cudnn.version())
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("FAIL: GPU not available. Use --allow-cpu only for a CPU check.")
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    model = torchvision.models.resnet18(weights=None, num_classes=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    images = torch.randn(2, 3, 64, 64, device=device)
    targets = torch.tensor([[0.0], [1.0]], device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"PASS: forward/backward/optimizer step on {device}; no pretrained weights downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
