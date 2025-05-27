#!/usr/bin/env python3

import torch
import os
from src.model_inference import RegimeFilterLite, export_pytorch_to_onnx

if __name__ == "__main__":
    # Determine device (cuda or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Instantiate RegimeFilterLite and move it to the device
    model = RegimeFilterLite().to(device)

    # Create dummy input tensor (batch size 1, 5 features) on the same device
    dummy_input_shape = (1, 5)
    dummy_input = torch.randn(dummy_input_shape, device=device)

    # Define output path (models/regime_filter_lite.onnx) and ensure models/ exists
    onnx_path = "models/regime_filter_lite.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Call export_pytorch_to_onnx
    export_pytorch_to_onnx(model, dummy_input_shape, onnx_path, opset_version=11)

    # Immediately clear PyTorch CUDA cache after export
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("INFO: Cleared PyTorch CUDA cache after ONNX export.")

    print("Export completed. (Check logs for details.)") 