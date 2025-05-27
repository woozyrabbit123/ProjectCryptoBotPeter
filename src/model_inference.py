"""
Model inference module for Project Crypto Bot Peter.
Handles PyTorch model definition, ONNX export, and TensorRT inference.
"""

import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Implement RegimeFilterLite model and TensorRTInferencer class 

class RegimeFilterLite(nn.Module):
    """
    A lightweight regime filter model for binary classification (stable=0, volatile=1) using PyTorch.
    Input: 5 features.
    Architecture:
      - Input layer (5 features)
      - One fully connected hidden layer (3 neurons, ReLU activation)
      - Output layer (2 neurons, logits)
    """

    def __init__(self):
        super(RegimeFilterLite, self).__init__()
        self.fc1 = nn.Linear(5, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_pytorch_to_onnx(model, dummy_input_shape, onnx_path, opset_version=11):
    """
    Exports a PyTorch model (RegimeFilterLite) to ONNX format.
    
    Args:
        model (nn.Module): An instance of the RegimeFilterLite model.
        dummy_input_shape (tuple): Shape of dummy input (e.g., (1, 5) for batch size 1, 5 features).
        onnx_path (str): Path (including filename) where the ONNX model is saved.
        opset_version (int, optional): ONNX opset version. Defaults to 11.
    """
    try:
        # Set model to evaluation mode
        model.eval()
        # Create dummy input tensor (matching dummy_input_shape and the model's device)
        dummy_input = torch.randn(dummy_input_shape, device=next(model.parameters()).device)
        # Export model using torch.onnx.export
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=opset_version
        )
        logger.info("Model exported to ONNX successfully at: %s", onnx_path)
    except Exception as e:
        logger.error("Error exporting model to ONNX: %s", str(e)) 

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInferencer:
    """
    Loads a TensorRT engine (.plan) and performs inference using PyCUDA.
    Handles engine/context loading, buffer allocation, and prediction.
    """
    def __init__(self, engine_path: str) -> None:
        """
        Initializes the TensorRT engine, execution context, and allocates host/device buffers.
        Args:
            engine_path (str): Path to the TensorRT engine file (.plan).
        """
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            logger.info(f"TensorRT engine loaded from {engine_path}")
            self.context = self.engine.create_execution_context()
            logger.info("Execution context created.")
            # Assume one input and one output
            self.input_idx = 0
            self.output_idx = 1
            self.input_name = self.engine.get_tensor_name(self.input_idx)
            self.output_name = self.engine.get_tensor_name(self.output_idx)
            self.input_shape = self.engine.get_tensor_shape(self.input_name)
            self.output_shape = self.engine.get_tensor_shape(self.output_name)
            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
            # Allocate pinned host buffers
            self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=self.input_dtype)
            self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=self.output_dtype)
            # Allocate device buffers
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            # Bindings list
            self.bindings = [int(self.d_input), int(self.d_output)]
            # CUDA stream
            self.stream = cuda.Stream()
            logger.info("Buffers and CUDA stream allocated.")
        except Exception as e:
            logger.error(f"Error initializing TensorRTInferencer: {e}")
            raise

    def predict(self, input_data_numpy: np.ndarray) -> np.ndarray:
        """
        Runs inference on input_data_numpy and returns the output as a NumPy array.
        Args:
            input_data_numpy (np.ndarray): Input data of shape (batch_size, 5).
        Returns:
            np.ndarray: Output of shape (batch_size, 2).
        """
        try:
            # Ensure input is C-contiguous and float32
            if not input_data_numpy.flags['C_CONTIGUOUS']:
                input_data_numpy = np.ascontiguousarray(input_data_numpy, dtype=np.float32)
            else:
                input_data_numpy = input_data_numpy.astype(np.float32)
            np.copyto(self.h_input, input_data_numpy.ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            return self.h_output.copy().reshape(self.output_shape)
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def destroy(self) -> None:
        """
        Explicitly free CUDA resources (optional, as pycuda.autoinit usually handles this).
        """
        try:
            if hasattr(self, "d_input") and self.d_input is not None and hasattr(self.d_input, 'free'):
                self.d_input.free()
            if hasattr(self, "d_output") and self.d_output is not None and hasattr(self.d_output, 'free'):
                self.d_output.free()
            # Do not call detach on the stream; pycuda.autoinit handles stream cleanup.
            logger.info("CUDA resources freed.")
        except Exception as e:
            logger.error(f"Error during destroy: {e}") 