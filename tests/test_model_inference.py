import pytest
import numpy as np
from src.model_inference import TensorRTInferencer

@pytest.fixture(scope="module")
def inferencer():
    engine_path = "models/regime_filter_lite.plan"
    infer = TensorRTInferencer(engine_path)
    yield infer
    infer.destroy()

def test_tensorrt_inferencer_predict(inferencer):
    # Create a sample input (batch size 1, 5 features)
    input_data = np.random.rand(1, 5).astype(np.float32)
    output = inferencer.predict(input_data)
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.shape == (1, 2), f"Output shape should be (1, 2), got {output.shape}" 