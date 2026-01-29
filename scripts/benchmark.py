import torch
import onnxruntime as ort
import time
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.train_nn import CreditNet

def run_benchmark():
    input_size = 23
    data = np.random.randn(100, input_size).astype(np.float32)
    
    # Тест PyTorch
    pt_model = CreditNet(input_size)
    start = time.time()
    for _ in range(100):
        _ = pt_model(torch.from_numpy(data))
    pt_time = time.time() - start

    # Тест ONNX
    session = ort.InferenceSession("models/model.onnx")
    start = time.time()
    for _ in range(100):
        _ = session.run(None, {"input": data})
    onnx_time = time.time() - start

    print(f"PyTorch Time: {pt_time:.4f}s")
    print(f"ONNX Time: {onnx_time:.4f}s")
    print(f"Speedup: {pt_time/onnx_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()