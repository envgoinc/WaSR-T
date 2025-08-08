import onnxruntime as ort
import numpy as np
import torch
import time

# Load ONNX model
session = ort.InferenceSession("wasr_t_unrolled.onnx", providers=["CUDAExecutionProvider"])

# Input names
input_names = {inp.name for inp in session.get_inputs()}
print("Inputs:", input_names)

# Dummy input
B, C, H, W = 1, 3, 384, 512
image = np.random.rand(B, C, H, W).astype(np.float32)
hist_images = np.random.rand(B, 5, C, H, W).astype(np.float32)

inputs = {
    "image": image,
    "hist_images": hist_images
}

# Warm-up
for _ in range(5):
    _ = session.run(["out"], inputs)

# Benchmark
print("\n🕒 Benchmarking ONNX Runtime...")
times = []
image = np.random.rand(B, C, H, W).astype(np.float32)
hist = np.random.rand(B, 5, C, H, W).astype(np.float32)
inputs["image"] = image
inputs["hist_images"] = hist
start = time.time()
_ = session.run(["out"], inputs)
end = time.time()
for _ in range(25):
    image = np.random.rand(B, C, H, W).astype(np.float32)
    hist = np.random.rand(B, 5, C, H, W).astype(np.float32)
    inputs["image"] = image
    inputs["hist_images"] = hist
    start = time.time()
    _ = session.run(["out"], inputs)
    end = time.time()
    times.append((end - start) * 1000)

avg_time = sum(times) / len(times)
print(f"✅ ONNX Runtime inference time: {avg_time:.2f} ms ({1000 / avg_time:.2f} FPS)")
