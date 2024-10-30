"""
Exercise 1: Understanding Quantization
Task 1.1: Implement Fixed-Point Quantization
"""
import numpy as np

def fixed_point_quantization(tensor: np.ndarray, bits: int) -> np.ndarray:
    min_val = min(tensor)
    max_val = max(tensor)
    out = np.ndarray()
    for n in tensor:
        

        

    

tensor = np.array([0.5, -0.8, 0.1, 0.7, -0.5], dtype=np.float32)
quantized_tensor = fixed_point_quantization(tensor, 8)
print(quantized_tensor)
