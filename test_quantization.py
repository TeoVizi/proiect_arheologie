import numpy as np
import quantization

weights = np.random.uniform(-2, 2, size=100).astype(np.float32)
quantized_weights, scale, zero_point = quantization.quantize(weights)
print("Weights:", weights, '\n')
print("Quantized Weights:", quantized_weights, '\n')
print("Scale:", scale, "Zero Point:", zero_point, '\n')
print('\n')

activations = np.random.uniform(0, 1, size=100).astype(np.float32)
quantized_activations, scale, zero_point = quantization.quantize(activations)
print("Activations:", activations, '\n')
print("Quantized Activations:", quantized_activations, '\n')
print("Scale:", scale, "Zero Point:", zero_point, '\n')
