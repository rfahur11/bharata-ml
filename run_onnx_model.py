import onnxruntime as ort
import numpy as np

# 1. Load the ONNX model
model_path = "model.onnx"  # Ganti dengan path model ONNX Anda
session = ort.InferenceSession(model_path)

# 2. Get input and output details
input_name = session.get_inputs()[0].name  # Nama input pertama model
input_shape = session.get_inputs()[0].shape  # Shape dari input
input_type = session.get_inputs()[0].type  # Tipe data input
print(f"Model Input Name: {input_name}")
print(f"Model Input Shape: {input_shape}")
print(f"Model Input Type: {input_type}")

output_name = session.get_outputs()[0].name  # Nama output pertama model
print(f"Model Output Name: {output_name}")

# 3. Generate dummy input data (contoh)
# Sesuaikan dengan shape yang diminta model
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# 4. Run inference
outputs = session.run([output_name], {input_name: dummy_input})

# 5. Display the result
print("Model Output:")
print(outputs[0])
