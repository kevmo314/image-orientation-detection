import torch
from run import Network

dummy_input = torch.randn(1, 1, 18, 32, device="cuda")
model = torch.load('model.pt')

input_names = [ "image" ]
output_names = [ "inversion_class" ]

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
