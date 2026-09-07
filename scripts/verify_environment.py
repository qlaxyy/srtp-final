import torch
import torchaudio
import torchcodec
import torchvision
import triton
import vllm
import easysteer
from vllm.steer_vectors import ApplySpec, SteeringSpec, VectorSpec

assert torch.__version__ == "2.11.0+cu129"
assert torchvision.__version__ == "0.26.0+cu129"
assert torchaudio.__version__ == "2.11.0+cu129"
assert torchcodec.__version__ == "0.16.0+cu129"
assert triton.__version__ == "3.6.0"
assert torch.version.cuda == "12.9"
assert torch.cuda.is_available()
assert torchvision.extension._has_ops()
print("GPU:", torch.cuda.get_device_name(0))
print("vLLM:", vllm.__version__, vllm.__file__)
print("EasySteer:", easysteer.__file__)
print("Environment verification: OK")
