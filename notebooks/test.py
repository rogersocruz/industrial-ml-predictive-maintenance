import torch
print(torch.rand(5,3))
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

