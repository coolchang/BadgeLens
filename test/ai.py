import torch

# 1차원 텐서 생성
tensor_1d = torch.tensor([10, 20, 30, 40, 50])

print(tensor_1d)
print(f'차원 수: {tensor_1d.ndimension()}')
print(f'크기: {tensor_1d.size()}')
