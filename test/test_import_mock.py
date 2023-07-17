import torch

import cuda_mock

a = torch.rand((1,)).cuda()

print(a + a)

