import torch

b,h,n,d = 4, 20, 1024, 64
tensor_1d = torch.randn(d).cuda()
tensor_2d = torch.randn(n,d).cuda()
tensor_3d = torch.randn(h,n,d).cuda()
tensor_4d = torch.randn(b,h,n,d).cuda()