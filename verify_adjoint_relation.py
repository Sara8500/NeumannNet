from src.blur_operators_cifar import blur_model_simple
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

image_shape = (1,3,32,32)

u = torch.randn(image_shape).to(device)
p = torch.randn(image_shape).to(device)

n = 3*32*32

# torch.dot neeeds two 1D tensors: https://discuss.pytorch.org/t/how-to-do-dot-product-of-two-tensors/3984
# reshape vs squeeze and unsqueeze:  https://deeplizard.com/learn/video/fCVuiW9AFzY

# < X.u , p >
res1 = torch.dot(blur_model_simple(u).reshape([1,n]).squeeze(),p.reshape([1,n]).squeeze())
print("res1: ", res1)

# < u , X'.p >
res2 = torch.dot(u.reshape([1,n]).squeeze(), blur_model_simple(p).reshape([1,n]).squeeze())
print("res2: ", res2)

print("error: ", res2-res1) # error very small, so res1=res2 up to some numerical accuracy

# conclusion: blur_model_simple equals its adjoint!
