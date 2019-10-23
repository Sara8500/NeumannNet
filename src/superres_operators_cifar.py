import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transform


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

noise_stddev = 0.001
kernel_size = 5
sigma = 2
padding = 1
factor = 0.5
h, w = 64, 64


# first we learn down sampling and then solving inverse problem up sampling by learning
def sample_forward(input_tensor):
    downsampled = input_tensor.transform.Resize((int(factor*h), int(factor*w)), interpolation=2)
    return downsampled


def sample_gramian(input_tensor):
    downsampled = input_tensor.transform.Resize((int(factor * h), int(factor * w)), interpolation=2)
    upsampled = downsampled.transform.Resize((h, w))
    return upsampled


def gauss_blur_kernel(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/(g.sum())


def blur_gauss_model(input_tensor):
    blur_kernel = gauss_blur_kernel(kernel_size, sigma)
    blur_kernel_torch = torch.from_numpy(blur_kernel).to(device)
    blurred_tensor = F.conv2d(input_tensor, blur_kernel_torch, padding=padding)
    return blurred_tensor


def corruption_model_add_gaussian_noise(input_tensor):

    input_shape = tuple(input_tensor.shape)
    noise_tensor = noise_stddev * torch.randn(input_shape)
    noise_tensor = noise_tensor.to(device)
    return noise_tensor


def blur_gramian(input_tensor):
    return blur_gauss_model(blur_gauss_model(input_tensor))


def gaussian_blur_gramian(inpur_tensor):
    blur_gauss_model(blur_gauss_model(inpur_tensor))
    return



def identity(input_tensor):
    return input_tensor