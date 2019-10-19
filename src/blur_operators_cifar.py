import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

blur_width = 3  # kernel size, should match padding in order to keep output dimensions
padding = 1
blur_factor = 0.08
noise_stddev = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_blur_kernel(blur_width, blur_factor):

    blur_kernel_2d = blur_factor * np.ones((blur_width, blur_width), dtype=np.float32)

    blur_kernel = np.ndarray((3, 3, blur_width, blur_width), dtype=np.float32)  # (number output layers, channels, h,w)
    tmp = np.ndarray((3, blur_width, blur_width), dtype=np.float32)

    active_dim = 0
    for j in range(0, 3):
        for i in range(0, 3):
            if active_dim == i:
                tmp[i, :, :] = blur_kernel_2d
            else:
                tmp[i, :, :] = np.zeros((blur_width, blur_width))
        blur_kernel[j, :, :, :] = tmp
        active_dim += 1

    blur_kernel_torch = torch.from_numpy(blur_kernel).to(device)

    return blur_kernel_torch


def blur_model_simple(input_tensor):

    blur_kernel_torch = get_blur_kernel(blur_width,blur_factor)
    blurred_tensor = F.conv2d(input_tensor,blur_kernel_torch,padding=padding)

    return blurred_tensor


def add_gaussian_noise(input_tensor):

    #print("device: ", device)

    #input_tensor = input_tensor.to(device)

    input_shape = tuple(input_tensor.shape);
    noise_tensor = noise_stddev * torch.randn(input_shape)
    noise_tensor = noise_tensor.to(device)

    #print("input shape", input_tensor.shape)
    #print("noise shape: ", noise_tensor.shape)

    return torch.add(input_tensor,noise_tensor)


def blur_gramian(input_tensor):
    return blur_model_simple(blur_model_simple(input_tensor))
