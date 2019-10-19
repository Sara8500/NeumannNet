import torch
import torchvision
from src.torchvision_transforms import transform_train

from src.utilities import show_tensor_image
from src.blur_operators_cifar import blur_model_simple, add_gaussian_noise, blur_gramian
from src.neumann_network import NeumannNetwork

import torch.backends.cudnn as cudnn

from datetime import datetime
import os

def main():


    #### Parameters: ####
    dataset_root_dir = "./data"
    batchsize = 1     # 1

    learning_rate = 1e-3
    B = 6

    number_of_epochs_to_train = 1            # 1
    checkpoint_name = "train_cifar10_v0.ckpt"


    #### driver ####
    # Load CIFAR10 data:
    trainset = torchvision.datasets.CIFAR10(dataset_root_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1);


    for batch_id, (pic, target) in enumerate(trainloader):

        # show an original CIFAR10 pic (random order)
        show_tensor_image(pic, 3, (32,32))

        #blur the picture and show it again
        pic_blur = blur_model_simple(pic)
        print(pic.shape)
        print(pic_blur.shape)
        show_tensor_image(pic_blur,3, (32,32))

        #add noise to the original picture and show it
        pic_noisy = add_gaussian_noise(pic)
        show_tensor_image(pic_noisy)

        # repeat for 3 pictures
        if batch_id > 3:
            break




if __name__ == '__main__':
    main()