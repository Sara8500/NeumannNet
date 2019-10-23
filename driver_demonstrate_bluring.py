import torch
import torchvision
from src.torchvision_transforms import transform_train, transform_to_tensor_only

from src.utilities import show_tensor_image
from src.blur_operators_cifar import blur_model_simple, blur_gramian, corruption_model_add_gaussian_noise


def main():
    #### Parameters: ####
    dataset_root_dir = "./data"
    transform = transform_train    # two options: transform_train or transform_to_tensor
    num_pictures = 3

    ####################
    # Load CIFAR10 data:
    batchsize = 1
    trainset = torchvision.datasets.CIFAR10(dataset_root_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1)

    for batch_id, (pic, target) in enumerate(trainloader):

        # show an original CIFAR10 pic (random order)
        show_tensor_image(pic, 3, (32, 32))

        # blur the picture and show it again
        pic_blur = blur_model_simple(pic)
        print(pic.shape)
        print(pic_blur.shape)
        show_tensor_image(pic_blur, 3, (32, 32))

        # add noise to the original picture and show it
        pic_noisy = corruption_model_add_gaussian_noise(pic)
        show_tensor_image(pic_noisy)

        # break after num_pictures
        if batch_id >= num_pictures-1:
            break


if __name__ == '__main__':
    main()