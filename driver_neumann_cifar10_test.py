import torch
import torchvision
from src.torchvision_transforms import transform_train
import matplotlib.pyplot as plt
from src.blur_operators_cifar import blur_model_simple, corruption_model_add_gaussian_noise, blur_gramian, identity
from src.neumann_network import NeumannNetwork


def main():


    #### Parameters: ####
    dataset_root_dir = "./data"
    batchsize = 1     # 1
    num_test_imgs = 3
    B = 6
    learning_rate = 1e-3
    #checkpoint_name = "checkpoints/burring_and_AWGN_checkpoints/operator_burring_and_AWGN_train_cifar10_v_4.ckpt"
    checkpoint_name =  "checkpoints/AWGN_checkpoints/operator_AWGN_train_cifar10_v_29.ckpt"
    # two options : corruption_model_add_gaussian_noise or AWGN
    corruption_model = corruption_model_add_gaussian_noise
    forward_adjoint= identity
    forward_gramian= identity
    B = 6


    #### driver ####
    # Load CIFAR10 data:
    testset = torchvision.datasets.CIFAR10(dataset_root_dir, train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=1)

    # Create Neumann Network instance for blur/deblur operation
    NeumannNet = NeumannNetwork(num_blocks=B, learning_rate=learning_rate, forward_adjoint=forward_adjoint,
                                forward_gramian=forward_gramian, corruption_model=corruption_model)

    # load checkpoint
    loaded_checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    NeumannNet.netR.load_state_dict(loaded_checkpoint['net'])
    NeumannNet.netR.eval()
    storage_test_images, storage_distorted_images, storage_reconstructed_images, loss = NeumannNet.test(testloader, num_test_imgs)


    # Show results:
    for i in range(num_test_imgs):
        original_image = storage_test_images[i, :, :, :]
        original_image = original_image.transpose((1, 2, 0))
        distorted_image = storage_distorted_images[i, :, :, :]
        distorted_image = distorted_image.transpose((1, 2, 0))
        reconstructed_image = storage_reconstructed_images[i, :, :, :]
        reconstructed_image = reconstructed_image.transpose((1, 2, 0))

        fig, axs = plt.subplots(1, 3)
        fig.suptitle('comparison of original, distorted, reconstructed image')
        axs[0].imshow(original_image)
        axs[1].imshow(distorted_image)
        axs[2].imshow(reconstructed_image)

        print('min_original, max img', original_image.min())
        print('max_original img', original_image.max())
        print('min_distorted img ', distorted_image.min())
        print('max_distorted img ', distorted_image.max())
        print('min_reconstructed img', reconstructed_image.min())
        print('max_reconstructed img', reconstructed_image.max())

    plt.show()


if __name__ == '__main__':
    main()