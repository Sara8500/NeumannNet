import torch
import torchvision
from src.torchvision_transforms import transform_train
from src.blur_operators_cifar import blur_model_simple, corruption_model_add_gaussian_noise, blur_gramian, identity
from src.neumann_network import NeumannNetwork
from datetime import datetime
import os


def main():
    #### Parameters: ####

    operator_name = "burring_and_AWGN" # or AWGN, or burring_and_AWGN
    dataset_root_dir = "./data"
    batchsize = 128     # 1
    corruption_model = corruption_model_add_gaussian_noise
    forward_adjoint= blur_model_simple
    forward_gramian= blur_gramian
    learning_rate = 1e-3
    B = 6
    number_of_epochs_to_train = 5



    #### driver ####
    # Load CIFAR10 data:
    trainset = torchvision.datasets.CIFAR10(dataset_root_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=1)

    # Create Neumann Network instance for blur/deblur operation
    NeumannNet = NeumannNetwork(num_blocks=B, learning_rate=learning_rate, forward_adjoint=forward_adjoint,
                                forward_gramian=forward_gramian, corruption_model=corruption_model)

    # Train Neumann Network
    for epoch in range(0, number_of_epochs_to_train):
        checkpoint_name = "operator_{}_train_cifar10_v_{}.ckpt".format(operator_name, epoch)
        print("**** training epoch ", epoch+1)
        NeumannNet.train(trainloader)
        print("finished training, saving checkpoint...")
        checkpoint_dir = './checkpoints/{}_checkpoints'.format(operator_name)
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        state = {
            'net': NeumannNet.netR.state_dict(),
            'datetime': datetime.now(),
            'batchsize': batchsize,
            'dataset': 'CIFAR10'
        }

        # Save state of Neumann Network (state of trained regularizer "netR")
        torch.save(state, os.path.join(checkpoint_dir, checkpoint_name))

    print("checkpoint successfully saved. ")


if __name__ == '__main__':
    main()