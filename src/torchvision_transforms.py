import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49, 0.48, 0.44), (0.24, 0.24, 0.26))  # values for CIFAR-10
    ])

transform_to_tensor_only = transforms.Compose([
    transforms.ToTensor()
    ])