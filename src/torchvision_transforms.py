import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.ToTensor()
    ])

transform_to_tensor_only = transforms.Compose([
    transforms.ToTensor()
    ])
