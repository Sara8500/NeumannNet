import torchvision
import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.ToTensor()
    # todo: normalize
    ])

