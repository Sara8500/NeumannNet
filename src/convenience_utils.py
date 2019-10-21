from sklearn import feature_extraction as fe
import os
import torchvision.transforms as transforms


def get_list_of_files(path_dir):

    return sorted([f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))])


def show_tensor(t):

    transform = transforms.Compose([
        transforms.ToPILImage()
        ])

    pil_image = transform(t)
    pil_image.show()


def generate_patches(image, patch_h, patch_w):
    pad = [[0, 0], [0, 0]]
    p_area = patch_h * patch_w
    image_ch = 3
    patches = fe.image.extract_patches(image, p_area, pad)

    return patches


def space_to_batch(_input, block_shape):
    resh_input = _input.reshape((_input.shape[0], int(_input.shape[1] / block_shape[0]), block_shape[0],
                                int(_input.shape[2] / block_shape[1]), block_shape[1], _input.shape[3]))
    perm_input = resh_input.transpose((2, 4, 0, 1, 3, 5))
    output = perm_input.reshape((block_shape[0] * block_shape[1] * _input.shape[0], int(_input.shape[1] / block_shape[0]),
                                int(_input.shape[2] / block_shape[1]), _input.shape[3]))
    return output





