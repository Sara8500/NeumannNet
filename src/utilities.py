import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io as io
import h5py
import numpy as np

# show tensor as an image
# remove leading 1 dimensions before plotting (example: [1,3,32,32])
def show_tensor_image(input_tensor, num_channels=3, img_size=(32, 32)):
    print("show_tensor_image: input tensor shape = ", input_tensor.shape)

    reshaped_tensor = input_tensor.view(num_channels, img_size[0], img_size[1])

    print("show_tensor_image: tensor shape after reshaping: ", reshaped_tensor.shape)

    trans = transforms.ToPILImage()

    pic_PIL = trans(reshaped_tensor)
    plt.imshow(pic_PIL)
    plt.show()

# reading the mat_files
mat_file = "/home/abbasis/projects/NeumannNetworkPytorch/knee/axial_t2/1/rawdata14.mat"
with h5py.File(mat_file, 'r') as f:
    f.keys()

print(f.keys())


arrays = {}
f = h5py.File(mat_file)
for k, v in f.items():
    arrays[k] = np.array(v)

print(arrays)