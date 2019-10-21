import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
