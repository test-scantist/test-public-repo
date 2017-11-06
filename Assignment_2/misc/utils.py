import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import imsave


# They following functions save images in a grid
def save_images(images, size, image_path):
    return imsave(image_path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


def save_plot(vals, name, label="train_error", xlabel="epoch", ylabel="Cross Entropy Loss"):
    plt.plot(vals, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("plots/"+name, dpi=300)
