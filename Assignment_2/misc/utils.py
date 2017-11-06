import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import imsave


# They following functions save images in a grid
def save_images(images, size, image_path):
    return imsave(image_path, merge(images, size))


def merge(images, size, spacing=5):
    h, w = images.shape[1], images.shape[2]
    img = np.ones(((h+spacing) * size[0], (w+spacing) * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h+j*spacing:j*h+j*spacing+h, i*w+i*spacing:i*w+i*spacing+w] = image

    return img


def save_plot(vals, name, label="train_error", xlabel="epoch", ylabel="Cross Entropy Loss"):
    plt.plot(vals, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig("plots/"+name, dpi=300)
    plt.clf()
