import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import _structural_similarity as ssim

from Evaluation import generate_image_from_vector


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float")[:, :, None] - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def compare_images(imageA, imageB, title):
    m = mse(imageA, imageB)
    s = ssim.compare_ssim(imageA, imageB)

    return [m, s]


def comparer(to_compare="search/dziekan-gajecki.jpg"):
    original = cv2.imread(to_compare)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    best_values = [sys.maxsize, 0, "img", "data"]
    best_values = [best_values] * 5
    min_mse = [sys.maxsize, "img", "data"]
    min_mse = [min_mse] * 5
    max_ssim = [0, "img", "data"]
    max_ssim = [max_ssim] * 5
    bv = 0
    mm = 0
    ms = 0

    while (1):
        vector = np.random.rand(100)
        generated = generate_image_from_vector(vector) * 255

        cv2.imwrite("search/generatedtmp.jpg", generated)
        generated_img = cv2.imread("search/generatedtmp.jpg")
        generated_gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)

        values = compare_images(original, generated_gray, "Original vs Generated")

        if values[0] < min_mse[mm][0]:
            name_mm = "search/img/min_mse{}.jpg".format(mm)
            data_mm = "search/data/min_mse{}.txt".format(mm)
            min_mse[mm][0] = values[0]
            min_mse[mm][1] = name_mm
            min_mse[mm][2] = data_mm

            cv2.imwrite(name_mm, generated)
            with open(data_mm, 'w+') as f:
                f.write("vector: {}\n".format(vector))
                f.write("MSE: {0:.2f}\n".format(values[0]))
                f.write("SSIM: {0:.2f}".format(values[1]))

            print("Found better MSE: %.2f" % (min_mse[mm][0]))
            mm = (mm + 1) % 5

        if values[1] > max_ssim[ms][0]:
            name_ms = "search/img/max_ssim{}.jpg".format(ms)
            data_ms = "search/data/max_ssim{}.txt".format(ms)
            max_ssim[ms][0] = values[1]
            max_ssim[ms][1] = name_ms
            max_ssim[ms][2] = data_ms

            cv2.imwrite(name_ms, generated)
            with open(data_ms, 'w+') as f:
                f.write("vector: {}\n".format(vector))
                f.write("MSE: {0:.2f}\n".format(values[0]))
                f.write("SSIM: {0:.2f}".format(values[1]))

            print("Found better SSIM: %.2f" % (max_ssim[ms][0]))
            ms = (ms + 1) % 5

        if values[0] < best_values[bv][0] and values[1] > best_values[bv][1]:
            name_bv = "search/img/bestvalues{}.jpg".format(bv)
            data_bv = "search/data/bestvalues{}.txt".format(bv)
            best_values[bv][0] = values[0]
            best_values[bv][1] = values[1]
            best_values[bv][2] = name_bv
            best_values[bv][3] = data_bv

            cv2.imwrite(name_bv, generated)
            with open(data_bv, 'w+') as f:
                f.write("vector: {}\n".format(vector))
                f.write("MSE: {0:.2f}\n".format(values[0]))
                f.write("SSIM: {0:.2f}".format(values[1]))

            print("Found better MSE: %.2f, SSIM: %.2f" % (best_values[bv][0], best_values[bv][1]))
            bv = (bv + 1) % 5


def show_best():
    original = cv2.imread("search/dziekan-gajecki.jpg")
    best = cv2.imread("search/img/bestvalues0.jpg")
    min_mse = cv2.imread("search/img/min_mse0.jpg")
    max_ssim = cv2.imread("search/img/max_ssim0.jpg")
    fig = plt.figure("Images")

    images = ("Original", original), ("Max SSIM", max_ssim), ("Both values", best), ("Min MSE", min_mse)

    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.set_title(name)
        plt.imshow(image)
        plt.axis("off")

    plt.show()
