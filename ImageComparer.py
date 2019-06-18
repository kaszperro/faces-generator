import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import _structural_similarity as ssim

from Evaluation import generate_image_from_vector


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
    # WARNING: the two images must have the same dimension
    err = np.sum((imageA.astype("float")[:, :, None] - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar" the two images are
    return err


def compare_images(imageA, imageB):
    # compare image using Mean Squared Error & Structural Similarity Measure
    m = mse(imageA, imageB)
    s = ssim.compare_ssim(imageA, imageB)

    # return both results as list
    return [m, s]


def comparer(to_compare="search/dziekan-gajecki.jpg", generator="./trained/faces/generator.pth"):
    # infinite function searching image as similar to "to_compare" possible
    # WARNING: overwrites last search
    original = cv2.imread(to_compare)
    # Using greyscale for easier searching
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # structure for storing results
    best_values = [sys.maxsize, 0, "img", "data"]
    best_values = [best_values] * 5
    min_mse = [sys.maxsize, "img", "data"]
    min_mse = [min_mse] * 5
    max_ssim = [0, "img", "data"]
    max_ssim = [max_ssim] * 5

    # non-pythonical indexes for iteration of lists - optimalization for editing results
    bv = 0
    mm = 0
    ms = 0

    while (1):
        # generate random image for comparison
        vector = np.random.rand(100)
        # mapping
        generated = generate_image_from_vector(vector, generator) * 255

        cv2.imwrite("search/generatedtmp.jpg", generated)
        generated_img = cv2.imread("search/generatedtmp.jpg")
        generated_gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)

        values = compare_images(original, generated_gray)

        # save five best images with minimal MSE
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

        # save five best images with maximal SSIM
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

        # save five best images with both parameters as good as possible
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


def show_best(original_img="search/dziekan-gajecki.jpg", index=0):
    # show comparison of original image with found one (with index 0)
    original = cv2.imread(original_img)
    best = cv2.imread("search/img/bestvalues{}.jpg".format(index))
    min_mse = cv2.imread("search/img/min_mse{}.jpg".format(index))
    max_ssim = cv2.imread("search/img/max_ssim{}.jpg".format(index))
    fig = plt.figure("Images")

    images = ("Original", original), ("Max SSIM", max_ssim), ("Both values", best), ("Min MSE", min_mse)

    for (i, (name, image)) in enumerate(images):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.set_title(name)
        plt.imshow(image)
        plt.axis("off")

    plt.show()

def menu_search():
    print_generator_menu()
    choice = input("Choose generator [1-2]: ")
    generator = make_generator_choise(int(choice))

    image_path = input("Give path to image for comparison [leave empty for default]: ")
    if not image_path:
        image_path = "search/dziekan-gajecki.jpg"

    print("Starting searching!")
    comparer(image_path, generator)

def menu_show():
    image_path = input("Give path to image for comparison [leave empty for default]: ")

    if not image_path:
        image_path = "search/dziekan-gajecki.jpg"

    index = input("Give index of saved images (default is 0) [0-4]: ")

    if not index:
        index = 0

    if int(index) > 4 or int(index) < 0:
        print("Wrong input, using index 0")
        show_best(image_path, 0)
    else:
        show_best(image_path, index)

def make_generator_choise(number):
    if number == 1:
        return "./trained/faces/generator.pth"
    elif number == 2:
        return "./trained/flowers/generator.pth"
    else:
        print("\nInvalid number. Try again:\n")
        menu_search()

def print_generator_menu():
    print(30 * '-')
    print("1. Faces")
    print("2. Flowers")
    print(30 * '-')

def make_main_choice(number):
    if number == 1:
        menu_search()
    elif number == 2:
        menu_show()
    elif number == 3:
        quit()
    else:
        print("\nInvalid number. Try again:\n")
        main()

    print("Done!")
    quit()

def print_main_menu():
    print(30 * '-')
    print("CHOOSE ONE")
    print(30 * '-')
    print("1. Search for similar image")
    print("2. Show latest results")
    print("3. Quit")
    print(30 * '-')

def main():
    print_main_menu()

    choice = input('Enter your choice [1-3]: ')
    make_main_choice(int(choice))

if __name__ == "__main__":
    main()