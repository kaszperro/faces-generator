import glob
import os
import re

import imageio


def make_gif(gif_name='gen.gif', images_path='./trained/faces/generated'):
    # make GIF file of results after nth iteration of generating generators
    images_list = glob.glob(images_path + '/*.png')
    images_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    gif = []

    for image in images_list:
        if os.path.exists(image):
            gif.append(imageio.imread(image))

    gifs_path = images_path + '/gifs/'
    if not os.path.exists(gifs_path):
        os.mkdir(gifs_path)

    imageio.mimsave(gifs_path + gif_name, gif)

def make_choice(number):
    if number == 1:
        make_gif(images_path='./trained/faces/generated')
    elif number == 2:
        make_gif(images_path='./trained/flowers/generated')
    elif number == 3:
        quit()
    else:
        print("\nInvalid number. Try again:\n")
        main()

    print("Done!")
    quit()

def print_menu():
    print(30 * '-')
    print("CHOOSE ONE")
    print(30 * '-')
    print("1. Faces")
    print("2. Flowers")
    print("3. Quit")
    print(30 * '-')

def main():
    print_menu()

    choice = input('Enter your choice [1-3]: ')
    make_choice(int(choice))

if __name__ == "__main__":
    main()