import glob
import os
import re

import imageio


def make_gif(gif_name='gen.gif', images_path='./trained/faces/generated'):
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
