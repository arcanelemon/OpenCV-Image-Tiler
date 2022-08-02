import numpy as np
from PIL import Image

def split(image, slice_size = 416):

    image_arr = np.array(image, dtype=np.uint8)
    height = image_arr.shape[0]
    width = image_arr.shape[1]

    images = []
    for i in range (height // slice_size):
        for j in range((width // slice_size)):
            left = i * slice_size;
            top = j * slice_size;
            right = (i + 1) * slice_size;
            bottom = (j + 1) * slice_size;
            images.append(Image.fromarray(image_arr[top:left, bottom:right]))

    return images

def tile(images, width, height):

    i, x, y = 0, 0, 0
    image_arr = np.array([])
    while y < height:

        horizontal = np.array([])

        while x < width:
            image = np.array(images[i], dtype=np.uint8)
            horizontal = np.concatenate(horizontal, image, 1)
            x += image.shape[1]
            i++

        image_arr = np.concatenate(image_arr, horizontal, 0)
        x = 0
        y += horizontal.shape[0]

    return Image.fromarray(image_arr)
