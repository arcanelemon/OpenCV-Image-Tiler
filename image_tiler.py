import math
import cv2
import numpy as np
from PIL import Image

def split(image, slice_size = 416):

    # scale image to match slice dimension
    height = image.shape[0]
    width = image.shape[1]
    y_scale = math.ceil(height / slice_size)
    x_scale = math.ceil(width / slice_size)
    dim = (x_scale * slice_size, y_scale * slice_size)
    
    # apply scale
    image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    height = image.shape[0]
    width = image.shape[1]
    
    # split image
    images = []
    for i in range (height // slice_size):
        for j in range((width // slice_size)):
            left = j * slice_size;
            top = i * slice_size;
            right = (j + 1) * slice_size;
            bottom = (i + 1) * slice_size;

            new_image = image[top:bottom, left:right]
            images.append(new_image)

    return images

def tile(images, width, height):
    
    # scale to tile size
    slice_size = images[0].shape[0]
    y_scale = math.ceil(height / slice_size)
    x_scale = math.ceil(width / slice_size)
    dim = (x_scale * slice_size, y_scale * slice_size)
    
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    final_image = cv2.resize(final_image, dim, interpolation=cv2.INTER_CUBIC)

    # tile image
    x, y = 0, 0
    for image in images:
        if x + image.shape[1] > final_image.shape[1]:
           x = 0
           y += image.shape[0]
        
        ## image = np.hstack((image, np.zeros((image.shape[0], image.shape[1], 3))))
        final_image[y:image.shape[0] + y, x:image.shape[1] + x, :] = image
        x += image.shape[1]

    # scale final image to target dimensions
    y_scale = height / final_image.shape[0]
    x_scale = width / final_image.shape[1]
    dim = (int(x_scale * final_image.shape[1]), int(y_scale * final_image.shape[0]))
    final_image = cv2.resize(final_image, dim, interpolation=cv2.INTER_CUBIC)

    return final_image
