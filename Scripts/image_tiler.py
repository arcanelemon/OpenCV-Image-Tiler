# import libraries

import cv2
import numpy as np


# helper functions

'''
Returns rect with dimensions clamped within a specified size.
'''
def clamp_extents(w, h, rect):
    
    x1 = rect[2]
    y1 = rect[0]
    x2 = rect[3]
    y2 = rect[1]

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    
    return (y1, y2, x1, x2)


# main functions

'''
Returns crops of dimension size from a given image.
'''
def crop(img, size, padding = 0):

    height = img.shape[0]
    width = img.shape[1]
    y =  -(height // -size)
    x = -(width // -size)
    
    # split image
    imgs = []
    for i in range (y):
        for j in range(x):
            x1 = j * size - padding
            y1 = i * size - padding
            x2 = (j + 1) * size + padding
            y2 = (i + 1) * size + padding

            rect = clamp_extents(width, height, (y1, y2, x1, x2))
            imgs.append(img[rect[0]:rect[1], rect[2]:rect[3]])

    return imgs


'''
Draws openCV rectangles of specified color and thickness at bounding box coordinates.
'''
def draw_boxes(img, boxes, color = (255, 0, 0), thickness = 5):

    for box in boxes:
        x1 = box[2][0]
        y1 = box[2][1]
        x2 = box[2][2]
        y2 = box[2][3]
        cls = box[0]
        conf = box[1]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, cls + ': ' + str(conf), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
