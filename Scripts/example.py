
# import image tiler
import image_tiler

# Perform low resolution crop
imgs = crop(img, 1024)

# Pass into algorithm ---->
# Collect bounding boxes <----

# Perform high resolution crop on images with active boxes
for i in active:
    imgs = crop(i, 416)

    # Pass into algorithm ---->
    # Collect bounding boxes <----

# Draw collected bounding boxes on original image
draw_boxes(img, boxes)

# Display final image
