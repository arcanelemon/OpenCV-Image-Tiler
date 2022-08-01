# OpenCV Image Tiler
### OpenCV Image Tiling Utilites
 A simple OpenCV ultility used to split and tile images. For utilization in AIML image tiling and classification models.  
 
 
 
 ## Contents
 #### [Setup](#setup)
 ####  [Python Setup](#python-setup)
 ####  [C Setup](#c-setup)
 #### [Usage](#usage)
 ####  [Python Usage](#python-usage)
 ####  [C Usage](#c-usage)
 
 
 ## Setup
 ### Python Setup
 To setup we simply need drop and
 ```python

 ```

 ### C Setup
 Unfortunately, using our C wrapper is a bit more complex than the strictly Python implementation. 

 After 
 ```c

 ```

 Compile
 ```c

 ```

 Alternatively, you can use the following to compile
 ```c

 ```

 ## Usage 
 ### Python Usage
 Usage in Python is straight-forward. For the purpose of demonstration, I'll be using by @jkjung-avt to demonstrate a basic tiling algorithm approach with Yolov3 Tiny.

 First, grab the frame we are about to process
 ```python

 ```

 Using the image tiler, split the image into desired size
 ```python

 ```

 For each new image, run image classification and draw detections
 ```python

 ```

 After we have classification, recompile image for display
 ```python

 ```
 
 
 ### Methods
  ### split
  > Splits standard numpy image data into an image array portional to slice size. Will resize provided image if slice size does not fit evenly.
  > 
  > Return numpy::ndarray[]
  > 
  > Arguments
  > - **img** (ndarray) The image provided to split.
  > - **slice_size** (int) The desired slice size of the new images.

 ```python
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
 ```
 
 
  ### tile
  > Tiles a numpy image array into a single image of desired dimensions. If output image does not match the desired dimensions, the image is resized to match.
  > 
  > Return numpy::ndarray
  > 
  > Arguments
  > - **imgs** (ndarray[]) The provided images to tile.
  > - **width** (int) The width of the tiled image.
  > - **height** (int) The height of the tiled image.

 ```python
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
 ```
 
 
 ### C Usage
 For usage in C, again we will utilize the [darknet](https://github.com/AlexeyAB/darknet) fork by @AlexeyAB as an example.

 ### Methods
  ### init_image_tiler
  > Initializes the C wrapper to link our python script. Must be called prior to implemenation.
  
  ```c
  
  ```

  ### split_image
  > Splits standard numpy image data into an image array portional to slice size. Will resize provided image if slice size does not fit evenly.
  > 
  > Return numpy::ndarray[]
  > 
  > Arguments
  > - **img** (cv::Mat) The image provided to split.
  > - **slice_size** (int) The desired slice size of the new images.

 ```c
 def split_image(image, slice_size = 416):

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
 ```


  ### tile_image
  > Tiles a numpy image array into a single image of desired dimensions. If output image does not match the desired dimensions, the image is resized to match.
  > 
  > Return numpy::ndarray
  > 
  > Arguments
  > - **imgs** (cv::Mat[]) The provided images to tile.
  > - **width** (int) The width of the tiled image.
  > - **height** (int) The height of the tiled image.

 ```c
 def tile_image(images, width, height):

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
 ```


 
 
