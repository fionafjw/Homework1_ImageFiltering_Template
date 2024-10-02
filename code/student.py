# Homework 1 Image Filtering Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the homework webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    height_k, width_k = kernel.shape
    if height_k%2 == 0 or width_k%2 == 0:
        raise Exception("Kernel has even dimensions must be odd")
    #print("kernel: ", kernel.shape)
    pad_h = height_k//2
    pad_w = width_k//2
    #print("pad_h, pad_w:", pad_h, pad_w)
    #rotate the kernel
    kernel = np.rot90(kernel, 2)
    
    '''
    if len(image.shape) == 3:
        height_i, width_i, channel = image.shape
        #print("input image: ", image.shape)
        #pad input image np.pad() with zeros based on kernel size
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
        #convolution function, for RGB images do convolution on every channel
        #place center of kernel at a (x, y) in input image
        for c in range(channel):
            for i in range(height_i):
                for j in range(width_i):
                    x = i+pad_h
                    y = j+pad_w
                #element wise multiplcation np.multiply() and then sum np.sum()
                    neighborhood = image[x-pad_h:x+pad_h+1, y-pad_w: y+pad_w+1, c]
                #place sum into (x, y) in output image
                    filtered_image[i, j, c] = np.sum(np.multiply(kernel, neighborhood))

    else:
        height_i, width_i = image.shape
        #print("input image: ", image.shape)
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
        for i in range(height_i):
            for j in range(width_i):
                x = i+pad_h
                y = j+pad_w
                neighborhood = image[x-pad_h:x+pad_h+1, y-pad_w: y+pad_w+1]
                filtered_image[i, j] = np.sum(np.multiply(kernel, neighborhood))
    '''
    #trying my_imfilter with shifts to optimize speed
    #multiply the image by each kernel value and store it in a temporary image add all together
    #loop by kernel size
    #'''
    if filtered_image.ndim == 3:
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
    else:
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), 'constant')
    filtered_image = np.zeros(image.shape)
    for i in range(height_k):
        for j in range(width_k):
            temp = np.multiply(kernel[i, j], image)
            #shift up or down
            temp = np.roll(temp, pad_h-i, axis = 0)
            #shift left or right
            temp = np.roll(temp, pad_w-j, axis = 1)
            #add to final image
            filtered_image += temp

    #print("after convolution: ", filtered_image.shape)
    height_i, width_i = image.shape[:2]
    if filtered_image.ndim == 3:
        filtered_image = filtered_image[pad_h:height_i-pad_h, pad_w:width_i-pad_w, :]
    else:
        filtered_image = filtered_image[pad_h:height_i-pad_h, pad_w:width_i-pad_w]
        
    #print("after reduction: ", filtered_image.shape)
    #'''
    ##################
    return filtered_image

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the homework webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here
    low_frequencies = my_imfilter(image1, kernel) # Replace with your implementation

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = image2 - my_imfilter(image2, kernel) # Replace with your implementation

    # (3) Combine the high frequencies and low frequencies, and make sure the hybrid image values are within the range 0.0 to 1.0
    # Your code here
    hybrid_image = np.clip(low_frequencies + high_frequencies, 0.0, 1.0) # Replace with your implementation

    '''
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(low_frequencies, cmap = 'gray')
    axarr[0,1].imshow(high_frequencies, cmap = 'gray')
    axarr[1,0].imshow(hybrid_image, cmap = 'gray')

    plt.show()
    '''

    return low_frequencies, high_frequencies, hybrid_image
