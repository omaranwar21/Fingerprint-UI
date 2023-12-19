from functions import Functions
from enhancement import FingerprintImageEnhancer
import FeatureExtraction
import cv2
from PIL import Image, ImageDraw

import numpy as np
from skimage import io
import glob
import random
import imageio
import PIL, cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import matplotlib.image as mpimg
import skimage
import math
from scipy.ndimage.filters import convolve
from PIL import Image,ImageFilter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# import numpy as np
# from scipy import signal
# from scipy import ndimage
# import math
# import scipy
# import sys
# import skimage
# import skimage.morphology
# from skimage.morphology import convex_hull_image, erosion
# from skimage.morphology import square
# from scipy import ndimage
# from scipy import signal
# import matplotlib.pyplot as plt


def Run(path):
    # Data Binarization
    input_image_path = path
    output_image_path = "finger_after_binarization.jpeg"

    binary_image = Functions.binarize_image(input_image_path)
    cv2.imwrite(output_image_path, binary_image)

    # Resize image
    image_path = "finger_after_binarization.jpeg"  # Replace with the path to your image
    new_width = 200  # Specify the new width of the image
    new_height = 500  # Specify the new height of the image

    Functions.resize_image(image_path, new_width, new_height)

    # # Sharp Crop
    # img = Image.open(r"resized_image.jpg")
    #
    # left = 10
    # top = 40
    # right = 300
    # bottom = 500
    #
    # img_res = img.crop((left, top, right, bottom))
    #
    # # Save the cropped image
    # img_res.save(r"Sharp_cropped_finger.jpg")

    image_path = "resized_image.jpg"  # Replace with the path to your image
    crop_height = 350  # Specify the height of the crop
    oval_width_ratio = 0.9  # Specify the ratio of the oval width to the image width (0.0 to 1.0)

    cropped_image = Functions.crop_top_oval(image_path, crop_height, oval_width_ratio)
    cropped_image.save("Oval_cropped_finger.jpg")  # Replace with the desired path to save the cropped image

    image_enhancer = FingerprintImageEnhancer()         # Create object called image_enhancer
    img = cv2.imread('Oval_cropped_finger.jpg', 0)

    enhanced_img = image_enhancer.enhance(img)     # run image enhancer
    image_enhancer.save_image(enhanced_img, 'Application_enhance.jpg')   # save enhanced image


# -------------------------
def minext(path):
    print(path)
    enhanced_image = cv2.imread(path, 0)  # Read as grayscale
    print(enhanced_image)
    ret11, img1 = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    skel1 = skimage.morphology.skeletonize(img1)
    skel1 = np.uint8(skel1)*255;
    mask1 = img1*255;

    (minutiaeTerm1, minutiaeBif1) = FeatureExtraction.getTerminationBifurcation(skel1, mask1);
    FeaturesTerm1, FeaturesBif1 = FeatureExtraction.extractMinutiaeFeatures(skel1, minutiaeTerm1, minutiaeBif1)

    BifLabel1 = skimage.measure.label(minutiaeBif1, connectivity=1);
    TermLabel1 = skimage.measure.label(minutiaeTerm1, connectivity=1);
    res1 = FeatureExtraction.ShowResults(skel1, TermLabel1, BifLabel1)
    # res1.save_image(extacted_image, 'extacted_image.jpg')   # save enhanced image
    cv2.imwrite('extracted_image.jpg', res1)


# if __name__ == '__main__':
#     Run("F_L3C.jpg")
#     minext("Application_enhance.jpg")