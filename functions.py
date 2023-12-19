import numpy as np
import cv2
from PIL import Image, ImageDraw


# from enhancement import FingerprintImageEnhancer


class Functions:
    # Data Binarization
    @staticmethod
    def binarize_image(image_path):
        # Load the image
        image = cv2.imread(image_path, 0)  # Read as grayscale

        # Apply adaptive thresholding
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

        return binary_image

    # # Example usage
    # input_image_path = "/content/L3_sample(3).jpg"
    # output_image_path = "/content/finger_after_binarization.jpeg"
    #
    # binary_image = binarize_image(input_image_path)
    # cv2.imwrite(output_image_path, binary_image)

    # Data Preparation
    @staticmethod
    def resize_image(image_path, new_width, new_height):
        # Open the image
        img = Image.open(image_path)

        # Get the original width and height
        width, height = img.size

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new dimensions
        if aspect_ratio > 1:
            new_width = new_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = int(new_height * aspect_ratio)
            new_height = new_height

        # Resize the image
        resized_img = img.resize((new_width, new_height))

        # Save the resized image
        resized_img.save("resized_image.jpg")  # Replace with the desired path to save the resized image

    # # Example usage
    # image_path = "/content/finger_after_binarization.jpeg"  # Replace with the path to your image
    # new_width = 200  # Specify the new width of the image
    # new_height = 500  # Specify the new height of the image
    #
    # resize_image(image_path, new_width, new_height)


    # # Sharp Crop
    # img = Image.open(r"/content/resized_image.jpg")
    #
    # left = 10
    # top = 40
    # right = 300
    # bottom = 500
    #
    # img_res = img.crop((left, top, right, bottom))
    #
    # # Save the cropped image
    # img_res.save(r"/content/Sharp_cropped_finger.jpg")

    # Ovl Crop
    @staticmethod
    def crop_top_oval(image_path, crop_height, oval_width_ratio):
        # Open the image
        img = Image.open(image_path)

        # Get the width and height of the image
        width, height = img.size

        # Calculate the coordinates of the oval shape
        oval_width = int(width * oval_width_ratio)
        oval_height = crop_height
        left = int((width - oval_width) / 2)
        top = 0
        right = left + oval_width
        bottom = oval_height + 80

        # Create a mask with the oval shape
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((left, top, right, bottom), fill=255)

        # Apply the mask to the image
        img_arr = np.array(img)
        mask_arr = np.array(mask)
        img_arr[mask_arr == 0] = 0
        cropped_img = Image.fromarray(img_arr)

        # Crop the image to the desired height
        cropped_img = cropped_img.crop((0, 0, width, crop_height))

        return cropped_img

    # # Example usage
    # image_path = "/content/Sharp_cropped_finger.jpg"  # Replace with the path to your image
    # crop_height = 350  # Specify the height of the crop
    # oval_width_ratio = 0.9  # Specify the ratio of the oval width to the image width (0.0 to 1.0)
    #
    # cropped_image = crop_top_oval(image_path, crop_height, oval_width_ratio)
    # cropped_image.save("/content/Oval_cropped_finger.jpg")  # Replace with the desired path to save the cropped image