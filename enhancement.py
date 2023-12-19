import numpy as np
import cv2
from scipy import signal
from scipy import ndimage
import math
import scipy
import sys
import skimage
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt

# from functions import Functions


class FingerprintImageEnhancer(object):
    # Set intial parameters
    def __init__(self):

        # image block size
        self.ridge_segment_blksze = 16

        # set the mask threshold (true as  its a ridge / valley)
        self.ridge_segment_thresh = 0.1

        # first gaussian kernal sigma
        self.gradient_sigma = 1
        # second gaussian kernal sigma used for smoothing
        self.block_sigma = 7
        # sigma for sine cosine smoothing using gaussian kernal
        self.orient_smooth_sigma = 7

        # determine the fequency block size
        self.ridge_freq_blksze = 38
        self.ridge_freq_windsze = 5
        self.min_wave_length = 10
        self.max_wave_length = 15

        self.kx = 0.9
        self.ky = 0.9
        self.angleInc = 3
        self.ridge_filter_thresh = -3

        self._mask = []
        self._normim = []
        self._orientim = []
        self._mean_freq = []
        self._median_freq = []
        self._freq = []
        self._freqim = []

        # self._binim = []
        self._binim = []

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Make normalization (The __normalise method is used to normalize an input image by subtracting its mean and dividing by its standard deviation)
# The purpose of normalization is to transform the pixel values to a standard scale or distribution, making them more suitable for further processing or analysis.
    def __normalise(self, img, mean, std):
        if(np.std(img) == 0):
            raise ValueError("Image standard deviation is 0. Please review image again")
        normed = (img - np.mean(img)) / (np.std(img))
        return (normed)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# The __ridge_segment function is used for segmenting ridges in a fingerprint image
# Ouput:      normim   - Image where the ridge regions are renormalised to
#                        have zero mean, unit standard deviation.
#              mask    - Mask indicating ridge-like regions of the image,
#                        0 for non ridge regions, 1 for ridge regions.
#              maskind - Vector of indices of locations within the mask.
    def __ridge_segment(self, img):

        # image row and columns
        rows, cols = img.shape

        # normalise to get zero mean and unit standard deviation
        im = self.__normalise(img, 0, 1)

        # # Show normalized image
        # plt.imshow(im, cmap='gray')
        # plt.title("Normalized Image")
        # plt.axis('off')
        # plt.show()


        # The function calculates the dimensions (new_rows and new_cols) of a padded image based on the block size (self.ridge_segment_blksze).
        # The padded image is resized to ensure that it can be divided into blocks of the specified size without any remainder.
        new_rows = np.int(self.ridge_segment_blksze * np.ceil((np.float(rows)) / (np.float(self.ridge_segment_blksze))))
        new_cols = np.int(self.ridge_segment_blksze * np.ceil((np.float(cols)) / (np.float(self.ridge_segment_blksze))))
        # A new array padded_img of zeros is created with the dimensions of the padded image.
        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))
        padded_img[0:rows][:, 0:cols] = im

        for i in range(0, new_rows, self.ridge_segment_blksze):
            for j in range(0, new_cols, self.ridge_segment_blksze):
                # extract each block from the image
                block = padded_img[i:i + self.ridge_segment_blksze][:, j:j + self.ridge_segment_blksze]
                # make an array with the same block size updated with the standared deviation values
                stddevim[i:i + self.ridge_segment_blksze][:, j:j + self.ridge_segment_blksze] = np.std(block) * np.ones(block.shape)

        # The calculated stddevim array is assigned to the corresponding block region in the stddevim array, which has the same dimensions as the padded image.
        # The stddevim array is cropped to the original dimensions of the image using array indexing (stddevim = stddevim[0:rows][:, 0:cols]).
        stddevim = stddevim[0:rows][:, 0:cols]
        # A binary mask (self._mask) is created by comparing the values in the stddevim array with a threshold value (self.ridge_segment_thresh)
        self._mask = stddevim > self.ridge_segment_thresh

        # The mean and standard deviation values of the normalized image (im) are computed only for the pixels identified by the mask (self._mask).
        mean_val = np.mean(im[self._mask])
        std_val = np.std(im[self._mask])

        # final segmentation output after second normalization (_normim)
        self._normim = (im - mean_val) / (std_val)

        # Print statements for debugging
        print("Image shape:", img.shape)
        print("Normalized image shape:", im.shape)
        print("Padded image shape:", padded_img.shape)
        print("Stddev image shape:", stddevim.shape)
        print("Mask shape:", self._mask.shape)
        print("Mean value:", mean_val)
        print("Std value:", std_val)

        # # Show normalized image
        # plt.imshow(self._normim, cmap='gray')
        # plt.title("Normalized Image")
        # plt.axis('off')
        # plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------------

# ridge orientation estimation
# Output:    orientim           - The orientation image in radians.
#                                 Orientation values are +ve clockwise
#                                 and give the direction *along* the
#                                 ridges.
    def __ridge_orient(self):

        # extracting the dimensions of the normalized image
        rows,cols = self._normim.shape

        # Calculate image gradients.
        # It is ensured to be an odd integer by adding 1 if it's even.
        sze = np.fix(6*self.gradient_sigma)
        if np.remainder(sze,2) == 0:
            sze = sze+1
        # A Gaussian kernel is generated using cv2.getGaussianKernel with a size of sze and standard deviation self.gradient_sigma.
        gauss = cv2.getGaussianKernel(np.int(sze),self.gradient_sigma)
        f = gauss * gauss.T

        # Gradient of Gaussian
        fy,fx = np.gradient(f)

        Gx = signal.convolve2d(self._normim, fx, mode='same')
        Gy = signal.convolve2d(self._normim, fy, mode='same')

        Gxx = np.power(Gx,2)
        Gyy = np.power(Gy,2)
        Gxy = Gx*Gy

        #Now smooth the covariance data to perform a weighted summation of the data.
        sze = np.fix(6*self.block_sigma)

        #generate another gaussian kernal
        gauss = cv2.getGaussianKernel(np.int(sze), self.block_sigma)
        f = gauss * gauss.T

        Gxx = ndimage.convolve(Gxx,f)
        Gyy = ndimage.convolve(Gyy,f)
        Gxy = 2*ndimage.convolve(Gxy,f)

        # Analytic solution of principal direction
        # np.finfo(float).eps is a small value (machine epsilon) added to avoid division by zero. It ensures numerical stability of the computation.
        denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps
        # Sine and cosine of doubled angles
        sin2theta = Gxy/denom
        cos2theta = (Gxx-Gyy)/denom

        # If the variable self.orient_smooth_sigma is set, the code applies additional smoothing to the computed sin2theta and cos2theta values.
        # This smoothing is done to further refine and enhance the estimated ridge orientations.
        # This convolution applies a weighted average to smooth the angles and reduce any potential noise or inconsistencies.
        if self.orient_smooth_sigma:
            sze = np.fix(6*self.orient_smooth_sigma)
            if np.remainder(sze,2) == 0:
                sze = sze+1
            gauss = cv2.getGaussianKernel(np.int(sze), self.orient_smooth_sigma)
            f = gauss * gauss.T
            # Smoothed sine and cosine of doubled angles
            cos2theta = ndimage.convolve(cos2theta,f)
            sin2theta = ndimage.convolve(sin2theta,f)


        # he ridge orientations (self._orientim) are calculated by taking the arctangent of the ratio between sin2theta and cos2theta.
        # divided by 2(take one angle), and adding π/2 to the result.
        self._orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2

        # Print statements for debugging
        print(self._orientim)




#----------------------------------------------------------------------------------------------------------------------------------------------
# ridge frequency estimation
    def __ridge_freq(self):

        rows, cols = self._normim.shape
        # A frequency matrix (freq) of zeros with the same shape as the normalized image is initialized.
        freq = np.zeros((rows, cols))

        # iterates through blocks of size self.ridge_freq_blksze within the image using nested for loops.
        # This allows processing the image in smaller blocks to estimate the ridge frequency locally.
        for r in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
            for c in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):

              # a block image (blkim) and its corresponding block orientation (blkor) are extracted from the normalized image (self._normim)
              # and the orientation map (self._orientim), respectively.
                blkim = self._normim[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]
                blkor = self._orientim[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]

                # The function self.__frequest is called to estimate the local ridge frequency for the current block
                # The estimated frequency values for the current block are assigned to the corresponding positions in the frequency matrix
                freq[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze] = self.__frequest(blkim, blkor)



        # The resulting frequency matrix (freq) is multiplied element-wise by the binary mask (self._mask).
        # This step masks out the frequency values in areas outside the fingerprint region.
        self._freq = freq * self._mask


        # The frequency matrix is reshaped into a 1D array (freq_1d) using np.reshape.
        freq_1d = np.reshape(self._freq, (1, rows * cols))
        # The indices of the non-zero elements in the frequency array are obtained using np.where(freq_1d > 0).
        ind = np.where(freq_1d > 0)
        ind = np.array(ind)
        # The obtained indices are extracted as a 1D array (ind) using ind = ind[1, :].
        ind = ind[1, :]
        # The non-zero frequency values are extracted from the frequency array.
        non_zero_elems_in_freq = freq_1d[0][ind]

        self._mean_freq = np.mean(non_zero_elems_in_freq)
        self._freq = self._mean_freq * self._mask

#----------------------------------------------------------------------------------------------------------------------------------------------

# The __frequest function is responsible for estimating the local ridge frequency for a given block image (blkim) and its corresponding block orientation (blkor)

    def __frequest(self, blkim, blkor):

        rows, cols = np.shape(blkim)
        # The cosine and sine of twice the block orientation (blkor) are calculated using np.cos(2 * blkor) and np.sin(2 * blkor)
        # The average cosine and sine values are obtained using np.mean to represent the dominant orientation of the block.
        # This avoids wraparound problems at the origin.
        cosorient = np.mean(np.cos(2 * blkor))
        sinorient = np.mean(np.sin(2 * blkor))
        orient = math.atan2(sinorient, cosorient) / 2

        # The block image (blkim) is rotated by the calculated orientation (orient)
        rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,mode='nearest')

        # The rotated block image (rotim) is cropped to a square region with a size proportional to the original block size.
        # The cropping is done to remove any black borders resulting from the rotation.
        # This prevents the projection down the columns from being mucked up.
        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]


        # Sum down the columns to get a projection of the grey values down the ridges.
        proj = np.sum(rotim, axis=0)
        # The projection is dilated using scipy.ndimage.grey_dilation to enhance the peaks representing the ridges
        dilation = scipy.ndimage.grey_dilation(proj, self.ridge_freq_windsze, structure=np.ones(self.ridge_freq_windsze))
        # This difference measures the prominence of the peaks in the projection.
        temp = np.abs(dilation - proj)

        peak_thresh = 2
        # A threshold (peak_thresh) is used to identify significant peaks.
        # The peaks are identified by comparing temp to the threshold and proj to its mean
        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)

        rows_maxind, cols_maxind = np.shape(maxind)

        # If the number of identified peaks is less than 2 (cols_maxind < 2), indicating a lack of significant peaks, a zero matrix is returned.
        if (cols_maxind < 2):
            return(np.zeros(blkim.shape))
        else:
            NoOfPeaks = cols_maxind
            waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
            if waveLength >= self.min_wave_length and waveLength <= self.max_wave_length:
                return(1 / np.double(waveLength) * np.ones(blkim.shape))
            else:
                return(np.zeros(blkim.shape))



#----------------------------------------------------------------------------------------------------------------------------------------------
# ridge filter using Gabor filters to enhance the visibility of ridges or features in an image
    def __ridge_filter(self):

        im = np.double(self._normim)
        rows, cols = im.shape
        # An empty image (newim) of the same size as the input image is created to store the filtered result.
        newim = np.zeros((rows, cols))

        # reshape the frequency spectrum
        freq_1d = np.reshape(self._freq, (1, rows * cols))
        # indices where the frequency is greater than 0 are extracted (ind).
        ind = np.where(freq_1d > 0)
        ind = np.array(ind)
        ind = ind[1, :]
        non_zero_elems_in_freq = freq_1d[0][ind]
        # The non-zero elements in the frequency spectrum are rounded and scaled to two decimal places.
        non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
        # The unique values are obtained and stored in unfreq.
        unfreq = np.unique(non_zero_elems_in_freq)

        # The standard deviations (sigmax and sigmay) for the Gabor filter are calculated based on the smallest frequency value in unfreq and scaling factors (self.kx and self.ky).
        sigmax = 1 / unfreq[0] * self.kx
        sigmay = 1 / unfreq[0] * self.ky

        # The size of the Gabor filter (sze) is determined based on the maximum standard deviation among sigmax and sigmay.
        sze = np.int(np.round(3 * np.max([sigmax, sigmay])))

        # A meshgrid x and y is created to represent the coordinates of the Gabor filter.
        x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))
        # The original Gabor filter is constructed by applying a Gaussian envelope and a sinusoidal carrier wave.
        reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
            2 * np.pi * unfreq[0] * x)        # this is the original gabor filter

        filt_rows, filt_cols = reffilter.shape

        # The Gabor filter is rotated at different angles (self.angleInc) to generate a bank of Gabor filters (gabor_filter) spanning 180 degrees.
        angleRange = np.int(180 / self.angleInc)
        gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))
        for o in range(0, angleRange):
            rot_filt = scipy.ndimage.rotate(reffilter, -(o * self.angleInc + 90), reshape=False)
            gabor_filter[o] = rot_filt


        maxsze = int(sze)

        temp = self._freq > 0
        validr, validc = np.where(temp)

        temp1 = validr > maxsze
        temp2 = validr < rows - maxsze
        temp3 = validc > maxsze
        temp4 = validc < cols - maxsze

        final_temp = temp1 & temp2 & temp3 & temp4

        finalind = np.where(final_temp)

        maxorientindex = np.round(180 / self.angleInc)
        orientindex = np.round(self._orientim / np.pi * 180 / self.angleInc)

        # do the filtering
        for i in range(0, rows):
            for j in range(0, cols):
                if (orientindex[i][j] < 1):
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if (orientindex[i][j] > maxorientindex):
                    orientindex[i][j] = orientindex[i][j] - maxorientindex
        finalind_rows, finalind_cols = np.shape(finalind)
        sze = int(sze)
        for k in range(0, finalind_cols):
            r = validr[finalind[0][k]]
            c = validc[finalind[0][k]]

            img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

            newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

        self._binim = newim < self.ridge_filter_thresh


#----------------------------------------------------------------------------------------------------------------------------------------------
# saves the enhanced image at the specified path
    def save_image(self, img, path):
         cv2.imwrite(path, (255 * self._binim))



#----------------------------------------------------------------------------------------------------------------------------------------------
# main function to enhance the image.
    def enhance(self, img, resize=True):
        if(resize):
            rows, cols = np.shape(img)
            aspect_ratio = np.double(rows) / np.double(cols)
            # randomly selected number
            new_rows = 350
            new_cols = new_rows / aspect_ratio

            img = cv2.resize(img, (np.int(new_cols), np.int(new_rows)))

        self.__ridge_segment(img)   # normalise the image and find a ROI
        self.__ridge_orient()       # compute orientation image
        self.__ridge_freq()         # compute major frequency of ridges
        self.__ridge_filter()       # filter the image using oriented gabor filter

        return(self._binim)