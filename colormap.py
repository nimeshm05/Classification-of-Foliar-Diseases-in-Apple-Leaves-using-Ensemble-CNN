import cv2
import os
import glob
import numpy as np
from google.colab.patches import cv2_imshow

folder_path = "<Enter path>"
destination_path = '<Enter path>'

# Apply Jet Colormap and gaussian blurring
'''
 We also should specify the standard deviation in the X and Y directions, sigmaX and sigmaY respectively. 
 If only sigmaX is specified, sigmaY is taken as the same as sigmaX. If both are given as zeros, they are calculated 
 from the kernel size. Gaussian blurring is highly effective in removing Gaussian noise from an image.

 sigmax = 5
 sigmay = 5
 bordertype = 0
'''
for image_file in glob.glob(folder_path + "/*.jpg"):
  image = cv2.imread(image_file)
  # add inferno, bone, hsv, and other colormaps when testing.
  jet_colormap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
  cv.GaussianBlur(jet_colormap,(5,5),0)
  cv2.imwrite(os.path.join(destination_path , f'{image_file}'), jet_colormap)
