import cv2 
import numpy as np
import math 
from pdf2image import convert_from_path
import os
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

Image.MAX_IMAGE_PIXELS = None


def canny2hough(img):
    
    """
    Since walls are straight lines we could apply Canny Edge filter to find shapes and 
    followed it by HoughLines which is algorithm to find straight lines. This should give us 
    simput for a algorithm that would choose 1 straight line per local region and therefore 
    identify walls. 
    """
  
    t_lower = 100 # Lower Threshold 
    t_upper = 200 # Upper threshold 
    aperture_size = 5 # Aperture size 
    L2Gradient = True # Boolean 
    
    # Applying the Canny Edge filter with L2Gradient = True 
    dst = cv2.Canny(img, t_lower, t_upper, L2gradient = L2Gradient) 
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    # Apply  HoughLines  algorithm:
    lines = cv2.HoughLines(dst,  #dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
                           1,    # lines: A vector that will store the parameters (r,θ) of the detected lines
                           np.pi / 180, # rho : The resolution of the parameter r in pixels. We use 1 pixel.
                           150, # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
                           None,  # threshold: The minimum number of intersections to "*detect*" a line
                           0, 0 # srn and stn: Default parameters to zero.
                           )
    
    # Draw the lines 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    # Apply Probabilitic HoughLines
    linesP = cv2.HoughLinesP(dst, # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
                             1, # lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
                             np.pi / 180, # rho : The resolution of the parameter r in pixels. We use 1 pixel.
                             50, # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
                             None, # threshold: The minimum number of intersections to "*detect*" a line
                             50, # minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
                             10 # maxLineGap: The maximum gap between two points to be considered in the same line.
                             )

    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    # Save image for RADME
    cv2.imwrite('cv-canny2hough.jpg', cdstP)

    
def watershed_seg(img):

    """source: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    The watershed is a classical algorithm used for segmentation, that is, for separating 
    different objects in an image. Starting from user-defined markers, the watershed algorithm 
    treats pixels values as a local topography (elevation). The algorithm floods basins from 
    the markers until basins attributed to different markers meet on watershed lines. 
    In many cases, markers are chosen as local minima of the image, from which basins are flooded.
    """

    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    # Convert the mean shift image to grayscale, then apply Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

    # Perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    
    # Loop over the unique labels returned by the Watershed algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue
        # Otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # Detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # Draw a bbox enclosing the object
        x, y, w, h = cv2.boundingRect(c) 
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save image for RADME
    cv2.imwrite('cv-watershed.jpg', img)


if __name__ == '__main__':
    
    FILE = 'A-192.pdf' # A-192.pdf, A-492.pdf
    TEST_PATH = os.path.join('data', 'test')
    page = convert_from_path(os.path.join(TEST_PATH, FILE), 500)[0]
    page_resized = page.resize((int(page.size[0]/ 10), int(page.size[1]/ 10)), Image.Resampling.LANCZOS)
    page_cropped = page_resized.crop((200, 0, 1800, 1800))
    img = cv2.cvtColor(np.array(page_cropped), cv2.COLOR_RGB2BGR)
    
    canny2hough(img)
    watershed_seg(img)