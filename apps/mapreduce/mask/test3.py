'''
https://stackoverflow.com/questions/44295099/how-to-identify-incomplete-rectangles-in-opencv
Provare ultimo consiglio
'''
import cv2
import numpy as np
from mask.shapeDetector import ShapeDetector
import imutils

img = cv2.imread('images/Merged_Image.png')
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 10)
dilate = cv2.dilate(erosion,kernel,iterations = 10)
cv2.bitwise_not ( dilate, dilate )
image = dilate
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
#thresh = dilate
# find contours in the thresholded image and initialize the
# shape detector
'''
'''
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

#cnts = cnts[0] if imutils.is_cv2() else cnts[1]

sd = ShapeDetector()

# loop over the contours
for c in cnts[0]:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour

    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c.astype("int"))

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)

    # show the output image
    # cv2.imshow("Image", image)
    cv2.imwrite("images/test3.png",image)
    # cv2.waitKey(0)