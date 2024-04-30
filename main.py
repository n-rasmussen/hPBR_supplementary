# import packages
import numpy as np
import cv2 as cv
import os
import matplotlib as plt

"""
This code is part of the supplementary work for "Hydrogel-based Photobioreactor for Solid-State Cultivation of Chlorella Vulgaris" 

This code looks at analyzing the surface coverage of algae in the hydrogel photobioreactor (hPBR). 


--------

Some material in this code was taken (or modified) from the following source: 
"How calculate the area of irregular object in an image (opencv)?" - Antonino La Rosa
https://stackoverflow.com/questions/64394768/how-calculate-the-area-of-irregular-object-in-an-image-opencv

"""



# Open Image and read it into uint8 data type (BGR data) & resize image to fit on screen
image = 'pPVAa4.jpg'  # image file name
img = cv.imread(image)
folder = image[:-4] # set folder name (here it is file name - extension (.jpeg)
folder_path = os.path.join(os.getcwd(), folder)
if not os.path.exists(folder_path):
    # If the folder does not exist, create it
    os.makedirs(folder_path)

print(img.shape)
img = cv.resize(img, (round(.25*img.shape[1]), round(.25*img.shape[0])))
assert img is not None,  "Did not read Correctly"

num = 2  # contour index in list to analyze (change value to counter of interest)

# convert image to gray scale and display it
gray_scaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("new", new)
# cv.waitKey(0),

# using Canny edge detection.
edges = cv.Canny(gray_scaled_img, 190, 200) # threshold for edge linking
cv.imshow("Edge detection", edges)
cv.imwrite("{}/Edge_detection.png".format(folder), edges)
cv.imshow("img", img)
cv.imwrite("{}/img.png".format(folder), img)


# dilate the edges so they are more defined.
kernel = np.ones((2, 2))
imgDil = cv.dilate(edges, kernel, iterations=3) #
imgThre = cv.erode(imgDil, kernel, iterations = 3) #eroding edges may help make contours more defined
# cv.imshow("Dil", imgDil)
# cv.imshow("Erod", imgThre)

# Project the contour lines from the Canny method onto the original image
contours, 	hierarchy = cv.findContours(imgThre, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 3)
# cv.imshow("w/ contours", img)

# make a list of the large contours so that the gel contour can be manually found.
a = []
cont = []
for con in contours:
    i = 0
    area = cv.contourArea(con)
    a.append(area)
    i += 1
    if area > 5000: # only look at contours larger than 5000 pixel^2
        perimeter = cv.arcLength(con, True)

        # smaller epsilon -> more vertices detected [= more precision]
        epsilon = 0.0002 * perimeter
        # check how many vertices
        approx = cv.approxPolyDP(con, epsilon, True)
        # print(len(approx))

        cont.append([len(approx), area, approx, con])

print(max(a))
print("---\nfinal number of contours: ", len(cont))


# Removing Background
# Get Dimensions
hh, ww = img.shape[:2]

# draw white contour on black background to make a mask
mask = np.zeros((hh,ww), dtype=np.uint8)

#contour number that represents hydrogel.
# num = 1 # contour number
cv.drawContours(mask,[cont[num][2]], -1, (255,255,255), cv.FILLED)

# apply mask to image
image_masked = cv.bitwise_and(img, img, mask=mask)

# convert to HSV colouring
hsv = cv.cvtColor(image_masked, cv.COLOR_BGR2HSV)
# set lower and upper color limits
lowerVal2 = np.array([28,100,0])
upperVal2 = np.array([80,255,255])
# (120,100,0) (80,255,255) pPVAa0
light_green_mask = cv.inRange(hsv, lowerVal2, upperVal2)
# mask for dark greens
lowerVal3 = np.array([40,10,0])
upperVal3 = np.array([75,255,200])
# Threshold the HSV image to get only red colors
dark_green_mask = cv.inRange(hsv, lowerVal3, upperVal3)
not_dark_green = cv.bitwise_not(dark_green_mask)
light_green_mask = cv.bitwise_and(light_green_mask, not_dark_green)
final_mask = cv.bitwise_or(light_green_mask, dark_green_mask)
# apply mask to original image
final = cv.bitwise_and(hsv, hsv, mask=final_mask) # can change to other mask to visualize like dark_green_mask
# gray final image after applying mask
gray = cv.cvtColor(final, cv.COLOR_BGR2GRAY)
algae_area = cv.countNonZero(gray)
dark_green = cv.countNonZero(dark_green_mask)
print('percent dark green')
try:
    print(dark_green/algae_area*100)
except:
    print("area is 0")

print("algae_area")
print(algae_area)
algae_coverage = algae_area / cont[num][1] * 100
print("Aglae Coverage %")
print(algae_coverage)


dst = cv.add(image_masked, final)
cv.imshow("final", final)
cv.imwrite("{}/final.png".format(folder), final)
cv.imshow("hsv", hsv)
cv.imwrite("{}/hsv.png".format(folder), hsv)
cv.imshow("over", dst)
cv.imwrite("{}/over.png".format(folder), dst)
cv.waitKey(0)