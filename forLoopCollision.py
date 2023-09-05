
import cv2 as cv
import numpy as np

# TechVidvan Object detection of similar color

import cv2
import numpy as np

all_coordinates = []
# Reading the image
for i in range(1, 17):
    img = cv2.imread('Slide'+str(i)+'.JPG')

    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color
    lower_bound = np.array([50, 20, 20])
    upper_bound = np.array([200, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Remove unnecessary noise from mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #print mask
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)

    # Segment only the detected region
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Find contours from the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contour on segmented image
    # output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    # Draw contour on original image
    #print(str(contours))
    output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    print(contours[0])
    all_coordinates.append(contours[0][0])

    #cv2.circle(img,(798,424), 5, (255,0,0), -1)
    # Showing the output

    # cv2.imshow("Image", img)
    #cv2.imshow("Output", output)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

for i in range(19, 50):
    img = cv2.imread('Slide'+str(i)+'.JPG')

    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower bound and upper bound for Green color
    lower_bound = np.array([50, 20, 20])
    upper_bound = np.array([200, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Remove unnecessary noise from mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #print mask
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)

    # Segment only the detected region
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Find contours from the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contour on segmented image
    # output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    # Draw contour on original image
    #print(str(contours))
    output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    print(contours[0])
    all_coordinates.append(contours[0][0])

    #cv2.circle(img,(798,424), 5, (255,0,0), -1)
    # Showing the output

    # cv2.imshow("Image", img)
    #cv2.imshow("Output", output)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

print("Test 2")
#print(all_coordinates)

