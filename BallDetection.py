
import cv2
import numpy as np
import matplotlib.pyplot as plt
# TechVidvan Object detection of similar color

all_coordinates = []

# Loop through each image (stopping at 45 as ball goes off screen)
array = list(range(1,46))
for i in array:
    img = cv2.imread('Slide'+str(i)+'.JPG')

    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Print HSV
    # cv2.imshow('hsv', hsv)
    # cv2.waitKey(0)

    # lower bound and upper bound for Green color
    lower_bound = np.array([40, 30, 20])
    upper_bound = np.array([150, 255, 255])

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
    output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    # Draw contour on original image
    # print(str(contours))
    output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # print('Frame: ' + str(i))
    # print(contours[0][0])
    all_coordinates.append(contours[0][0])

    # cv2.circle(img, (all_coordina
    # tes[i-1][0][0], all_coordinates[i-1][0][1]), 5, (255, 0, 0), -1)

    # Showing the output
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

all_y = []
all_y_cm = []
for index in all_coordinates:
    all_y.append(index[0][1])
for i in all_y:
    all_y_cm.append(i/6)
print('All y-values: ' + str(all_y))

# Create time list and append based on regular interval based on list
# Might not be correct because we skipped some frames?
time = []
for i in array:
    time.append(round(1/30*i,3))

# Print out time, final lists, and y values
print('Time Values: ' + str(time))
final = [str(time[i])+','+str(all_y[i]) for i in range(len(time))]
print('Final List: ' + str(final))


# Position vs time graph

# x axis and y-axis values
x = time
y_pos = all_y_cm

# plotting the points
plt.plot(x, all_y_cm)

# naming the axes
plt.xlabel('x - axis - time (s)')
plt.ylabel('y - axis - pixel position (cm)')

# giving a title to my graph
plt.title('Position vs Time')

# function to show the plot
plt.show()


# Velocity vs time graph

velocities = []
for x in range(len(all_y_cm)-1):
    dif = all_y_cm[x+1] - all_y_cm[x]
    velocities.append(dif)


# x-axis values and corresponding y-axis values
time_len = len(time)
time_vel = time[1:]
x = time_vel

y_vel = velocities

# plotting the points
plt.plot(x, y_vel)

# naming the axes
plt.xlabel('x - axis (time - s)')
plt.ylabel('y - axis (velocity - cm)')

# giving a title to my graph
plt.title('Velocity vs Time')

# function to show the plot
plt.show()


# Acceleration vs time graph

acceleration = []
for x in range(len(velocities)-1):
    dif2 = velocities[x+1] - velocities[x]
    acceleration.append(dif2)

# x-axis values and corresponding y-axis values
time_len = len(time)
time = time[2:]
x_accel = time
y_accel = acceleration

# plotting the points
plt.plot(x_accel, y_accel)

# giving a title to my graph
plt.title('Acceleration vs Time')
# naming the axes
plt.xlabel('x - axis (time)')
plt.ylabel('y - axis (acceleration')

# function to show the plot
plt.show()
