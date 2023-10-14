
# Co-Author: Dylan Winer
# Co-Author: Rishabh Kanodia
# Supervisor: Sean Psulkowski
# Supervisor: Bryant Rodriguez
# Research Sponsor: Dr. Dickens
# Date 06/21/2022
# Version 1.1
# License = "MIT License"
# Dylan Winer email = "dywiner@shorecrest.org"
# Rishabh Kanodia email = "rishabhkanodia6@gmail.com"
# Status = "Prototype"

# Description: Analyzes live video feed to track the center of a green ball's position
# over time and use that to calculate its velocity and acceleration.
# While a green ball is detected, a timer is also kept to later make graphs.
# The program can also accept a video as its input, but that has not been tested yet.
# Utilizes opencv and the matplotlib packages to track and graph the ball.
# Current program tracks position correctly, but whether the ball's velocity
# and acceleration graphs are accurate is unknown.
# Inputs to take into consideration: color range for object [ln 48-49], code for camera [ln 54]
# or whether using a video).

# Import libraries
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time
import matplotlib.pyplot as plt

# Create empty variables as stand-ins
all_coordinates = []  # will later hold ball's center coordinates
ts = [0, 0]  # list of start and end times
counter = 0  # counter to check the start time when the ball is first detected
times = []  # empty list of times to continually append to later

# Create arguments for whether video or webcam will be used
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="ball_tracking_example.mp4")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# Establish lower and upper bounds for the color of ball that you want to detect
# In this case, the program only detects green balls
# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
greenLower = (140,130,140)
greenUpper = (180,180,180)
pts = deque(maxlen=args["buffer"])


if not args.get("video.mp4", False):
    # Camera takes in video capture from webcam on system
    # Depending on device the number (camera ID) might need to be changed (most commonly '0' or '1')
    camera = cv2.VideoCapture(0)
else:
    # if there is a video, will go to video
    camera = cv2.VideoCapture(args["video.mp4"])

# allow the camera or video file to warm up
time.sleep(2.0)

# in while loop, frames are consistently inputted and analyzed
while True:
    # Reads frame from camera
    (grabbed, frame) = camera.read()

    # Breaks if video frames run out
    if args.get("video") and not grabbed:
        break

    # Resizes frame to width of 600 pixels
    frame = imutils.resize(frame, width=600)
    # Applies a gaussian blue to the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # Converts frame to black and white for masked window
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Creates mask over ball from RGB ranges specified earlier
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Finds the counters surrounding the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # If contours are found
    if len(cnts) > 0:
        # If this is the first time that a ball has been found
        if counter == 0:
            # Find time from 1990
            start = time.time()
            # Set this as first time that ball is found
            ts[0] = start
            # Append the time to times list
            times.append(ts[0])
            # Add 1 to the counter
            counter += 1
        # Else if the counter is greater than one
        elif counter > 0:
            # Gets next time that the ball is detected
            next1 = time.time()
            # Subtracts from start time to get time difference
            next2 = next1 - start
            times.append(next2)

        # Finds area around edge of circle
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Finds the center of the ball and generates its coordinates
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Append the coordinates of the center to list of all_coordinates
        all_coordinates.append(center)

        # If the radius of the ball is at least greater than 3
        if radius > 3:
            # Generates circle with radius of ball at the center of the ball to show
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # Appends center to different list of points
    pts.appendleft(center)

    # Loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Show windows of the frame (video from webcam) and mask (black and white video of ball)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Can optionally print out all coordinates to observe how they are changing
    # print(all_coordinates)

    key = cv2.waitKey(1) & 0xFF

    # If user presses q
    if key == ord("q"):
        # Take end time and subtract from start to find difference in start and end
        ts[1] = time.time() - start
        break  # break from while loop

# Closes the camera and mask windows
camera.release()
cv2.destroyAllWindows()

# Divisor to show rest of code after while loop
print("Rest of code\n")

# Can optionally print out how many coordinates exist
# print(len(all_coordinates))

# Print out time difference between start and end
print("Time Difference between Start and End")
print(ts[1], "seconds")

# Create list for all y-values of center circle
all_y = []
# For each center coordinate, append its y value
for point in all_coordinates:
    all_y.append(point[1])

# Optionally, print out y-values to the user
# print('All y-values: ' + str(all_y))

# Create list for all x-values of center circle
all_x = []
for point in all_coordinates:
    # For each center coordinate, append its x value
    all_x.append(point[0])

# Optionally, print out x-values to the user
# print('All x-values: ' + str(all_x))


# POSITION VS. TIME GRAPH

# x-axis is composed of ball's center x values
x = all_x
# y-axis is made of ball's center y values
y = all_y

# Creating the graph with x and y-axis
plt.plot(x, y)

# Label the x-axis
plt.xlabel('x-axis: position (pixels)')
# Label the y-axis
plt.ylabel('y-axis: position (pixels)')

# giving a title to my graph
plt.title('Y Position (pixels) of Ball vs X Position (pixels)')

# Function to show the plot in a window
plt.show()


# Y POSITION VS TIME

# X axis values are the time
times[0] = 0  # Convert first time value to 0

# Optionally, print out list of times to the user
# print(times)

# x axis values and corresponding y-axis values
x = times
y = all_y  # y positions of center of ball

# Creating the graph with x and y-axis
plt.plot(x, y)

# Label the x-axis
plt.xlabel('x-axis: time (s)')
# Label the y-axis
plt.ylabel('y-axis: position (pixels)')

# giving a title to my graph
plt.title('Y Position vs Time')

# Function to show the plot
plt.show()


# VELOCITY VS. TIME GRAPH

velocities = []
# Function to find the difference in position divided by difference in time to find velocity
for x in range(len(all_y)-1):
    dif = all_y[x+1] - all_y[x]
    dif_time = times[x+1] - times[x]
    vel = dif / dif_time
    velocities.append(vel)

# Modify times list by excluding first time of 0 so that time and y values are the same value
times = times[1:]

# Defining x-axis values and y-axis values
x = times
y = velocities

# Plotting the points
plt.plot(x, y)

# naming the x and y-axis
plt.xlabel('x-axis: Time (s)')
plt.ylabel('y-axis: Velocity (pixels/s)')

# Title  the graph
plt.title('Velocity vs Time')

# function to show the plot
plt.show()


# ACCELERATION VS TIME GRAPH

# Start with empty list of accelerations
acceleration = []

# function to find difference in velocity and divide by the difference in time to find acceleration
for x in range(len(velocities)-1):
    dif2 = velocities[x+1] - velocities[x]
    dif_time = times[x + 1] - times[x]
    accel = dif2 / dif_time
    # Add acceleration to list of accelerations
    acceleration.append(accel)

# Modify times list by excluding first time of 0 so that time and y values are the same value
times = times[1:]

# x axis value (times) and corresponding y-axis value (acceleration)
x = times
y = acceleration

# plotting the points onto the graph
plt.plot(x, y)

# Label the x and y-axis
plt.xlabel('x-axis: Time (s)')
plt.ylabel('y-axis: Acceleration (pixels/s^2)')

# Title  the graph
plt.title('Acceleration vs Time')

# function to show the plot
plt.show()

work = 0.5 * mass * vel**2
impactForce = (mass/time) * {(1 + c) * [(2/mass)*(work)]**0.5 - (radius/distance)*(c*s+s2)}
