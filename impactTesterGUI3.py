
# Co-Author: Dylan Winer
# Co-Author: Rishabh Kanodia
# Supervisor: Sean Psulkowski
# Supervisor: Bryant Rodriguez
# Research Sponsor: Dr. Dickens
# Date 07/07/2022
# Version 1.3
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
# Inputs to take into consideration: color range for object [ln 46-47], code for camera [ln 54]
# or whether using a video).

# New Update: Now a graphic user interface using PyQt

# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'ballTrackerGUI.ui'
# Created by: PyQt5 UI code generator 5.9.2

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time
import matplotlib.pyplot as plt
import os

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ballTrackerGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
# pyuic5 -x ballTrackerGUI.ui -o ballTrackerGUI3.py
# pyuic5 -x test.ui -o rish.py

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.browse = QPushButton(self.centralwidget)
        self.browse.setGeometry(QRect(130, 80, 201, 51))
        self.browse.setObjectName("browse")
        self.title = QLabel(self.centralwidget)
        self.title.setGeometry(QRect(200, 10, 361, 51))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.title.sizePolicy().hasHeightForWidth())
        self.title.setSizePolicy(sizePolicy)
        self.title.setSizeIncrement(QSize(2, 0))
        font = QFont()
        font.setPointSize(36)
        self.title.setFont(font)
        self.title.setMouseTracking(False)
        self.title.setAutoFillBackground(False)
        self.title.setObjectName("title")
        self.label1 = QLabel(self.centralwidget)
        self.label1.setGeometry(QRect(10, 100, 111, 21))
        font = QFont()
        font.setPointSize(18)
        self.label1.setFont(font)
        self.label1.setObjectName("label1")
        self.label2 = QLabel(self.centralwidget)
        self.label2.setGeometry(QRect(10, 160, 151, 21))
        font = QFont()
        font.setPointSize(18)
        self.label2.setFont(font)
        self.label2.setObjectName("label2")
        self.label3 = QLabel(self.centralwidget)
        self.label3.setGeometry(QRect(280, 160, 151, 21))
        font = QFont()
        font.setPointSize(18)
        self.label3.setFont(font)
        self.label3.setObjectName("label3")
        self.height = QLineEdit(self.centralwidget)
        self.height.setGeometry(QRect(170, 160, 71, 21))
        self.height.setObjectName("height")
        self.output = QLabel(self.centralwidget)
        self.output.setGeometry(QRect(320, 240, 81, 31))
        font = QFont()
        font.setPointSize(24)
        self.output.setFont(font)
        self.output.setObjectName("output")
        self.mass = QLineEdit(self.centralwidget)
        self.mass.setGeometry(QRect(450, 160, 71, 21))
        self.mass.setObjectName("mass")
        self.final_2 = QPushButton(self.centralwidget)
        self.final_2.setGeometry(QRect(580, 160, 151, 32))
        self.final_2.setObjectName("final_2")
        self.label4 = QLabel(self.centralwidget)
        self.label4.setGeometry(QRect(360, 100, 41, 21))
        font = QFont()
        font.setPointSize(24)
        self.label4.setFont(font)
        self.label4.setObjectName("label4")
        self.label5 = QLabel(self.centralwidget)
        self.label5.setGeometry(QRect(420, 100, 111, 21))
        font = QFont()
        font.setPointSize(18)
        self.label5.setFont(font)
        self.label5.setObjectName("label5")
        self.camera = QPushButton(self.centralwidget)
        self.camera.setGeometry(QRect(550, 80, 201, 51))
        self.camera.setObjectName("camera")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 758, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.final_2.clicked.connect(self.show_line_btn)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.browse.setText(_translate("MainWindow", "Browse for video"))
        self.title.setText(_translate("MainWindow", "Impact Tester Program"))
        self.label1.setText(_translate("MainWindow", "1.  Input Video:"))
        self.label2.setText(_translate("MainWindow", "2.  Input Height (ft):"))
        self.label3.setText(_translate("MainWindow", "3.  Input Mass (kg):"))
        self.output.setText(_translate("MainWindow", "Output"))
        self.final_2.setText(_translate("MainWindow", "Press when finished"))
        self.label4.setText(_translate("MainWindow", "OR:"))
        self.label5.setText(_translate("MainWindow", "Use Camera:"))
        self.camera.setText(_translate("MainWindow", "Use Camera"))
        self.browse.clicked.connect(self.browse_handler)
        self.camera.clicked.connect(self.camera_handler)

    def camera_handler(self):
        global camera_true
        camera_true = False
        print("Chose Camera")

    def browse_handler(self):
        print("I am Elon Musk Jr.")
        self.open_dialog_box()

    def ballTracker3(self):
        # Create empty variables as stand-ins
        all_coordinates = []  # will later hold ball's center coordinates
        ts = [0, 0]  # list of start and end times
        counter = 0  # counter to check the start time when the ball is first detected
        times = []  # empty list of times to continually append to later

        m = mass
        g = 9.80665
        h = height

        h = h * 0.3048  # convert from ft to meters

        Energy = m * g * h
        print("Potential Energy in Joules: ", Energy)

        # Create arguments for whether video or webcam will be used

        ap = argparse.ArgumentParser()
        if camera_true == True:
            ap.add_argument("-v", "--" + newname, help="video")
        ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
        args = vars(ap.parse_args())

        # Establish lower and upper bounds for the color of ball that you want to detect
        # In this case, the program only detects green balls
        # Potential yellowLower: (219,156,22)
        # Potential yellowUpper: (255,220,95)
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        pts = deque(maxlen=args["buffer"])

        # Comment out this code if want to use video
        # '''
        if camera_true == False:
            # Camera takes in video capture from webcam on system
            # Depending on device the number (camera ID) might need to be changed (most commonly '0' or '1')
            camera = cv2.VideoCapture(1)
        else:
         # '''
            # if there is a video, will go to video
            # If commenting out the camera, need to remove indent for camera on line below
            camera = cv2.VideoCapture(str(newname))

        # allow the camera or video file to warm up
        time.sleep(2.0)

        on = True
        # in while loop, frames are consistently inputted and analyzed
        while on:
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
                # Finds area around edge of circle
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)

                # Finds the center of the ball and generates its coordinates
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # Append the coordinates of the center to list of all_coordinates
                all_coordinates.append(center)

                if counter == 0:
                    # Find time from 1990
                    start = time.time()
                    # Set this as first time that ball is found
                    ts[0] = start
                    # Append the time to times list
                    times.append(ts[0])
                    # Add 1 to the counter
                    counter += 1
                if counter > 0:
                    # Gets next time that the ball is detected
                    next1 = time.time()
                    # Subtracts from start time to get time difference
                    next2 = next1 - start
                    times.append(next2)

                for x in range(4, len(all_coordinates)):
                    print(all_coordinates[x])
                    if all_coordinates[x] == all_coordinates[x - 1] == all_coordinates[x - 2] == all_coordinates[
                        x - 3] == all_coordinates[x - 4]:
                        # Take end time and subtract from start to find difference in start and end
                        ts[1] = time.time() - start
                        on = False

                # If the radius of the ball is at least greater than 3
                if radius > 0.1:
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
            # print(all_coordinates[x])

            key = cv2.waitKey(1) & 0xFF

            # If user presses q
            if key == ord("q"):
                break  # break from while loop

        # Closes the camera and mask windows
        camera.release()
        cv2.destroyAllWindows()

        # Divisor to show rest of code after while loop
        print("Rest of code\n")

        # Can optionally print out how many coordinates exist
        print(len(all_coordinates))

        # Print out time difference between start and end
        print("Time Difference between Start and End")
        print(ts[1], "seconds")

        # Create list for all y-values of center circle
        all_y = []
        # For each center coordinate, append its y value
        # 32 pixels = 1 inch
        for point in all_coordinates:
            y_point = point[1] / 32
            all_y.append(y_point)

        # Optionally, print out y-values to the user
        print('All y-values: ' + str(all_y))

        # Create list for all x-values of center circle
        # 32 pixels = 1 inch
        all_x = []
        for point in all_coordinates:
            # For each center coordinate, append its x value
            x_point = point[0] / 32
            all_x.append(x_point)

        # Optionally, print out x-values to the user
        print('All x-values: ' + str(all_x))

        def x_y_graph():
            # Y POSITION VS. X POSITION GRAPH

            # x-axis is composed of ball's center x values
            x = all_x
            # y-axis is made of ball's center y values
            y = all_y

            # Creating the graph with x and y-axis
            plt.figure(0)
            plt.plot(x, y)

            # Label the x-axis
            plt.xlabel('x-axis: position (feet)')
            # Label the y-axis
            plt.ylabel('y-axis: position (feet)')

            # giving a title to my graph
            plt.title('Y Position (feet) of Ball vs X Position (feet)')

            # Function to show the plot in a window
            print("SHOw?")
            plt.show()

        # Y POSITION VS TIME
        def y_time_graph():
            # X axis values are the time
            times[0] = 0  # Convert first time value to 0

            # x axis values and corresponding y-axis values
            times.pop(1)

            # Optionally, print out list of times to the user
            print("Times:", times)

            x = times
            y = all_y  # y positions of center of ball

            # Creating the graph with x and y-axis
            plt.figure(1)
            plt.plot(x, y)

            # Label the x-axis
            plt.xlabel('x-axis: time (s)')
            # Label the y-axis
            plt.ylabel('y-axis: position (feet)')

            # giving a title to my graph
            plt.title('Y Position vs Time')

            # Function to show the plot
            plt.show()
            return times

        # VELOCITY VS. TIME GRAPH
        def velocity_time_graph(times):
            velocities = []
            # Function to find the difference in position divided by difference in time to find velocity
            for x in range(len(all_y) - 1):
                dif = all_y[x + 1] - all_y[x]
                dif_time = times[x + 1] - times[x]
                vel = dif / dif_time
                velocities.append(vel)

            # Modify times list by excluding first time of 0 so that time and y values are the same value
            times = times[1:]

            # Defining x-axis values and y-axis values
            x = times
            y = velocities
            max_vel = max(velocities)

            # Plotting the points
            plt.figure(2)
            plt.plot(x, y)

            # naming the x and y-axis
            plt.xlabel('x-axis: Time (s)')
            plt.ylabel('y-axis: Velocity (feet/s)')

            # Title  the graph
            plt.title('Velocity vs Time')

            # function to show the plot
            plt.show()
            return velocities

        # ACCELERATION VS TIME GRAPH
        def acceleration_time(times, velocities):
            # Start with empty list of accelerations
            acceleration = []

            # function to find difference in velocity and divide by the difference in time to find acceleration
            for x in range(len(velocities) - 1):
                dif2 = velocities[x + 1] - velocities[x]
                dif_time = times[x + 1] - times[x]
                accel = dif2 / dif_time
                # Add acceleration to list of accelerations
                acceleration.append(accel)

            # Modify times list by excluding first time of 0 so that time and y values are the same value
            times = times[1:]

            # x axis value (times) and corresponding y-axis value (acceleration)
            x = times
            times.pop(1)
            y = acceleration

            # plotting the points onto the graph
            plt.figure(3)
            plt.plot(x, y)

            # Label the x and y-axis
            plt.xlabel('x-axis: Time (s)')
            plt.ylabel('y-axis: Acceleration (feet/s^2)')

            # Title  the graph
            plt.title('Acceleration vs Time')
            plt.show()

        x_y_graph()
        y_time_graph()
        velocities = velocity_time_graph(times)
        acceleration_time(times, velocities)

# Force Calculations
    '''
    K_Energy = (m * (max_vel ** 2)) / 2
    print("Kinetic Energy (Joules):", K_Energy)

    Work = (1 / 2) * m * (max_vel ** 2)

    max_vel = (2 / m) * Work
    max_vel = max_vel ** 0.5

    # r = ratio of the mass center radius
    r = 1
    c = 0.1
    max_pos = max(all_y)
    min_pos = min(all_y)
    d = max_pos - min_pos

    p1 = (2 / m) * Work ** 0.5
    p1 = p1 * (d / r)

    p2 = c * -p1

    v1 = 0
    difv = v1 - max_vel
    difv = (r / d) * (p1 - p2)

    cs1 = 0
    s2 = 0

    difv = (r / d) * ((d / r)((2 / M)(Work)) ** 0.5 - cs1 - s2 + c((d / r)((2 / M)(Work)) ** 0.5))

    Impulse = m(v1 - max_vel)

    impact_impulse = m * {(1 + c)[(2 / m)(Work)] ** 0.5 - (r / d)(cs1 + s2)}

    t = max(times)
    impact_force = (m / t) * {(1 + c) * [(2 / m) * (Work)] ** 0.5 - (r / d) * (cs1 + s2)}

    print("Impact force from test: ", impact_force)
    '''

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        print(filename)
        path = filename[0]
        print(path)
        global newname
        newname = os.path.basename(path)
        print("Newname", newname)
        global camera_true
        camera_true = True

    def show_line_btn(self):
        self.show_line(self.ballTracker3)

    def show_line(self, ballTracker3):
        global height
        global mass
        height = float(self.height.text())
        mass = float(self.mass.text())
        print(type(ballTracker3))
        ballTracker3()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
