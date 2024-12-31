import cv2
import numpy as np
from time import sleep
import time

width_min = 55  # Minimum width of rectangle
height_min = 55  # Minimum height of rectangle
offset = 8  # Allowed error between pixels
line_pos_y = 550  # Count line position
delay = 60  # Video FPS

dect1 = []  # center points of cars in lane 1
dect2 = []  # center points of cars in lane 2


def find_center(x, y, w, h):  # x,y start_points and w,h end_points
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('testvideo.mp4')  # live = id of camera

# Extract moving objects from static background
sub_algo = cv2.createBackgroundSubtractorMOG2()

cars1 = 0
cars2 = 0
totalcars1 = 0
totalcars2 = 0

while True:
    ret, frame1 = cap.read()
    
    if not ret:
        print("Error: Couldn't read the frame. Exiting...")
        break

    time.sleep(1 / delay)

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    img_sub = sub_algo.apply(blur)

    dilate = cv2.dilate(img_sub, np.ones((5, 5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (30, line_pos_y), (550, line_pos_y), (0, 255, 0), 4)
    cv2.line(frame1, (720, line_pos_y), (1200, line_pos_y), (0, 255, 0), 4)

    for i, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= width_min) and (h >= height_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 255), 3)

        center = find_center(x, y, w, h)

        if x < 550:
            dect1.append(center)
        else:
            dect2.append(center)

        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    # Handling cars detected in lane 1
    for (x, y) in dect1[:]:
        if y < (line_pos_y + offset) and y > (line_pos_y - offset):
            cars1 += 1
            totalcars1 += 1
            dect1.remove((x, y))
            cv2.line(frame1, (30, line_pos_y), (550, line_pos_y), (0, 127, 255), 4)
            print(f"Car detected in Lane 1: {cars1}")

    # Handling cars detected in lane 2
    for (x, y) in dect2[:]:
        if y < (line_pos_y + offset) and y > (line_pos_y - offset):
            cars2 += 1
            totalcars2 += 1
            dect2.remove((x, y))
            cv2.line(frame1, (720, line_pos_y), (1200, line_pos_y), (0, 127, 255), 4)
            print(f"Car detected in Lane 2: {cars2}")

    # Display vehicle count on screen
    cv2.putText(frame1, f"VEHICLE COUNT lane 2 : {cars2}", (650, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame1, f"VEHICLE COUNT lane 1 : {cars1}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Show the frames
    cv2.imshow("Implementation", frame1)
    cv2.imshow("Mask", dilated)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
