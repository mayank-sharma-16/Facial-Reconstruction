import cv2
import numpy as np
import RBM

DEBUG = False

# set up video capture and window
cv2.namedWindow("FaceDream")
video_capture = cv2.VideoCapture(0)

# resizing capture to later calculate match for the trained model
# TODO: connect this with the model itself (low priority since model dimensions are unlikely to change), but should still avoid
# hardcoding 
video_capture.set(3, 256)
video_capture.set(4, 256)

# initialize capture
if video_capture.isOpened():
    rval, frame = video_capture.read()
else:
    rval = False

# while capture is active
while rval:

    # convert current frame to greyscale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if DEBUG == True:
        cv2.namedWindow("Default")
        cv2.imshow("Default", grey_frame)

    # introduce artificial errors
    for i in range(len(grey_frame)):
        for j in range(20):
            grey_frame[int(len(grey_frame)/4)+j][i] = 0

    if DEBUG == True:
        cv2.namedWindow("Artificial Errors")
        cv2.imshow("Artificial Errors", grey_frame)

    # resize current frame to match model size
    small_grey_frame = cv2.resize(grey_frame, (64, 64))
    fixed_frame = RBM.get_stacked(small_grey_frame)

    if DEBUG == True:
        cv2.namedWindow("Fixed Frame")
        cv2.imshow("Fixed Frame", fixed_frame)

    # create second window to demonstrate modification
    cv2.namedWindow("FaceDream2")
    large_fixed_frame = cv2.resize(fixed_frame, (256, 256))
    cv2.imshow("FaceDream2", large_fixed_frame)

    # repair the broken image
    for i in range(len(small_grey_frame)):
        for j in range(20):
            small_grey_frame[int(len(small_grey_frame)/4)+j][i] = fixed_frame[int(len(small_grey_frame)/4+j)][i]

    grey_frame = cv2.resize(small_grey_frame, (256, 256))

    # show the repaired version
    cv2.imshow("FaceDream", grey_frame)

    # get next frame
    rval, frame = video_capture.read()
    key = cv2.waitKey(20)

    # release capture on ESC
    if key == 27:
        video_capture.release()
        break

cv2.destroyWindow("FaceDream")
