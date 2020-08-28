# main.py

import cv2 
import numpy as np 
import sys

# Custom imports:
import RBM

def InitializeSettings() -> dict:
    """Reads command line arguments and changes settings as needed.
    
    Currently checks for: 
        - debug mode.

    Args:
        None

    Returns:
        A dictionary that contains the parameters for each setting.    
    """
    
    # TODO: cover cases where user enters non-standard arguments

    # Initialize default settings.
    settings = {
        "debug" : False
    }
    
    # Check for any new settings.
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--debug":
            settings["debug"] = True

    return settings


def SetupWindows(settings: dict) -> cv2.VideoCapture:
    """Sets up all the windows that need to opened.
    
    Takes into account whether results alone or debug windows
    must be created.
    
    Args:
        settings: dictionary containing strings as keys and various
          types as values depending on the setting.
          
    Returns:
        input_capture: the VideoCapture object used to get facial images.

    Raises: 
        CaptureOpenError: Unable to open default video capture channel.
    """

    # Opens video capture feed from default camera option.
    # TODO: allow user to change this setting or even use prerecorded videos.
    input_capture = cv2.VideoCapture(0)

    input_capture.set(3, 256)
    input_capture.set(4, 256)

    # Checks if video capture was successfully opened. If not, raise exception.
    class CaptureOpenError(Exception):
        
        def __init__(self):
            Exception.__init__(self, "Unable to open default video capture channel.")

    if not input_capture.isOpened():
        raise CaptureOpenError

    cv2.namedWindow("Input Face")
    cv2.namedWindow("Reconstructed Face")

    # Creates windows for intermediary transformations if debug mode is on.
    if settings["debug"]:
        cv2.namedWindow("Artificial Errors")
        cv2.namedWindow("Reconstructed Frame - Small")

    return input_capture

def MakeErrors(frame: np.ndarray, pattern_number: int = 0) -> np.ndarray:
    """Corrupt a frame according to a predesignated pattern.
    
    Various types of patterns can be selected that change the values
    of the arrays of pixels for a single frame.

    Args:
        frame: a 2-dimensional np.ndarray with the black-and-white values of
          each pixel as an integer.
        pattern_number: integer that correlates to a particular pattern.

    Returns:
        A new 2-dimensional np.ndarray with black-and-white values of each
          pixel as an integer.
    """

    # TODO: add more patterns
    # TODO: make a better process

    broken_frame = np.copy(frame)
    
    if pattern_number == 0:
        for i in range(len(frame)):
            for j in range(20):
                broken_frame[int(len(broken_frame)/4)+j][i] = 0
    
    return broken_frame

if __name__ == "__main__":

    settings = InitializeSettings()

    input_capture = SetupWindows(settings)

    # Initializes capture.
    rval, frame = input_capture.read()

    # Begins main loop.
    while rval:
        
        # Converts current frame to greyscale.
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Introduce Artificial Errors.
        broken_grey_frame = MakeErrors(grey_frame, 0)

        # Update all windows.
        cv2.imshow("Input Face", grey_frame)
        cv2.imshow("Reconstructed Face", broken_grey_frame)

        # Gets next frame.
        rval, frame = input_capture.read()
        key = cv2.waitKey(20)

        # Releases capture on ESC
        if key == 27:
            input_capture.release()
            break

    cv2.destroyAllWindows()




