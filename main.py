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
        input_capture: the VideoCapture object that feeds into the initial Windows.

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

    # Creates essential windows
    cv2.namedWindow("Artificial Errors")
    cv2.namedWindow("Reconstructed Face")

    # Creates windows for intermediary transformations if debug mode is on.
    if settings["debug"]:
        cv2.namedWindow("Input Face")
        cv2.namedWindow("Reconstructed Face - Small")

    return input_capture

def MakeErrors(frame: np.ndarray, pattern_number: int = 0) -> np.ndarray:
    """Corrupt a frame according to a predesignated pattern.
    
    Various types of patterns can be selected that change the values
    of the arrays of pixels for a single frame.

    Args:
        frame: a 2-dimensional np.ndarray with the black-and-white values of
          each pixel as a double.
        pattern_number: integer that correlates to a particular pattern.

    Returns:
        A new 2-dimensional np.ndarray with black-and-white values of each
          pixel as a double.
    """

    # TODO: add more patterns
    # TODO: refactor to account for more patterns

    broken_frame = np.copy(frame)
    
    if pattern_number == 0:
        for i in range(len(frame)):
            for j in range(20):
                broken_frame[int(len(broken_frame)/4)+j][i] = 0
    
    return broken_frame

def OverlayFrame(top_frame: np.ndarray, broken_frame: np.ndarray, pattern_number=0):
    """Fills in the broken bottom frame with parts of the top frame as per the specified
    pattern number.

    Runs on the assumption that the bottom frame has 'corrupted' blacked out portions that
    the top frame can fill in, using the same pattern used to corrupt it.

    Args:
        top_frame: a 2-dimensional np.ndarray with the black and white values of each pixel
          as a double.
        bottom_frame: a 2-dimensional np.ndarray with the black and white values of each pixel
          as a double.
        pattern_number: integer that correlates to a particular pattern.

    Returns:
        A new 2-dimensional np.ndarray with black-and-white values of each
          pixel as a double.

    """
    
    broken_frame = np.copy(broken_frame)

    if pattern_number == 0:
        for i in range(len(broken_frame)):
            for j in range(20):
                broken_frame[int(len(broken_frame)/4)+j][i] = int(top_frame[int(len(top_frame)/4)+j][i]*254)

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

        # Introduce artificial errors.
        broken_grey_frame = MakeErrors(cv2.resize(grey_frame, (256, 256)), 0)

        # Resize the frame to match the model dimensions.
        small_bgf = cv2.resize(broken_grey_frame, (64, 64))
        generated_frame = cv2.resize(RBM.get_RBM(small_bgf), (256, 256))
        fixed_frame = OverlayFrame(generated_frame, broken_grey_frame, 0)

        if settings["debug"] == True:
            cv2.imshow("Input Face", grey_frame)
            cv2.imshow("Reconstructed Face - Small", generated_frame)

        # Update all windows.
        cv2.imshow("Artificial Errors", broken_grey_frame)
        cv2.imshow("Reconstructed Face", fixed_frame)

        # Gets next frame.
        rval, frame = input_capture.read()
        key = cv2.waitKey(1)

        # Releases capture on ESC
        if key == 27:
            input_capture.release()
            break

    cv2.destroyAllWindows()




