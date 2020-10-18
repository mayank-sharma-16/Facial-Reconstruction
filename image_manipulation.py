import numpy as np

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
        return __replacePixels(range(20), range(len(frame)), broken_frame, 0, vertical = 50)
    if pattern_number == 1:
        for i in range(30):
            for j in range(30):
                broken_frame[int(len(broken_frame)/4)+j+20][i+20] = 0
    
    return broken_frame

def OverlayFrame(top_frame: np.ndarray, broken_frame: np.ndarray, pattern_number: int = 0) -> np.ndarray:
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
        return __replacePixels(range(20), range(len(broken_frame)), broken_frame, top_frame, vertical=50)
    if pattern_number == 1:
        for i in range(30):
            for j in range(30):
                broken_frame[int(len(broken_frame)/4)+j+20][i+20] = int(top_frame[int(len(top_frame)/4)+j+20][i+20]*254)

    return broken_frame
    
def __replacePixels(i_range, j_range, under, over, horizontal=0, vertical=0):

    under = np.copy(under)

    if type(over) == int:
        for i in i_range:
            for j in j_range:
                under[i + vertical][j + horizontal] = over
    else:
        for i in i_range:
            for j in j_range:
                under[i + vertical][j + horizontal] = over[i][j]

    return under