import cv2
from imutils.perspective import four_point_transform
import imutils
from data import get_sudoku

# localize sudoku in image
# flatten perspective?
# TODO split cells
# TODO OCR cells
# TODO formulate constraints (other types of sudoku?)
# TODO solve puzzle


class ImageError(Exception):
    pass


def crop_sudoku_img(img, debug=False):
    """
    Extract the sudoku from an image that mostly contains a sudoku.

    Base method taken from [1] but extended with my own ideas, namely selecting
    the bounding box that has large area and is convex.

    [1] https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
    """
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1
    )
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow("threshold", thresh)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            # We improve the classification by:
            # - checking for convex contours
            # - checking for minimum area contours
            # - TODO square-ish?
            # - TODO maximum size compared to others? instead of the first large enough
            if cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                rel_area = area / (img.shape[0] * img.shape[1])
                if rel_area > 0.1:
                    puzzleCnt = approx
                    break
    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise ImageError(("Could not find Sudoku puzzle outline."))

    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = img.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    # puzzle = four_point_transform(img, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        # cv2.imshow("Puzzle Transform", puzzle)
        cv2.imshow("Warped greyscale", warped)

    return warped


for img, data in get_sudoku(n=None):
    try:
        warped = crop_sudoku_img(img, debug=False)
    except ImageError as e:
        cv2.imshow("image", img)
        cv2.waitKey(0)


cv2.destroyAllWindows()
