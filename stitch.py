import cv2
import imutils
from Stitcher import Stitcher
import copy
import numpy as np

if __name__ == "__main__":
    imageA = cv2.imread("images/boat1.jpg")
    imageB = cv2.imread("images/boat2.jpg")
    imageA = imutils.resize(imageA, width=600)
    imageB = imutils.resize(imageB, width=600)
    # stitch the images together to create a panorama
    stitcher = Stitcher()
    (result, H, vis) = stitcher.stitch(imageB, imageA, showMatches=True)
    # get the wrapped image that we use for blending
    wrapped_image = cv2.warpPerspective(imageB, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
    src = copy.deepcopy(wrapped_image)
    # convert it to grayscale image
    im_bw = cv2.cvtColor(wrapped_image, cv2.COLOR_RGB2GRAY)
    # threshold the image
    ret, thresh_im = cv2.threshold(im_bw, 0, 255, 0)
    cv2.imshow("thresholded wrapped image", thresh_im)
    # calculate the contours from the black and white image
    _, contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # approximate the contour by polygon
    # this polygon is used as mask for the blending
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    polygon = cv2.approxPolyDP(contours[0], epsilon, True)
    src_mask = np.zeros(src.shape, src.dtype)
    cv2.fillPoly(src_mask, [polygon], (255, 255, 255))
    # calculate moment of contour
    M = cv2.moments(contours[0])
    # calculate center of contour from moment
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    dest = copy.deepcopy(result)
    # center of the srouce image
    center = (cx, int(dest.shape[0]/2))


    # dest = copy.deepcopy(result)
    # # src = cv2.imread("images/airplane.jpg")
    # src = imageB[0:400, 0:600] #imageB (400,600)
    # # Create a rough mask around the airplane.
    # src_mask = np.zeros(src.shape, src.dtype)
    # # poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
    # poly = np.array([[0, 0], [0, src.shape[0]-0], [src.shape[1]-0, src.shape[0]-0], [src.shape[1]-0, 0]])
    # cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    #
    # center = (300, 200)  # (imageB.shape[0]/2, imageB.shape[1]/2)
    output = cv2.seamlessClone(src, dest, src_mask, center, cv2.NORMAL_CLONE)
    cv2.imshow("src", src)
    cv2.imshow("blended", output)
    cv2.imshow("wrapped_image", wrapped_image)
    # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    # Save result
    cv2.imwrite("images/panorama.jpg", result)