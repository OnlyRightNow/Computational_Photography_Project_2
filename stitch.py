import time

import cv2
import imutils
from Stitcher import Stitcher
import copy
import numpy as np

if __name__ == "__main__":
    imageA = cv2.imread("images/random_forest2.jpg")
    imageB = cv2.imread("images/random_forest1_lowexposure.jpg")
    # imageA = cv2.imread("images/boat1.jpg")
    # imageB = cv2.imread("images/boat2.jpg")
    imageA = imutils.resize(imageA, width=2000)
    imageB = imutils.resize(imageB, width=2000)
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
    start = time.time()
    # apply poisson blending
    output = cv2.seamlessClone(src, dest, src_mask, center, cv2.NORMAL_CLONE)
    end = time.time()
    print("running time poisson blending: ", end-start)
    # highlight the blended image by overlaying it the with the transparent blending mask
    alpha = 0.3
    blk = np.zeros(output.shape, output.dtype)
    cv2.fillPoly(blk, [polygon], (0, 0, 255))
    output_mask = cv2.addWeighted(output, 1-alpha, blk, alpha, 0)
    cv2.imshow("src", src)
    cv2.imshow("Wrapped Image", wrapped_image)
    cv2.imshow("Blending mask", src_mask)
    # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Normal Panorama", result)
    cv2.imshow("Blended Panorama", output)
    cv2.waitKey(0)
    # Save result
    name = "randomforest_lowexposure"
    cv2.imwrite("images/results/" + name + "_panorama_naive.jpg", result)
    cv2.imwrite("images/results/" + name + "_panorama_blended.jpg", output)
    cv2.imwrite("images/results/" + name + "_panorama_blended_mask.jpg", output_mask)
    cv2.imwrite("images/results/" + name + "_blending_mask.jpg", src_mask)
    cv2.imwrite("images/results/" + name + "_matches.jpg", vis)