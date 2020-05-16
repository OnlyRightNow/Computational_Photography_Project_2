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
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    dest = copy.deepcopy(result)
    # src = cv2.imread("images/airplane.jpg")
    src = imageB[0:200, 0:300] #imageB (400,600)
    # Create a rough mask around the airplane.
    src_mask = np.zeros(src.shape, src.dtype)
    # poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
    poly = np.array([[10, 10], [10, src.shape[1]-10], [src.shape[0]-10, src.shape[1]-10], [src.shape[0]-10, 10]])
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    center = (520, 160)  # (imageB.shape[0]/2, imageB.shape[1]/2)
    output = cv2.seamlessClone(src, dest, src_mask, center, cv2.NORMAL_CLONE)
    cv2.imshow("src", src)
    cv2.imshow("blended", output)

    # show the images
    # cv2.imshow("Image A", imageA)
    # cv2.imshow("Image B", imageB)
    # cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)