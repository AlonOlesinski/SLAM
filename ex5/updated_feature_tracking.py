import random
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

class Tracker:
    """
    class for tracking features in two images
    """
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

    def calculate_kps_and_descs(self, img):
        """
        calculate keypoints and descriptors for an image
        :param img: image to calculate kps and desc for
        :return: list of keypoints and list of descriptors
        """
        # return self.detector.detectAndCompute(img, None)

        kps, desc = self.detector.detectAndCompute(img, None)

        # map each location to a list of indices of keypoints which are located in that location.
        kp_locations_dict = {} # key: (x,y) value: list of keypoints indices
        for i, kp in enumerate(kps):
            if (kp.pt[0], kp.pt[1]) not in kp_locations_dict:
                kp_locations_dict[(kp.pt[0], kp.pt[1])] = [i]
            else:
                kp_locations_dict[(kp.pt[0], kp.pt[1])].append(i)

        # remove keypoints that share the same location
        final_kps = []
        final_desc = []
        for v in kp_locations_dict.values():
            if len(v) == 1:
                final_kps.append(kps[v[0]])
                final_desc.append(desc[v[0]])
        final_kps = tuple(final_kps)
        final_desc = np.array(final_desc)
        return final_kps, final_desc

    def calculate_matches(self, desc1, desc2, *args, **kwargs):
        """
        calculate matches between two images
        :param desc1: list of descriptors of first image
        :param desc2: list of descriptors of second image
        :return: list of matches
        """
        inliers = self.matcher.match(desc1, desc2) # BFMatcher.match returns the best match.
        return inliers, []




class RatioTestTracker(Tracker):
    """
    class for tracking features in two images with ratio test.
    """
    def __init__(self, detector, matcher, ratio=0.5):
        super().__init__(detector, matcher)
        self.ratio = ratio

    def calculate_matches(self, desc1, desc2, **kwargs):
        """
        calculate matches between two images with ratio test
        :param desc1: list of descriptors of first image
        :param desc2: list of descriptors of second image
        :return: list of matches
        """
        inliers = []
        outliers = []
        if self.ratio > 0:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            for m1, m2 in matches:
                if m1.distance < self.ratio * m2.distance:
                    inliers.append(m1)
                else:
                    outliers.append(m1)

        else:
            inliers = self.matcher.match(desc1, desc2)

        # remove from inliers every match that shares a keypoint with another match
        inliers_keypoints_query = {}
        inliers_keypoints_train = {}
        for m in inliers:
            if m.queryIdx not in inliers_keypoints_query:
                inliers_keypoints_query[m.queryIdx] = [m]
            else:
                inliers_keypoints_query[m.queryIdx].append(m)
            if m.trainIdx not in inliers_keypoints_train:
                inliers_keypoints_train[m.trainIdx] = [m]
            else:
                inliers_keypoints_train[m.trainIdx].append(m)
        final_inliers = []
        for m in inliers:
            if len(inliers_keypoints_query[m.queryIdx]) == 1 and len(inliers_keypoints_train[m.trainIdx]) == 1:
                final_inliers.append(m)
        return final_inliers, outliers


if __name__ == "__main__":

    # # 1.1 + 1.2
    # left_1, right_1 = utils.read_images(0)
    # tracker = Tracker(cv2.AKAZE_create(), cv2.BFMatcher())
    # utils.plot_kps(tracker, [left_1, right_1], dest_path='1.1.png', print_desc=True)

    # 1.3

    detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher()  # cv2.NORM_L2 by default
    tracker = Tracker(detector, matcher)
    # utils.plot_matches(tracker, left_1, right_1, dest_path='1.3.png')
    #
    # # 1.4 show 20 true positive
    # left_1, right_1 = utils.read_images(0)
    # detector = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher()  # cv2.NORM_L2 by default
    # utils.plot_matches(RatioTestTracker(detector, matcher, ratio=0.5), left_1, right_1, dest_path='1.4.png')
    #
    #
    # # 1.4 show one false negative
    # left_1, right_1 = utils.read_images(0)
    # detector = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher()
    # utils.plot_false_negative_match(detector, matcher, left_1, right_1, dest_path='1.4fn.png')

    # print the time it take to find the keypoints and descriptors vs the time it takes to find the matches
    avg_kps_desc_time = 0
    avg_matches_time = 0
    for i in range(200):
        left_1, right_1 = utils.read_images(i)
        start = time.time()
        kps1, desc1 = tracker.calculate_kps_and_descs(left_1)
        kps2, desc2 = tracker.calculate_kps_and_descs(right_1)
        avg_kps_desc_time += time.time() - start
        start = time.time()
        inliers, outliers = tracker.calculate_matches(desc1, desc2)
        avg_matches_time += time.time() - start
    print("Average time to find keypoints and descriptors: ", avg_kps_desc_time / 200)
    print("Average time to find matches: ", avg_matches_time / 200)

