import numpy as np


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

    @staticmethod
    def remove_inliers_with_sharing_kps(inliers):
        """
        remove from inliers every match that shares a keypoint with another match
        :param inliers: list of matches
        :return: list of matches without matches that share keypoints
        """
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
            if len(inliers_keypoints_query[m.queryIdx]) == 1 and len(
                    inliers_keypoints_train[m.trainIdx]) == 1:
                final_inliers.append(m)
        return final_inliers


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
        inliers = Tracker.remove_inliers_with_sharing_kps(inliers)
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


class Y_GeometricRejectionTracker(Tracker):
    """
    Tracker class that uses geometric rejection based on the y-axis difference between the keypoints
    """
    def __init__(self, detector, matcher, threshold=2):
        super().__init__(detector, matcher)
        self.threshold = threshold

    def calculate_matches(self, desc1, desc2, **kwargs):
        """
        usage: class_instance.calculate_matches(desc1, desc2, kp1=kp1, kp2=kp2)
        calculate matches between two images, using geometric rejection with a threshold
        :param desc1: list of descriptors of first image
        :param desc2: list of descriptors of second image
        :return: list of inliers and list of outliers
        Note: this function uses the keypoints from the kwargs dictionary
        """
        kp1 = kwargs['kp1']
        kp2 = kwargs['kp2']
        inliers = []
        outliers = []
        matches = self.matcher.match(desc1, desc2)
        for m in matches:
            if abs(round(kp1[m.queryIdx].pt[1]) - round(kp2[m.trainIdx].pt[1])) <= self.threshold:
                inliers.append(m)
            else:
                outliers.append(m)
        return inliers, outliers

    def get_deviations(self, desc1, desc2, kp1, kp2):
        """
        calculate the absolute rounded y value difference of every match of desc1 and desc2
        :param desc1: list of descriptors of first image
        :param desc2: list of descriptors of second image
        :param kp1: list of keypoints of first image
        :param kp2: list of keypoints of second image
        :return: list of deviations
        """
        matches = self.matcher.match(desc1, desc2)
        deviations = []
        for m in matches:
            dv = abs(round(kp1[m.queryIdx].pt[1]) - round(kp2[m.trainIdx].pt[1]))
            deviations.append(dv)
        return deviations


class XY_GeometricRejectionTracker(Y_GeometricRejectionTracker):

    def calculate_matches(self, desc1, desc2, **kwargs):
        """
        usage: class_instance.calculate_matches(desc1, desc2, kp1=kp1, kp2=kp2).
        calculate matches between two images, using geometric rejection with a threshold and also rejection based on
        triangulation - pixels with left image x value smaller than right image x value are rejected.
        :param desc1: list of descriptors of first image
        :param desc2: list of descriptors of second image
        :return: list of inliers, list of outliers that were rejected by Y axis geometric rejection, list of outliers
        that were rejected by X axis geometric rejection.
        Note: this function uses the keypoints from the kwargs dictionary
        """
        kp1 = kwargs['kp1']
        kp2 = kwargs['kp2']
        inliers = []
        y_rejected = []
        x_rejected = []
        matches = self.matcher.match(desc1, desc2)
        for m in matches:
            if abs(round(kp1[m.queryIdx].pt[1]) - round(kp2[m.trainIdx].pt[1])) <= self.threshold:
                if kp1[m.queryIdx].pt[0] >= kp2[m.trainIdx].pt[0]:
                    inliers.append(m)
                else:
                    x_rejected.append(m)
            else:
                y_rejected.append(m)

        inliers = Tracker.remove_inliers_with_sharing_kps(inliers)
        return inliers, y_rejected, x_rejected



