import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
import updated_feature_tracking as ft

DOCS_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\docs\ex2\\'


class Y_GeometricRejectionTracker(ft.Tracker):
    """
    Tracker class that uses geometric rejection based on the y axis difference between the keypoints
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
        return final_inliers, y_rejected, x_rejected




def triangulate_point(P, Q, p, q):
    """
    calculate the location of a point in 3d based on location in two images
    :param P - 3x4 projection matrix of the first camera (to the center of the first camera)
    :param Q - 3x4 projection matrix of the second camera (to the center of the first camera)
    :param p - a point seen by the first camera
    :param q - a point seen by the second camera (hopefully the same point in the real world as p)
    return - a 4d (homogenous) vector representing the real world location of the point, (in
    relation to the first camera's position)
    """
    A_row_1 = P[2] * p[0] - P[0]
    A_row_2 = P[2] * p[1] - P[1]
    A_row_3 = Q[2] * q[0] - Q[0]
    A_row_4 = Q[2] * q[1] - Q[1]
    A = np.array([A_row_1, A_row_2, A_row_3, A_row_4])
    u, s, vh = np.linalg.svd(A)
    return vh[-1].reshape((4, 1))

def triangulate_multiple_points(P, Q, p_array, q_array):
    """
    calculate the location of multiple points in 3d based on location in two images. Performed in a vectorized manner.
    :param P - 3x4 projection matrix of the first camera (to the center of the first camera)
    :param Q - 3x4 projection matrix of the second camera (to the center of the first camera)
    :param p_array - a list of points seen by the first camera
    :param q_array - a list of points seen by the second camera (hopefully the same point in the real world as p)
    return - a list of 4d (homogenous) vectors representing the real world location of the points, (in
    relation to the first camera's position)
    """
    A_row_1_stack = np.outer(p_array[:, 0] ,P[2]) - P[0]
    A_row_2_stack = np.outer(p_array[:, 1] ,P[2]) - P[1]
    A_row_3_stack = np.outer(q_array[:, 0] ,Q[2]) - Q[0]
    A_row_4_stack = np.outer(q_array[:, 1] ,Q[2]) - Q[1]
    # stack the above rows into a Nx4x4 matrix
    A = np.stack((A_row_1_stack, A_row_2_stack, A_row_3_stack, A_row_4_stack), axis=1)
    u, s, vh = np.linalg.svd(A)
    return vh[:, -1].T


if __name__ == "__main__":

    # # 2.1
    # left_1, right_1 = utils.read_images(0)
    # akaze = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # deviations_hist(left_1, right_1, akaze, matcher)
    # tracker = GeometricRejectionTracker(akaze, matcher)
    # kp1, desc1 = tracker.calculate_kps_and_descs(left_1)
    # kp2, desc2 = tracker.calculate_kps_and_descs(right_1)
    # matches = tracker.calculate_matches(desc1, desc2, kp1=kp1, kp2=kp2)
    # deviations = tracker.get_deviations(desc1, desc2, kp1, kp2)
    # utils.plot_deviation_histogram(deviations, DOCS_PATH + '2.1.png')



    # 2.2
    # left_1, right_1 = utils.read_images(0)
    # akaze = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # tracker = Y_GeometricRejectionTracker(akaze, matcher)
    # utils.plot_geometric_rejection(tracker, left_1, right_1, DOCS_PATH + '2.2_alt.png')

    # 2.3
    left_1, right_1 = utils.read_images(0)
    akaze = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    tracker1 = XY_GeometricRejectionTracker(akaze, matcher)
    tracker2 = XY_GeometricRejectionTracker(akaze, matcher)
    # points1 = show_triangulated_points(left_1, right_1, akaze, matcher, triangulate_point)
    # points2 = show_triangulated_points(left_1, right_1, akaze, matcher, cv2.triangulatePoints)
    points1 = utils.plot_triangulated_points(tracker1, left_1, right_1, triangulate_point, DOCS_PATH + '2.3_my_func_alt.png')
    points2 = utils.plot_triangulated_points(tracker1, left_1, right_1, cv2.triangulatePoints, DOCS_PATH + '2.3_cv2_func_alt.png')
    print(f'median distance: {np.median(np.abs(np.array(points1) - np.array(points2)))}')
