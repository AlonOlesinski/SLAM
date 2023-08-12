import cv2
import numpy as np
import random

import consensus_matching_dir.feature_tracking as ft
import consensus_matching_dir.triangulation as tr
import consensus_matching_dir.utils as utils
from shared_utils import read_images

random.seed(0)  # for reproducibility


class Localizer:
    """
    Class that localizes a camera in the next stereo pair, given the previous stereo pair and the previous camera pose.
    """

    def __init__(self, frame0_index, detector, matcher, left0_img, right0_img, left1_img,
                 right1_img, left0_ex_camera,
                 right0_ex_camera, intrinsic_matrix, track_db=None,
                 kp_left0=None, desc_left0=None, kp_right0=None, desc_right0=None,
                 left0_right0_matches=None):
        self.frame0_index = frame0_index
        self.track_db = track_db
        self.basic_tracker = ft.RatioTestTracker(detector, matcher, ratio=0.6)
        self.geometric_rejection_tracker = ft.XY_GeometricRejectionTracker(detector, matcher)
        self.left0_img = left0_img
        self.right0_img = right0_img
        self.left1_img = left1_img
        self.right1_img = right1_img
        self.left0_ex_camera = left0_ex_camera
        self.right0_ex_camera = right0_ex_camera
        self.intrinsic_matrix = intrinsic_matrix
        self.left0_camera = intrinsic_matrix @ left0_ex_camera
        self.right0_camera = intrinsic_matrix @ right0_ex_camera
        self.kp_left1, self.desc_left1 = self.basic_tracker.calculate_kps_and_descs(left1_img)
        self.kp_right1, self.desc_right1 = self.basic_tracker.calculate_kps_and_descs(right1_img)
        self.kp_left0, self.desc_left0 = kp_left0, desc_left0
        self.kp_right0, self.desc_right0 = kp_right0, desc_right0
        self.kp_right0, self.desc_right0 = self.basic_tracker.calculate_kps_and_descs(right0_img)
        self.left0_right0_matches = left0_right0_matches
        if left0_right0_matches is None:
            self.kp_left0, self.desc_left0 = self.basic_tracker.calculate_kps_and_descs(left0_img)
            self.kp_right0, self.desc_right0 = self.basic_tracker.calculate_kps_and_descs(
                right0_img)
            self.left0_right0_matches = \
                self.geometric_rejection_tracker.calculate_matches(self.desc_left0,
                                                                   self.desc_right0,
                                                                   kp1=self.kp_left0,
                                                                   kp2=self.kp_right0)[0]
        self.left1_right1_matches = \
            self.geometric_rejection_tracker.calculate_matches(self.desc_left1, self.desc_right1,
                                                               kp1=self.kp_left1,
                                                               kp2=self.kp_right1)[0]
        self.left0_left1_matches = \
            self.basic_tracker.calculate_matches(self.desc_left0, self.desc_left1)[0]

        self.matches_idx_dict = {}  # key: left0_left1_match_idx, value: (left0_right0_match_idx, left1_right1_match_idx)
        self.create_matches_dict()

    def create_matches_dict(self):
        """
        create a dictionary that maps matches from left0_left1 to left0_right0 and left1_right1.
        """
        left0_right0_dict = {m.queryIdx: i for i, m in enumerate(self.left0_right0_matches)}
        left1_right1_dict = {m.queryIdx: i for i, m in enumerate(self.left1_right1_matches)}
        matches_dict = {}
        for m_idx in range(len(self.left0_left1_matches)):
            m = self.left0_left1_matches[m_idx]
            if m.queryIdx in left0_right0_dict.keys() and m.trainIdx in left1_right1_dict.keys():
                matches_dict[m_idx] = (left0_right0_dict[m.queryIdx], left1_right1_dict[m.trainIdx])
        self.matches_idx_dict = matches_dict

    def fill_track_db(self, supporters, R_t):
        """
        fill the track database with the matches between the left0 and right0 images.
        :param supporters: list of indices of matches between left0 and left1 that are used to add
        the matches to the track
        :param R_t: extrinsic matrix of left1 relative to left0 (it takes points at the coordinate system of left0
        and transforms them to the coordinate system of left1)
        """
        l0_id = self.frame0_index
        if l0_id == 0:
            self.track_db.relative_R_t_list.append(self.left0_ex_camera)
        l1_id = self.frame0_index + 1
        self.track_db.relative_R_t_list.append(R_t)
        for m_idx in supporters:
            left0_right0_match_idx, left1_right1_match_idx = self.matches_idx_dict[m_idx]
            left0_right0_match = self.left0_right0_matches[left0_right0_match_idx]
            left1_right1_match = self.left1_right1_matches[left1_right1_match_idx]
            x_l0, y_l0 = self.kp_left0[left0_right0_match.queryIdx].pt
            x_r0, y_r0 = self.kp_right0[left0_right0_match.trainIdx].pt
            x_l1, y_l1 = self.kp_left1[left1_right1_match.queryIdx].pt
            x_r1, y_r1 = self.kp_right1[left1_right1_match.trainIdx].pt
            self.track_db.add_match(l0_id, x_l0, y_l0, x_r0, y_r0, l1_id, x_l1, y_l1, x_r1, y_r1)
        self.track_db.reset_last_added_tracks()

    def calculate_pnp(self, left0_left1_match_idxs):
        """
        calculate the pose of the camera in the next stereo pair, given 4 matches between the left images
        of the previous and next stereo pairs.
        :param left0_left1_match_idxs: list of 4 indices of matches between left0 and left1 images.
        :return: 3x4 matrix, which transforms the previous coordinate system to the next coordinate system.
        """
        if len(left0_left1_match_idxs) != 4:
            solver = cv2.SOLVEPNP_ITERATIVE
        else:
            solver = cv2.SOLVEPNP_AP3P
        image_points = []  # 2d points on left1 image plane
        world_points = []  # 3d points calculated by triangulation process of left0, right0.
        for m_idx in left0_left1_match_idxs:
            # add the 2d point on left1 image plane
            image_points.append(self.kp_left1[self.left0_left1_matches[m_idx].trainIdx].pt)
            # add the 3d point calculated by triangulation process of left0, right0.
            left0_right0_match_idx, _ = self.matches_idx_dict[m_idx]
            left0_right0_match = self.left0_right0_matches[left0_right0_match_idx]
            p3d = self.triangulate(left0_right0_match.queryIdx, left0_right0_match.trainIdx)
            world_points.append(p3d)
        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(world_points, image_points, self.intrinsic_matrix, None,
                                           flags=solver)
        if success:
            rot, _ = cv2.Rodrigues(rvec)
            return np.hstack((rot, tvec))

    def calculate_supporters_and_deniers(self, R_t, threshold=1, return_indices=True):
        """
        calculate the supporters and deniers of the given R_t matrix.
        :param R_t: 3x4 matrix, which transforms the previous coordinate system to the next coordinate system.
        :param threshold: threshold for the reprojection error.
        :param return_indices: if True, return the indices of the supporters and deniers, otherwise return the
        supporters and deniers themselves.
        """

        left0_kp_idx = [self.left0_left1_matches[m_idx].queryIdx for m_idx in
                        self.matches_idx_dict.keys()]
        right0_kp_idx = [self.left0_right0_matches[self.matches_idx_dict[m_idx][0]].trainIdx for
                         m_idx in
                         self.matches_idx_dict.keys()]
        left1_kp_idx = [self.left0_left1_matches[m_idx].trainIdx for m_idx in
                        self.matches_idx_dict.keys()]
        p3d = self.triangulate_multiple_l0_r0(left0_kp_idx, right0_kp_idx)
        p2d_projected = utils.project_points(p3d, self.intrinsic_matrix @ R_t).T
        p2d = np.array([self.kp_left1[kp_idx].pt for kp_idx in left1_kp_idx])
        supporters = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) < threshold)[0]
        deniers = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) >= threshold)[0]
        # convert the supporters and deniers to the original indices
        supporters = [list(self.matches_idx_dict.keys())[i] for i in supporters]
        deniers = [list(self.matches_idx_dict.keys())[i] for i in deniers]
        if return_indices:
            return supporters, deniers
        else:
            return [(self.kp_left0[self.left0_right0_matches[self.matches_idx_dict[i][0]].queryIdx],
                     self.kp_left1[self.left1_right1_matches[self.matches_idx_dict[i][1]].queryIdx])
                    for i in supporters], \
                   [(self.kp_left0[self.left0_right0_matches[self.matches_idx_dict[i][0]].queryIdx],
                     self.kp_left1[self.left1_right1_matches[self.matches_idx_dict[i][1]].queryIdx])
                    for i in deniers]

    def get_l0l1_kps_from_match_idx(self, idx):
        """
        get the keypoints in left0 and left1 images, given the index of the match between them.
        :param idx: index of the match between left0 and left1 images.
        :return: keypoints in left0 and left1 images.
        """
        left0_kp_idx, left1_kp_idx = self.left0_left1_matches[idx].queryIdx, \
                                     self.left0_left1_matches[idx].trainIdx
        return self.kp_left0[left0_kp_idx], self.kp_left1[left1_kp_idx]

    def triangulate(self, left0_kp_idx, right0_kp_idx):
        """
        triangulate the 3d point from the left0 and right0 images.
        :param left0_kp_idx: index of the keypoint in left0 image
        :param right0_kp_idx: index of the keypoint in right0 image
        :return: 3d point
        """
        p = self.kp_left0[left0_kp_idx].pt
        q = self.kp_right0[right0_kp_idx].pt
        p4d = tr.triangulate_point(self.left0_camera, self.right0_camera, p, q)
        p3d = p4d[:3] / p4d[3]
        return p3d

    def triangulate_multiple_l0_r0(self, left0_kp_idxs, right0_kp_idxs):
        """
        triangulate multiple 3d points from the left0 and right0 images.
        :param left0_kp_idxs: list of indices of the keypoints in left0 image
        :param right0_kp_idxs: list of indices of the keypoints in right0 image
        :return: list of 3d points
        """
        p = np.array([self.kp_left0[idx].pt for idx in left0_kp_idxs])
        q = np.array([self.kp_right0[idx].pt for idx in right0_kp_idxs])
        p4d = tr.triangulate_multiple_points(self.left0_camera, self.right0_camera, p, q)
        p3d = p4d[:3] / p4d[3]
        return p3d

    def triangulate_multiple_l1_r1(self, left1_kp_idxs, right1_kp_idxs, left1_extrinsic_matrix,
                                   right1_extrinsic_matrix):
        """
        triangulate multiple 3d points from the left1 and right1 images.
        :param left1_kp_idxs: list of indices of the keypoints in left1 image
        :param right1_kp_idxs: list of indices of the keypoints in right1 image
        :return: list of 3d points
        """
        p = np.array([self.kp_left1[idx].pt for idx in left1_kp_idxs])
        q = np.array([self.kp_right1[idx].pt for idx in right1_kp_idxs])
        p4d = tr.triangulate_multiple_points(left1_extrinsic_matrix, right1_extrinsic_matrix, p, q)
        p3d = p4d[:3] / p4d[3]
        return p3d

    def triangulate_all_l0_r0(self):
        """
        triangulate all the 3d points from the left0 and right0 images.
        :return: list of 3d points
        """
        left0_kp_idxs = [self.left0_right0_matches[m_idx[0]].queryIdx for m_idx in
                         self.matches_idx_dict.values()]
        right0_kp_idxs = [self.left0_right0_matches[m_idx[0]].trainIdx for m_idx in
                          self.matches_idx_dict.values()]
        return self.triangulate_multiple_l0_r0(left0_kp_idxs, right0_kp_idxs)

    def triangulate_all_l1_r1(self, left1_extrinsic_matrix, right1_extrinsic_matrix):
        """
        triangulate all the 3d points from the left1 and right1 images.
        :return: list of 3d points
        """
        left1_kp_idxs = [self.left1_right1_matches[m_idx[1]].queryIdx for m_idx in
                         self.matches_idx_dict.values()]
        right1_kp_idxs = [self.left1_right1_matches[m_idx[1]].trainIdx for m_idx in
                          self.matches_idx_dict.values()]
        return self.triangulate_multiple_l1_r1(left1_kp_idxs, right1_kp_idxs,
                                               left1_extrinsic_matrix,
                                               right1_extrinsic_matrix)

    def plot_match_on_4_images(self, match_idx, title=""):
        """
        Show four images (top left: left0, top right: right0, bottom left: left1, bottom right: right1), and plot the
         keypoints that were matched.
        :param match_idx: integer - index of the match in the matches_idx_dict
        :param title: string - title of the plot
        """
        left0_right0_match_idx, left1_right1_match_idx = self.matches_idx_dict[match_idx]
        left0_right0_match = self.left0_right0_matches[left0_right0_match_idx]
        left1_right1_match = self.left1_right1_matches[left1_right1_match_idx]
        left0_kp = self.kp_left0[left0_right0_match.queryIdx]
        right0_kp = self.kp_right0[left0_right0_match.trainIdx]
        left1_kp = self.kp_left1[left1_right1_match.queryIdx]
        right1_kp = self.kp_right1[left1_right1_match.trainIdx]
        utils.plot_match_on_4_images(self.left0_img, self.right0_img, self.left1_img,
                                     self.right1_img, left0_kp,
                                     right0_kp, left1_kp, right1_kp, title)

    def pnp_ransac(self, p=0.99999999, get_inliers=False, dir_for_plot=""):
        """
        Run pnp ransac algorithm on the matches_idx_dict.
        :param p: desired probability of outlier-free sample
        :param get_inliers: if true return two lists of coordinates, one for the first frame inliers
        and one for the second frame
        :param dir_for_plot: if a non-empty string is given, the function will save the plots of the
         inliers and outliers in the given directory
        :return:
        """

        # return if there are less than 4 matches
        if len(self.matches_idx_dict) < 30 and get_inliers:
            return None, 0.0, None, None

        eps = (len(self.matches_idx_dict) - 4) / len(self.matches_idx_dict)
        n = int(np.log(1 - p) / np.log(1 - (1 - eps) ** 4))
        best_inliers = []
        i = 0
        while i < n:
            # sample 4 matches
            sample = random.sample(self.matches_idx_dict.keys(), 4)
            R_t = self.calculate_pnp(sample)
            if R_t is None:
                continue
            supporters, deniers = self.calculate_supporters_and_deniers(R_t)
            if len(supporters) > len(best_inliers):
                best_inliers = supporters
                eps = (len(self.matches_idx_dict) - len(best_inliers)) / len(self.matches_idx_dict)
                n = int(np.log(1 - p) / np.log(1 - (1 - eps) ** 4))
            i += 1

        # refine the best R_t
        best_R_t = self.calculate_pnp(best_inliers)
        return_indices = not get_inliers
        supporters, deniers = self.calculate_supporters_and_deniers(best_R_t, return_indices=return_indices)
        if dir_for_plot:
            utils.plot_supporters_and_deniers(self.left0_img, self.left1_img, supporters, deniers,
                                              dir_for_plot)
        if self.track_db:
            if len(supporters) > 0:
                assert type(supporters[
                                0]) == int  # make sure that supporter are indices and not measurements
            self.fill_track_db(supporters, best_R_t)
            self.track_db.inliers_outliers_count.append((len(best_inliers), len(deniers)))
        inliers_percent = np.round(len(best_inliers) / len(self.matches_idx_dict) * 100)
        if get_inliers:  # return the inliers (xl, xr, y) for every index in inliers
            inliers_xl_xr_y_frame0 = []
            inliers_xl_xr_y_frame1 = []
            for idx in best_inliers:
                left0_right0_match_idx, left1_right1_match_idx = self.matches_idx_dict[idx]
                left0_right0_match = self.left0_right0_matches[left0_right0_match_idx]
                left1_right1_match = self.left1_right1_matches[left1_right1_match_idx]
                left0_kp = self.kp_left0[left0_right0_match.queryIdx]
                right0_kp = self.kp_right0[left0_right0_match.trainIdx]
                left1_kp = self.kp_left1[left1_right1_match.queryIdx]
                right1_kp = self.kp_right1[left1_right1_match.trainIdx]
                inliers_xl_xr_y_frame0.append((left0_kp.pt[0], right0_kp.pt[0], left0_kp.pt[1]))
                inliers_xl_xr_y_frame1.append((left1_kp.pt[0], right1_kp.pt[0], left1_kp.pt[1]))
            return best_R_t, inliers_percent, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1
        return best_R_t, inliers_percent

    def get_frame_1_kp_desc_matches(self):
        """
        :return: keypoints, descriptors and matches of the left1 and right1 images
        """
        return self.kp_left1, self.desc_left1, self.kp_right1, self.desc_right1,\
               self.left1_right1_matches


def consensus_matching(detector, matcher, frame0_id, frame1_id, left0_extrinsic,
                       right0_extrinsic, k, track_db=None, prev_desc_and_matches=None,
                       get_desc_and_matches=False, get_inliers=False, plot_target_dir=""):
    """
    Run the consensus matching algorithm on two frames.
    :param detector: the detector to use
    :param matcher: the matcher to use
    :param frame0_id: the id of the first frame
    :param frame1_id: the id of the second frame
    :param left0_extrinsic: the extrinsic matrix of the first frame
    :param right0_extrinsic: the extrinsic matrix of the second frame
    :param k: the intrinsic matrix
    :param track_db: the track db where the tracks are saved, if None then the tracks are not saved
    :param prev_desc_and_matches: the descriptors and matches of the previous frame, if None then
    the descriptors and matches are calculated from the images
    :param get_desc_and_matches: if true return the descriptors and matches of the current frame
    :param get_inliers: if true return the inliers of the current frame (xl, xr, y) for every index
    in addition to the R_t and the inliers percent
    :param plot_target_dir: if a non empty string then plot the supporters and deniers of the
    consensus matching algorithm in the given directory
    :return: R_t, inliers_percent, if get_desc_and_matches is true then also return the descriptors
    and matches of the current frame, if get_inliers is true then also return the inliers (xl, xr, y)
    """
    left0_img, right0_img = read_images(frame0_id)
    left1_img, right1_img = read_images(frame1_id)
    if prev_desc_and_matches is None:
        kp_left0 = None
        desc_left0 = None
        kp_right0 = None
        desc_right0 = None,
        left0_right0_matches = None
    else:
        kp_left0, desc_left0, kp_right0, desc_right0, left0_right0_matches = prev_desc_and_matches
    blur_factor = 10
    if plot_target_dir == "":
        left0_img = cv2.blur(left0_img, (blur_factor, blur_factor))
        right0_img = cv2.blur(right0_img, (blur_factor, blur_factor))
        left1_img = cv2.blur(left1_img, (blur_factor, blur_factor))
        right1_img = cv2.blur(right1_img, (blur_factor, blur_factor))

    localizer = Localizer(frame0_id, detector, matcher, left0_img, right0_img, left1_img, right1_img
                          , left0_extrinsic, right0_extrinsic, k, track_db, kp_left0, desc_left0,
                          kp_right0, desc_right0, left0_right0_matches)
    if get_desc_and_matches:
        next_prev_desc_and_matches = localizer.get_frame_1_kp_desc_matches()
    else:
        next_prev_desc_and_matches = None

    pnp_result = localizer.pnp_ransac(
        get_inliers=get_inliers, dir_for_plot=plot_target_dir)

    if get_inliers:
        R_t, inliers_percent, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1 = pnp_result
        return R_t, len(localizer.matches_idx_dict), inliers_percent, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1, \
               next_prev_desc_and_matches
    else:
        R_t, inliers_percent = pnp_result
        return R_t, inliers_percent, next_prev_desc_and_matches
