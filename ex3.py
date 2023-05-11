import time
import utils
import updated_feature_tracking as aft
import updated_geometric_rejection_and_triangulation as agt
import cv2
import numpy as np
import itertools

import matplotlib.pyplot as plt
import random
random.seed(0) # for reproducibility
triangulate_func = agt.triangulate_point
triangulate_func_multiple = agt.triangulate_multiple_points

class Localizer():
    """
    Class that localizes a camera in the next stereo pair, given the previous stereo pair and the previous camera pose.
    """
    def __init__(self, detector, matcher, left0, right0, left1, right1, left0_camera, right0_camera, intrinsic_matrix, left0_right0_matches=None):
        self.basic_tracker = aft.RatioTestTracker(detector, matcher, ratio=0.5)
        self.geometric_rejection_tracker = agt.XY_GeometricRejectionTracker(detector, matcher)
        self.left0 = left0
        self.right0 = right0
        self.left1 = left1
        self.right1 = right1
        self.left0_camera = left0_camera
        self.right0_camera = right0_camera
        self.intrinsic_matrix = intrinsic_matrix
        self.kp_left0, self.desc_left0 = self.basic_tracker.calculate_kps_and_descs(left0)
        self.kp_right0, self.desc_right0 = self.basic_tracker.calculate_kps_and_descs(right0)
        self.kp_left1, self.desc_left1 = self.basic_tracker.calculate_kps_and_descs(left1)
        self.kp_right1, self.desc_right1 = self.basic_tracker.calculate_kps_and_descs(right1)
        self.left0_right0_matches = left0_right0_matches
        if left0_right0_matches is None:
            self.left0_right0_matches = self.geometric_rejection_tracker.calculate_matches(self.desc_left0, self.desc_right0, kp1=self.kp_left0, kp2=self.kp_right0)[0]
        self.left1_right1_matches = self.geometric_rejection_tracker.calculate_matches(self.desc_left1, self.desc_right1, kp1=self.kp_left1, kp2=self.kp_right1)[0]
        self.left0_left1_matches = self.basic_tracker.calculate_matches(self.desc_left0, self.desc_left1)[0]

        self.matches_idx_dict = {} # key: left0_left1_match_idx, value: (left0_right0_match_idx, left1_right1_match_idx)
        self.create_matches_dict()

    def create_matches_dict(self):
        """
        create a dictionary that maps matches from left0_left1 to left0_right0 and left1_right1.
        """
        left0_right0_dict = {m.queryIdx: i for i,m in enumerate(self.left0_right0_matches)}
        left1_right1_dict = {m.queryIdx: i for i,m in enumerate(self.left1_right1_matches)}
        matches_dict = {}
        for m_idx in range(len(self.left0_left1_matches)):
            m = self.left0_left1_matches[m_idx]
            if m.queryIdx in left0_right0_dict.keys() and m.trainIdx in left1_right1_dict.keys():
                matches_dict[m_idx] = (left0_right0_dict[m.queryIdx], left1_right1_dict[m.trainIdx])
        self.matches_idx_dict = matches_dict

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
        image_points = [] # 2d points on left1 image plane
        world_points = [] # 3d points calculated by triangulation process of left0, right0.
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
        success, rvec, tvec = cv2.solvePnP(world_points, image_points, self.intrinsic_matrix, None, flags=solver)
        if success:
            rot, _ = cv2.Rodrigues(rvec)
            return np.hstack((rot, tvec))


    def calculate_supporters_and_deniers(self, R_t, threshold=2):
        """
        calculate the supporters and deniers of the given R_t matrix.
        :param R_t: 3x4 matrix, which transforms the previous coordinate system to the next coordinate system.
        :param threshold: threshold for the geometric distance between the 2d and 3d points.
        :return: supporters, deniers
        """
        supporters = []
        deniers = []
        for m_idx in self.matches_idx_dict.keys():
            left0_right0_match_idx, left1_right1_match_idx = self.matches_idx_dict[m_idx]
            left0_right0_match = self.left0_right0_matches[left0_right0_match_idx]
            left1_right1_match = self.left1_right1_matches[left1_right1_match_idx]
            p3d = self.triangulate(left0_right0_match.queryIdx, left0_right0_match.trainIdx)
            p2d_projected = project(p3d, self.intrinsic_matrix @ R_t)
            p2d = self.kp_left1[left1_right1_match.queryIdx].pt
            if np.linalg.norm(p2d_projected - p2d) < threshold:
                supporters.append(m_idx)
            else:
                deniers.append(m_idx)
        return supporters, deniers

    def calculate_supporters_and_deniers_fast(self, R_t, threshold=1):
        """
        calculate the supporters and deniers of the given R_t matrix, but doing it with vectorization instead of loops.
        """

        left0_kp_idx = [self.left0_left1_matches[m_idx].queryIdx for m_idx in self.matches_idx_dict.keys()]
        right0_kp_idx = [self.left0_right0_matches[self.matches_idx_dict[m_idx][0]].trainIdx for m_idx in self.matches_idx_dict.keys()]
        left1_kp_idx = [self.left0_left1_matches[m_idx].trainIdx for m_idx in self.matches_idx_dict.keys()]
        p3d = self.triangulate_multiple_l0_r0(left0_kp_idx, right0_kp_idx)
        p2d_projected = project_points(p3d, self.intrinsic_matrix @ R_t).T
        p2d = np.array([self.kp_left1[kp_idx].pt for kp_idx in left1_kp_idx])
        supporters = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) < threshold)[0]
        deniers = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) >= threshold)[0]
        # convert the supporters and deniers to the original indices
        supporters = [list(self.matches_idx_dict.keys())[i] for i in supporters]
        deniers = [list(self.matches_idx_dict.keys())[i] for i in deniers]
        return supporters, deniers # TODO - this return value might be wrong

    def get_l0l1_kps_from_match_idx(self, idx):
        """
        get the keypoints in left0 and left1 images, given the index of the match between them.
        :param idx: index of the match between left0 and left1 images.
        :return: keypoints in left0 and left1 images.
        """
        left0_kp_idx, left1_kp_idx = self.left0_left1_matches[idx].queryIdx, self.left0_left1_matches[idx].trainIdx
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
        p4d = triangulate_func(self.left0_camera, self.right0_camera, p, q)
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
        p4d = triangulate_func_multiple(self.left0_camera, self.right0_camera, p, q)
        p3d = p4d[:3] / p4d[3]
        return p3d

    def triangulate_multiple_l1_r1(self, left1_kp_idxs, right1_kp_idxs, left1_extrinsic_matrix, right1_extrinsic_matrix):
        """
        triangulate multiple 3d points from the left1 and right1 images.
        :param left1_kp_idxs: list of indices of the keypoints in left1 image
        :param right1_kp_idxs: list of indices of the keypoints in right1 image
        :return: list of 3d points
        """
        p = np.array([self.kp_left1[idx].pt for idx in left1_kp_idxs])
        q = np.array([self.kp_right1[idx].pt for idx in right1_kp_idxs])
        p4d = triangulate_func_multiple(left1_extrinsic_matrix, right1_extrinsic_matrix, p, q)
        p3d = p4d[:3] / p4d[3]
        return p3d

    def triangulate_all_l0_r0(self):
        """
        triangulate all the 3d points from the left0 and right0 images.
        :return: list of 3d points
        """
        left0_kp_idxs = [self.left0_right0_matches[m_idx[0]].queryIdx for m_idx in self.matches_idx_dict.values()]
        right0_kp_idxs = [self.left0_right0_matches[m_idx[0]].trainIdx for m_idx in self.matches_idx_dict.values()]
        return self.triangulate_multiple_l0_r0(left0_kp_idxs, right0_kp_idxs)

    def triangulate_all_l1_r1(self, left1_extrinsic_matrix, right1_extrinsic_matrix):
        """
        triangulate all the 3d points from the left1 and right1 images.
        :return: list of 3d points
        """
        left1_kp_idxs = [self.left1_right1_matches[m_idx[1]].queryIdx for m_idx in self.matches_idx_dict.values()]
        right1_kp_idxs = [self.left1_right1_matches[m_idx[1]].trainIdx for m_idx in self.matches_idx_dict.values()]
        return self.triangulate_multiple_l1_r1(left1_kp_idxs, right1_kp_idxs,left1_extrinsic_matrix, right1_extrinsic_matrix)

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
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(self.left0, cmap='gray')
        ax[0, 0].scatter(left0_kp.pt[0], left0_kp.pt[1], c='r', s=3)
        ax[0, 1].imshow(self.right0, cmap='gray')
        ax[0, 1].scatter(right0_kp.pt[0], right0_kp.pt[1], c='r', s=3)
        ax[1, 0].imshow(self.left1, cmap='gray')
        ax[1, 0].scatter(left1_kp.pt[0], left1_kp.pt[1], c='r', s=3)
        ax[1, 1].imshow(self.right1, cmap='gray')
        ax[1, 1].scatter(right1_kp.pt[0], right1_kp.pt[1], c='r', s=3)
        plt.suptitle(title)
        plt.show()

    def pnp_ransac(self, p=0.99999999):
        eps = (len(self.matches_idx_dict) - 4)/len(self.matches_idx_dict)
        n = int(np.log(1-p)/np.log(1-(1-eps)**4))
        best_inliers = []
        i = 0
        while i < n:
            # sample 4 matches
            sample = random.sample(self.matches_idx_dict.keys(), 4)
            R_t = self.calculate_pnp(sample)
            if R_t is None:
                continue
            supporters, deniers = self.calculate_supporters_and_deniers_fast(R_t)
            if len(supporters) > len(best_inliers):
                best_inliers = supporters
                eps = (len(self.matches_idx_dict) - len(best_inliers))/len(self.matches_idx_dict)
                n = int(np.log(1-p)/np.log(1-(1-eps)**4))
            i += 1

        print(f'the number of ransac iterations is {i}, the number of matches is {len(self.matches_idx_dict)} '
              f'and the number of inliesrs is {len(best_inliers)}')
        # refine the best R_t
        best_R_t = self.calculate_pnp(best_inliers)
        return best_R_t




def camera_location_from_extrinsic_matrix(R_t):
    """
    The input is n+1 camera matrix [R|t], which transforms the n'th coordinate system to the
    n+1 coordinate system. The camera location (in the n'th coordinate system) is the negative of the inverse (=transpose)
    of R,  as seen in question 2.3.
    :param R_t: 3x4 matrix
    :return: camera location in world coordinate, according to the n'th coordinate system.
    """
    return (-R_t[:3, :3].T).dot(R_t[:3, 3])

def compose_affine_transformations(outer_affine, inner_affine):
    """
    compute the composition of the affine transformation represented by camera1 and camera2. The result transform takes
    first transforms to the coordinate system of camera1 and then transforms to the coordinate system of camera2.
    :param outer_affine: 3x4 matrix
    :param inner_affine: 3x4 matrix
    :return: 3x4 matrix
    """
    # we use the fact that [R2|t2] @ append_row([R1|t1], e4) = [R2@R1|t2+R2@t1]
    temp_mat = np.append(inner_affine, np.array([[0, 0, 0, 1]]), axis=0)
    return outer_affine @ temp_mat

def project(p3d, camera):
    """
    project the 3d point to the image plane.
    :param p3d: 3d point
    :param camera: 3x4 matrix (K @ [R|t])
    :return: 2d point
    """
    p3d = np.append(p3d, 1)
    p2d = camera @ p3d
    p2d = p2d[:2] / p2d[2]
    return p2d

def project_points(points, camera):
    """
    project the 3d points to the image plane.
    :param points: 3xn matrix
    :param camera: 3x4 matrix (K @ [R|t])
    :return: 2xn matrix
    """
    ones_row = np.ones((1, points.shape[1]))
    points = np.append(points, ones_row, axis=0)
    points = camera @ points
    points = points[:2] / points[2]
    return points

def run_3_3():
    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left0_camera = k@left0_extrinsic
    right0_camera = k@right0_extrinsic
    left0, right0 = utils.read_images(0)
    left1, right1 = utils.read_images(1)
    localizer = Localizer(feature_detector, bf, left0, right0, left1, right1, left0_camera, right0_camera, k)
    # Although we were asked to use ransac only on a later stage, I also used it here to get a sensible result:
    R_t = localizer.pnp_ransac()
    left1_extrinsic = compose_affine_transformations(R_t, left0_extrinsic)
    right1_extrinsic = compose_affine_transformations(right0_extrinsic, left1_extrinsic)
    left0_pose = camera_location_from_extrinsic_matrix(left0_extrinsic)
    left1_pose = camera_location_from_extrinsic_matrix(left1_extrinsic)
    right0_pose = camera_location_from_extrinsic_matrix(right0_extrinsic)
    right1_pose = camera_location_from_extrinsic_matrix(right1_extrinsic)
    utils.plot_left_right_camera_position([left0_pose, left1_pose], [right0_pose, right1_pose])

def run_3_4():

    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left0_camera = k @ left0_extrinsic
    right0_camera = k @ right0_extrinsic
    left0, right0 = utils.read_images(0)
    left1, right1 = utils.read_images(1)
    localizer = Localizer(feature_detector, bf, left0, right0, left1, right1, left0_camera, right0_camera, k)
    sample = random.sample(localizer.matches_idx_dict.keys(), 4)
    R_t = localizer.calculate_pnp(sample)
    supporters_match_idx, deniers_match_idx = localizer.calculate_supporters_and_deniers_fast(R_t, threshold=2)
    supporters = []
    deniers = []
    for smi in supporters_match_idx:
        match = localizer.left0_left1_matches[smi]
        supporters.append((localizer.kp_left0[match.queryIdx], localizer.kp_left1[match.trainIdx]))
    for dmi in deniers_match_idx:
        match = localizer.left0_left1_matches[dmi]
        deniers.append((localizer.kp_left0[match.queryIdx], localizer.kp_left1[match.trainIdx]))
    utils.plot_supporters_and_deniers(left0, left1, supporters, deniers)

def run_3_5():

    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left0_camera = k @ left0_extrinsic
    right0_camera = k @ right0_extrinsic
    left0, right0 = utils.read_images(0)
    left1, right1 = utils.read_images(1)
    localizer = Localizer(feature_detector, bf, left0, right0, left1, right1, left0_camera, right0_camera, k)
    R_t = localizer.pnp_ransac()
    p3d_l0_r0 = localizer.triangulate_all_l0_r0()
    p3d_l0_r0 = np.append(p3d_l0_r0, np.ones((1, p3d_l0_r0.shape[1])), axis=0)
    p3d_l0_r0 = R_t @ p3d_l0_r0
    p3d_l1_r1 = localizer.triangulate_all_l1_r1(left0_camera, right0_camera)
    close_points_num = np.sum(np.linalg.norm(p3d_l0_r0 - p3d_l1_r1, axis=0) < 1)
    print(f"number of points that are close in both triangulations: {close_points_num}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p3d_l0_r0[0], p3d_l0_r0[1], p3d_l0_r0[2], c='b', marker='o')
    ax.scatter(p3d_l1_r1[0], p3d_l1_r1[1], p3d_l1_r1[2], c='r', marker='o')
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # add title:
    ax.set_title("3D points of pair 0 after transformation in blue, and the 3D points of pair 1 in red")
    plt.show()
    supporters_match_idx, deniers_match_idx = localizer.calculate_supporters_and_deniers_fast(R_t, threshold=2)
    supporters = []
    deniers = []
    for smi in supporters_match_idx:
        match = localizer.left0_left1_matches[smi]
        supporters.append((localizer.kp_left0[match.queryIdx], localizer.kp_left1[match.trainIdx]))
    for dmi in deniers_match_idx:
        match = localizer.left0_left1_matches[dmi]
        deniers.append((localizer.kp_left0[match.queryIdx], localizer.kp_left1[match.trainIdx]))
    utils.plot_supporters_and_deniers(left0, left1, supporters, deniers)

def run_3_6():
    ITERATIONS = 2559
    feature_detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()

    left0_camera = k @ left0_extrinsic
    right0_camera = k @ right0_extrinsic

    gt_left1_extrinsics = utils.get_gt_left_camera_matrices(ITERATIONS + 1)

    first_gt_location = camera_location_from_extrinsic_matrix(gt_left1_extrinsics[0])
    gt_locations = [first_gt_location]
    computed_matrices_global = [left0_extrinsic]
    computed_locations = [first_gt_location]

    start = time.time()
    for i in range(ITERATIONS):
        print("iteration {}".format(i))
        left0, right0 = utils.read_images(i)
        left1, right1 = utils.read_images(i + 1)
        # blur the images for better feature detection
        left0 = cv2.GaussianBlur(left0, (9, 9), 0)
        right0 = cv2.GaussianBlur(right0, (9, 9), 0)
        left1 = cv2.GaussianBlur(left1, (9, 9), 0)
        right1 = cv2.GaussianBlur(right1, (9, 9), 0)
        localizer = Localizer(feature_detector, bf, left0, right0, left1, right1, left0_camera, right0_camera, k)
        computed_left1_extrinsic = localizer.pnp_ransac()
        computed_matrices_global.append(
            compose_affine_transformations(computed_left1_extrinsic, computed_matrices_global[-1]))
        computed_location = camera_location_from_extrinsic_matrix(computed_matrices_global[-1])
        computed_locations.append(computed_location)

        gt_left1_extrinsic = gt_left1_extrinsics[i + 1]
        gt_location = camera_location_from_extrinsic_matrix(np.array(gt_left1_extrinsic))
        gt_locations.append(gt_location)

    end = time.time()
    print("time: {}".format(end - start))
    gt_locations = np.array(gt_locations)
    computed_locations = np.array(computed_locations)
    plt.plot(gt_locations[:, 0], gt_locations[:, 2], label='gt')
    plt.plot(computed_locations[:, 0], computed_locations[:, 2], label='computed')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # run_3_1()
    # run_3_2()
    # run_3_3()
    # run_3_4()
    #run_3_5()
    run_3_6()




