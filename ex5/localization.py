import time
import utils
import updated_feature_tracking as aft
import updated_geometric_rejection_and_triangulation as agt
import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import random

random.seed(0)  # for reproducibility
triangulate_func = agt.triangulate_point
triangulate_func_multiple = agt.triangulate_multiple_points


class Track:
    def __init__(self, first_frame_id, first_frame_location_left: tuple[int, int],
                 first_frame_location_right: tuple[int, int]):
        """
        :param track_id: tuple of (frame_id, x, y). frame_id is the first frame the track appears in. x, y are the pixel
        coordinates of the track in the first left image.
        """
        self.first_frame_id = first_frame_id
        self.locations = [(first_frame_location_left,
                           first_frame_location_right)]  # list of ((x_l, y_l),(x_r, y_r)) pixel coordinates of the
        # track in the left and right images.

    def __id__(self):
        return self.first_frame_id, self.locations[0][0]

    def __len__(self):
        return len(self.locations)


    def add_location(self, location_left: tuple[int, int], location_right: tuple[int, int]):
        self.locations.append((location_left, location_right))

    def get_frames(self):
        """
        (function that is required in 4.1, point 4)
        :return: all frames that the track appears in.
        """
        return [self.first_frame_id + i for i in range(len(self.locations))]

    def get_4_frame_ids_and_left_img_location(self):
        """
        :return 4 equally spaced frame ids and location on the left images of the track in those frames. The first frame id is the first
        frame the track appears in. The last frame id is the last frame the track appears in.
        """
        frame_ids = [self.first_frame_id]
        locations = [self.locations[0][0]]
        if len(self.locations) == 1:
            return frame_ids, locations
        frame_ids.append(self.first_frame_id + (len(self.locations) - 1) // 3)
        locations.append(self.locations[(len(self.locations) - 1) // 3][0])
        frame_ids.append(self.first_frame_id + (len(self.locations) - 1) // 3 * 2)
        locations.append(self.locations[(len(self.locations) - 1) // 3 * 2][0])
        frame_ids.append(self.first_frame_id + len(self.locations) - 1)
        locations.append(self.locations[-1][0])
        return frame_ids, locations

    def get_consecutive_frame_ids_and_locations(self, num_frames):
        assert len(self.locations) >= num_frames
        frame_ids = [self.first_frame_id]
        locations = [self.locations[0]]
        for i in range(1, num_frames):
            frame_ids.append(self.first_frame_id + i)
            locations.append(self.locations[i])

        return frame_ids, locations


class TrackDB:
    def __init__(self):
        self.relative_R_t_list = []  # list of relative R_t
        self.global_R_t_list = []  # list of global R_t
        self.frameId_track_dict = {}  # value: dict of track_id: track
        # define two dictionaries according to which we start a new track / continue an existing track.
        self.last_added_tracks = {}  # key: x, y coordinates of the track in the last frame it appears in, value: track
        self.next_added_tracks = {}  # key: x, y coordinates of the track in the next frame it appears in, value: track
        self.inliers_outliers_count = [] # key: frame id, value: tuple of (inliers, outliers) count

    def get_relative_R_t(self, frame_id):
        """
        :param frame_id: frame id
        :return: relative R_t of the frame
        """
        return self.relative_R_t_list[frame_id]

    def get_global_R_t(self, frame_id):
        """
        :param frame_id: frame id
        :return: global R_t of the frame
        """
        return self.global_R_t_list[frame_id]

    def get_frame_tracks(self, frame_id):
        """
        :param frame_id: frame id
        :return: list of track ids that appear in the frame.
        """
        return self.frameId_track_dict[frame_id].values()

    def get_track_frames(self, track_id):
        """
        :param track_id: track id
        :return: list of frames that the track appears in.
        """
        frame = track_id[0]
        return self.frameId_track_dict[frame][track_id].get_frames()

    def add_match(self, l0_id: int, x_l0: int, y_l0: int, x_r0: int, y_r0: int,
                  l1_id: int, x_l1: int, y_l1: int, x_r1: int, y_r1: int):
        """
        :param l0_id: left0 track id
        :param l1_id: left1 track id
        :param x_l0: x coordinate of the track in left0
        :param y_l0: y coordinate of the track in left0
        :param x_r0: x coordinate of the track in right0
        :param y_r0: y coordinate of the track in right0
        :param x_l1: x coordinate of the track in left1
        :param y_l1: y coordinate of the track in left1
        :param x_r1: x coordinate of the track in right1
        :param y_r1: y coordinate of the track in right1
        """
        if l0_id not in self.frameId_track_dict.keys():
            self.frameId_track_dict[l0_id] = {}
        if l1_id not in self.frameId_track_dict.keys():
            self.frameId_track_dict[l1_id] = {}

        # check if the track already exists in the frame
        if (x_l0, y_l0) in self.last_added_tracks.keys():
            track = self.last_added_tracks[(x_l0, y_l0)]
        else:
            track = Track(l0_id, (x_l0, y_l0), (x_r0, y_r0))
            self.frameId_track_dict[l0_id][track.__id__()] = track

        track.add_location((x_l1, y_l1), (x_r1, y_r1))
        self.frameId_track_dict[l1_id][track.__id__()] = track
        self.next_added_tracks[(x_l1, y_l1)] = track

    def reset_last_added_tracks(self):
        """
        Restarts the two dictionaries that are used to determine if a track already exists in a frame.
        Should be used after each frame.
        :return:
        """
        self.last_added_tracks = self.next_added_tracks
        self.next_added_tracks = {}

    def get_location(self, frame_id, track_id):
        """
        :param frame_id: frame id
        :param track_id: track id
        :return: location of the track in the frame.
        """
        track = self.frameId_track_dict[frame_id][track_id]
        xl, y = track.locations[frame_id - track.first_frame_id][0]
        xr, _ = track.locations[frame_id - track.first_frame_id][1]
        return xl, xr, y

    def get_all_unique_tracks(self):
        """
        :return: list of all tracks in the track db.
        """
        tracks = []
        for track_dict in self.frameId_track_dict.values():
            for track in track_dict.values():
                tracks.append(track)

        return set(tracks)

    @staticmethod
    def serialize(track_db, filename):
        """
        :return: list of all tracks in the track db.
        """
        with open(filename, 'wb') as f:
            pickle.dump([track_db.frameId_track_dict,
                         track_db.inliers_outliers_count,
                         track_db.relative_R_t_list,
                         track_db.global_R_t_list], f)

    @staticmethod
    def deserialize(filename):
        track_db = TrackDB()
        with open(filename, 'rb') as f:
            loaded_list = pickle.load(f)
            track_db.frameId_track_dict = loaded_list[0]
            track_db.inliers_outliers_count = loaded_list[1]
            track_db.relative_R_t_list = loaded_list[2]
            track_db.global_R_t_list = loaded_list[3]

        return track_db

    def get_statistics(self):
        """
        Return the following statistics for:
        1. Number of tracks
        2. Number of frames
        3. Mean track length, maximum track length, minimum track length
        4. Mean number of frame links (number of tracks on an average frame)
        :return:
        """
        all_tracks = self.get_all_unique_tracks()
        num_tracks = len(all_tracks)
        mean_track_length = np.mean([len(track.locations) for track in all_tracks])
        max_track_length = np.max([len(track.locations) for track in all_tracks])
        min_track_length = np.min([len(track.locations) for track in all_tracks])
        num_frames = len(self.frameId_track_dict.keys())
        mean_num_frame_links = np.mean([len(self.frameId_track_dict[k]) for k in self.frameId_track_dict.keys()])
        return num_tracks, num_frames, mean_track_length, max_track_length, min_track_length, mean_num_frame_links

    def get_longest_track(self):
        all_tracks = self.get_all_unique_tracks()
        max_track_length = np.max([len(track.locations) for track in all_tracks])
        for track in all_tracks:
            if len(track.locations) == max_track_length:
                return track

    def get_frame_connectivity(self, frame_id):
        """
        :param frame_id: frame id
        :return: number of tracks in the frame
        """
        track_dict = self.frameId_track_dict[frame_id]
        count = 0
        for track in track_dict.values():
            track_first_frame_id = track.first_frame_id
            if track_first_frame_id + len(track) - 1  > frame_id:
                count += 1

        return count

    def get_track_with_len_at_least_10(self):
        all_tracks = self.get_all_unique_tracks()
        for track in all_tracks:
            if len(track.locations) >= 10:
                return track

    def calc_global_R_t(self):
        """
        Fill the global R_t list by composing the matrices from the relative R_t list.
        :return:
        """
        for i in range(len(self.relative_R_t_list)):
            if i == 0:
                self.global_R_t_list.append(self.relative_R_t_list[i])
            else:
                self.global_R_t_list.append(compose_affine_transformations(self.relative_R_t_list[i],
                                                                            self.global_R_t_list[i - 1]))

    def get_number_of_frames(self):
        return len(self.frameId_track_dict.keys())

    def get_median_track_length_from_frame(self, frame):
        """
        return the median track length, counting from the given frame for all frame links.
        """
        histogram = []
        for t in self.get_frame_tracks(frame):
            histogram.append(len(t) - (frame - t.first_frame_id))
        return np.ceil(np.percentile(histogram, q=70)).astype(int)



class Localizer:
    """
    Class that localizes a camera in the next stereo pair, given the previous stereo pair and the previous camera pose.
    """

    def __init__(self, frame0_index, track_db, detector, matcher, left0, right0, left1, right1, left0_ex_camera,
                 right0_ex_camera, intrinsic_matrix,
                 kp_left0=None, desc_left0=None, kp_right0=None, desc_right0=None, left0_right0_matches=None):
        self.frame0_index = frame0_index
        self.track_db = track_db
        self.basic_tracker = aft.RatioTestTracker(detector, matcher, ratio=0.6)
        self.geometric_rejection_tracker = agt.XY_GeometricRejectionTracker(detector, matcher)
        self.left0 = left0
        self.right0 = right0
        self.left1 = left1
        self.right1 = right1
        self.left0_ex_camera = left0_ex_camera
        self.right0_ex_camera = right0_ex_camera
        self.intrinsic_matrix = intrinsic_matrix
        self.left0_camera = intrinsic_matrix @ left0_ex_camera
        self.right0_camera = intrinsic_matrix @ right0_ex_camera
        self.kp_left1, self.desc_left1 = self.basic_tracker.calculate_kps_and_descs(left1)
        self.kp_right1, self.desc_right1 = self.basic_tracker.calculate_kps_and_descs(right1)
        self.kp_left0, self.desc_left0 = kp_left0, desc_left0
        self.kp_right0, self.desc_right0 = kp_right0, desc_right0
        self.kp_right0, self.desc_right0 = self.basic_tracker.calculate_kps_and_descs(right0)
        self.left0_right0_matches = left0_right0_matches
        if left0_right0_matches is None:
            self.kp_left0, self.desc_left0 = self.basic_tracker.calculate_kps_and_descs(left0)
            self.kp_right0, self.desc_right0 = self.basic_tracker.calculate_kps_and_descs(right0)
            self.left0_right0_matches = \
                self.geometric_rejection_tracker.calculate_matches(self.desc_left0, self.desc_right0, kp1=self.kp_left0,
                                                                   kp2=self.kp_right0)[0]
        self.left1_right1_matches = \
            self.geometric_rejection_tracker.calculate_matches(self.desc_left1, self.desc_right1, kp1=self.kp_left1,
                                                               kp2=self.kp_right1)[0]
        self.left0_left1_matches = self.basic_tracker.calculate_matches(self.desc_left0, self.desc_left1)[0]

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
        right0_kp_idx = [self.left0_right0_matches[self.matches_idx_dict[m_idx][0]].trainIdx for m_idx in
                         self.matches_idx_dict.keys()]
        left1_kp_idx = [self.left0_left1_matches[m_idx].trainIdx for m_idx in self.matches_idx_dict.keys()]
        p3d = self.triangulate_multiple_l0_r0(left0_kp_idx, right0_kp_idx)
        p2d_projected = project_points(p3d, self.intrinsic_matrix @ R_t).T
        p2d = np.array([self.kp_left1[kp_idx].pt for kp_idx in left1_kp_idx])
        supporters = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) < threshold)[0]
        deniers = np.where(np.linalg.norm(p2d_projected - p2d, axis=1) >= threshold)[0]
        # convert the supporters and deniers to the original indices
        supporters = [list(self.matches_idx_dict.keys())[i] for i in supporters]
        deniers = [list(self.matches_idx_dict.keys())[i] for i in deniers]
        return supporters, deniers  # TODO - this return value might be wrong

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
        return self.triangulate_multiple_l1_r1(left1_kp_idxs, right1_kp_idxs, left1_extrinsic_matrix,
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
            supporters, deniers = self.calculate_supporters_and_deniers_fast(R_t)
            if len(supporters) > len(best_inliers):
                best_inliers = supporters
                eps = (len(self.matches_idx_dict) - len(best_inliers)) / len(self.matches_idx_dict)
                n = int(np.log(1 - p) / np.log(1 - (1 - eps) ** 4))
            i += 1

        print(f'the number of ransac iterations is {i}, the number of matches is {len(self.matches_idx_dict)} '
              f'and the number of inliesrs is {len(best_inliers) } ({np.round(len(best_inliers) / len(self.matches_idx_dict) * 100)}%)')
        # refine the best R_t
        best_R_t = self.calculate_pnp(best_inliers)
        supporters, deniers = self.calculate_supporters_and_deniers_fast(best_R_t)
        self.fill_track_db(supporters, best_R_t)
        self.track_db.inliers_outliers_count.append((len(best_inliers), len(deniers)))
        return best_R_t

    def get_frame_1_kp_desc_matches(self):
        """
        :return: keypoints, descriptors and matches of the left1 and right1 images
        """
        return self.kp_left1, self.desc_left1, self.kp_right1, self.desc_right1, self.left1_right1_matches


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


def create_track_db(track_db_path):
    ITERATIONS = 2559
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()


    gt_left1_extrinsics = utils.get_gt_left_camera_matrices(ITERATIONS + 1)

    first_gt_location = camera_location_from_extrinsic_matrix(gt_left1_extrinsics[0])
    gt_locations = [first_gt_location]
    computed_matrices_global = [left0_extrinsic]
    computed_locations = [first_gt_location]
    prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches = None, None, None, None, None
    start = time.time()
    track_db = TrackDB()

    for i in range(ITERATIONS):
        print("iteration {}".format(i))
        left0, right0 = utils.read_images(i)
        left1, right1 = utils.read_images(i + 1)
        # blur the images for better feature detection
        blur_factor = 10
        left0 = cv2.blur(left0, (blur_factor, blur_factor))
        right0 = cv2.blur(right0, (blur_factor, blur_factor))
        left1 = cv2.blur(left1, (blur_factor, blur_factor))
        right1 = cv2.blur(right1, (blur_factor, blur_factor))
        localizer = Localizer(i, track_db, feature_detector, bf, left0, right0, left1, right1, left0_extrinsic,
                              right0_extrinsic, k,
                              prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches)
        prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches = localizer.get_frame_1_kp_desc_matches()
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
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(gt_locations[:, 0], gt_locations[:, 2], label='gt')
    ax.plot(computed_locations[:, 0], computed_locations[:, 2], label='computed')
    plt.legend()
    plt.show()
    plt.clf()
    track_db.calc_global_R_t()
    TrackDB.serialize(track_db, track_db_path)

def triangulate_last_frame(track):
    """
    triangulate the features in the last frame of the track using the ground truth camera matrices of this frame.
    :param track: Track object
    :return: a 3d point
    """
    x_l, y_l = track.locations[-1][0]
    x_r, y_r = track.locations[-1][1]
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    frame_id = track.first_frame_id + len(track.locations) - 1
    left_camera_extrinsic_matrix = utils.get_gt_left_camera_matrices(2559)[frame_id]
    right_camera_extrinsic_matrix = compose_affine_transformations(right0_extrinsic, left_camera_extrinsic_matrix)
    point = agt.triangulate_point(k@left_camera_extrinsic_matrix, k@right_camera_extrinsic_matrix, (x_l, y_l), (x_r, y_r))
    return point

def triangulate_first_frame(track):
    x_l, y_l = track.locations[0][0]
    x_r, y_r = track.locations[0][1]
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    frame_id = track.first_frame_id
    left_camera_extrinsic_matrix = utils.get_gt_left_camera_matrices(2559)[frame_id]
    right_camera_extrinsic_matrix = compose_affine_transformations(right0_extrinsic, left_camera_extrinsic_matrix)
    point = agt.triangulate_point(k @ left_camera_extrinsic_matrix, k @ right_camera_extrinsic_matrix, (x_l, y_l),
                                  (x_r, y_r))
    return point




def run_4_2(db_path = 'track_db_akaze.pkl'):
    try:
        track_db = TrackDB.deserialize(db_path)
    except (FileNotFoundError, IndexError):
        create_track_db(db_path)
        track_db = TrackDB.deserialize(db_path)

    stats = track_db.get_statistics()
    print(f"total number of tracks: {stats[0]}"
          f"\nnumber of frames: {stats[1]}"
          f"\nmean track length: {stats[2]}"
          f"\nmax track length: {stats[3]}"
          f"\nmin track length: {stats[4]}"
          f"\nmean number of frame links: {stats[5]}")

def run_4_3(track = None):
    try:
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    except FileNotFoundError:
        create_track_db()
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    if track is None:
        track = track_db.get_longest_track()
    frame_ids, locations = track.get_consecutive_frame_ids_and_locations(len(track))
    utils.plot_consecutive_matches_from_location(frame_ids, locations)

def run_4_4():
    try:
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    except FileNotFoundError:
        create_track_db()
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    connectivities = []
    frames = list(range(2559))
    for frame in frames:
        connectivities.append(track_db.get_frame_connectivity(frame))

    # make the font large
    plt.rcParams.update({'font.size': 22})
    plt.plot(frames, connectivities)
    plt.xlabel('frame')
    plt.ylabel('outgoing tracks')
    plt.title('connectivity')
    plt.xlim(0, 2559)
    plt.yticks(np.arange(0, 700, 50))
    # make the plot wider
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    # draw the mean as a horizontal line
    plt.axhline(y=np.mean(connectivities), color='g', linestyle='-')
    # draw horizontal line at 100
    plt.axhline(y=100, color='r', linestyle='-')
    plt.savefig(utils.EX4_DOCS_PATH+ '/' + '4_4.png')

def run_4_5():
    try:
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    except FileNotFoundError:
        create_track_db()
        track_db = TrackDB.deserialize('track_db_akaze.pkl')

    inliers_percentages = []
    for i in range(len(track_db.inliers_outliers_count)):
        inliers_num = track_db.inliers_outliers_count[i][0]
        outliers_num = track_db.inliers_outliers_count[i][1]
        inliers_percentages.append(inliers_num / (inliers_num + outliers_num))

    # make the font large
    plt.rcParams.update({'font.size': 22})
    plt.plot(inliers_percentages)
    plt.xlabel('frame')
    plt.ylabel('inliers percentage')
    plt.title('inliers percentage')
    plt.xlim(0, 2559)
    plt.ylim(0, 1)
    # make the plot wider
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    # draw the mean as a horizontal line
    plt.axhline(y=np.mean(inliers_percentages), color='g', linestyle='-')
    plt.savefig(utils.EX4_DOCS_PATH+ '/' + '4_5.png')

def run_4_6():
    try:
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    except FileNotFoundError:
        create_track_db()
        track_db = TrackDB.deserialize('track_db_akaze.pkl')

    lengths = []
    for track in track_db.get_all_unique_tracks():
        lengths.append(len(track))

    # make the font large
    plt.rcParams.update({'font.size': 22})
    hist, bins = np.histogram(lengths, bins=range(0, 137, 1))
    plt.plot(bins[:-1], hist)
    plt.xlabel('track length')
    plt.ylabel('number of tracks')
    plt.title('track length histogram')
    plt.xlim(0, 135)
    plt.xticks(np.arange(0, 135, 5))
    # make the plot wider
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(utils.EX4_DOCS_PATH+ '/' + '4_6.png')

def run_4_7():
    try:
        track_db = TrackDB.deserialize('track_db_akaze.pkl')
    except FileNotFoundError:
        create_track_db()
        track_db = TrackDB.deserialize('track_db_akaze.pkl')

    # get a track with length 10 or more:
    track = track_db.get_track_with_len_at_least_10()
    print(f"track length: {len(track)}")
    # calculate the 3d point of the last\first location in the track, according to the ground truth matrices
    point_last = triangulate_last_frame(track)
    point_last = point_last[:3] / point_last[3]
    point_first = triangulate_first_frame(track)
    point_first = point_first[:3] / point_first[3]
    # project the point to all the frames of the track (both left and right cameras)
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left_cameras_gt = utils.get_gt_left_camera_matrices(2559)
    left_projection_errors_last_frame = []
    right_projection_errors_last_frame = []
    left_projection_errors_first_frame = []
    right_projection_errors_first_frame = []
    for i in range(len(track)):
        cur_frame = track.first_frame_id + i
        left_camera_extrinsic_matrix = left_cameras_gt[cur_frame]
        right_camera_extrinsic_matrix = compose_affine_transformations(right0_extrinsic, left_camera_extrinsic_matrix)
        left_projection_last = project(point_last, k @ left_camera_extrinsic_matrix)
        right_projection_last = project(point_last, k @ right_camera_extrinsic_matrix)
        left_projection_first = project(point_first, k @ left_camera_extrinsic_matrix)
        right_projection_first = project(point_first, k @ right_camera_extrinsic_matrix)

        left_projection_errors_last_frame.append(np.linalg.norm(left_projection_last - track.locations[i][0]))
        right_projection_errors_last_frame.append(np.linalg.norm(right_projection_last - track.locations[i][1]))
        left_projection_errors_first_frame.append(np.linalg.norm(left_projection_first - track.locations[i][0]))
        right_projection_errors_first_frame.append(np.linalg.norm(right_projection_first - track.locations[i][1]))


    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)), left_projection_errors_last_frame, label='left camera')
    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)), right_projection_errors_last_frame, label='right camera')
    plt.xlabel('frame')
    plt.ylabel('projection error')
    plt.title(f'reprojection error - last frame (mean error: {np.round(np.mean(left_projection_errors_last_frame + right_projection_errors_last_frame),2)})')
    # show legend:
    plt.legend(loc='upper right')
    # save the plot
    plt.savefig(utils.EX4_DOCS_PATH+ '/' + '4_7_last.png')
    plt.clf()

    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)), left_projection_errors_first_frame, label='left camera')
    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)), right_projection_errors_first_frame, label='right camera')
    plt.xlabel('frame')
    plt.ylabel('projection error')
    plt.title(f'reprojection error - first frame (mean error: {np.round(np.mean(left_projection_errors_first_frame + right_projection_errors_first_frame),2)})')
    # show legend:
    plt.legend(loc='upper left')
    # save the plot
    plt.savefig(utils.EX4_DOCS_PATH+ '/' + '4_7_first.png')




if __name__ == '__main__':
    # run_4_2()
    # run_4_3()
    # run_4_4()
    # run_4_5()
    # run_4_6()
    # run_4_7()
    create_track_db()
