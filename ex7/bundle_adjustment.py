import operator

import gtsam
import numpy as np
import matplotlib.pyplot as plt

import localization
import utils
from localization import *
from gtsam.utils import plot
import tqdm

NEGATIVE_POINTS = 0


def calculate_inverse_of_R_t(R_t):
    """
    calculate the inverse of R_t
    :param R_t: R_t
    :return: inverse of R_t
    """
    R = R_t[:3, :3]
    t = R_t[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return np.hstack((R_inv, t_inv.reshape(-1, 1)))


def gtsam_calib_mat():
    """
    convert the calibration matrix to gtsam format
    :return: gtsam calibration matrix
    """
    K, _, right_R_t = utils.read_cameras()
    fx = K[0, 0]
    fy = K[1, 1]
    s = K[0, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    b = -right_R_t[0, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, b)


def gtsam_pose_from_global_R_t(R_t):
    """
    convert the global R_t to gtsam pose
    :param R_t: R_t
    :return: gtsam pose
    """
    R_t_inv = calculate_inverse_of_R_t(R_t)
    return gtsam.Pose3(gtsam.Rot3(R_t_inv[:3, :3]), gtsam.Point3(R_t_inv[:3, 3]))


def global_R_t_to_C0_R_t(R_t, R_t0):
    """
    Return a gtsam pose3 of R_t in the coordinates of R_t0
    :param R_t: as returned by TrackDB.get_global_R_t (W -> Ci)
    :param R_t0: inverse of the matrix returned by TrackDB.get_global_R_t (C0 -> W)
    :return: gtsam pose3 (Ci -> C0)
    """
    R_t = compose_affine_transformations(R_t, R_t0)  # C0 -> Ci
    R_t = calculate_inverse_of_R_t(R_t)  # Ci -> C0
    return R_t


class LocalBundle:
    def __init__(self, track_db, start_kf_idx, end_kf_idx, first_location_prior_sigma, debug=False):
        self.track_db = track_db
        self.start_kf_idx = start_kf_idx
        self.end_kf_idx = end_kf_idx
        self.first_location_prior_sigma = first_location_prior_sigma
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.k = gtsam_calib_mat()
        self.last_camera_pose = None
        self.landmarks_idx = []
        self.transformed_values = None

        # for debugging (question 5.2)
        self.debug = debug
        self.test_stereo_camera_id = start_kf_idx
        self.test_point_3D_sym = None
        self.test_camera_pose_sym = None
        self.test_point_id = 0
        self.test_stereo_point = None
        self.test_measurement = None
        self.test_factor = None

    def add_cameras(self):
        """
        Add the cameras to the graph, relative to the first camera. Also adds a prior-factor on the
        first camera.
        :return:
        """
        R_t0 = utils.read_cameras()[1]  # W -> C0
        # R_t0 = calculate_inverse_of_R_t(R_t0)  # C0 -> W
        first_camera_noise = gtsam.noiseModel.Diagonal.Sigmas(self.first_location_prior_sigma)
        sym = gtsam.symbol("c", self.start_kf_idx)
        pose = gtsam.Pose3(R_t0).inverse()  # C0 -> W
        if self.debug:
            self.test_camera_pose_sym = sym
        self.graph.add(gtsam.PriorFactorPose3(sym, pose, first_camera_noise))
        self.values.insert(sym, pose)

        composed_R_t = [R_t0]
        for frame_id in range(self.start_kf_idx + 1, self.end_kf_idx + 1):
            sym = gtsam.symbol("c", frame_id)
            R_ti_relative = self.track_db.get_relative_R_t(frame_id)  # Ci-1 -> Ci
            R_ti_relative = calculate_inverse_of_R_t(R_ti_relative)  # Ci -> Ci-1
            composed_R_t.append(
                compose_affine_transformations(R_ti_relative, composed_R_t[-1]))  # Ci -> W
            R_ti = composed_R_t[-1]
            pose = gtsam.Pose3(R_ti)
            self.values.insert(sym, pose)

        self.last_camera_pose = pose

    def tracks_gen(self):
        """
        Generator for the tracks in the given range of frames.
        :return:
        """
        for t in self.track_db.get_frame_tracks(self.start_kf_idx):
            yield t

    def add_points_and_projection_factors(self):
        """
        Add the points and the projection factors to the graph. We add a point only if it is present in
        the first frame and atleast 2 other frames.
        """
        debug_flag = False  # flag to debug only one track
        sigma_value_x = 3e-2
        sigma_value_y = 3e-2
        print("x projection sigma: {}".format(sigma_value_x), " (should be between 2 and 4)")
        projection_noise_model = gtsam.noiseModel.Diagonal.Sigmas([sigma_value_x, sigma_value_x,
                                                                       sigma_value_y])
        for i, t in enumerate(self.tracks_gen()):
            last_frame_id = min(t.first_frame_id + len(t) - 1, self.end_kf_idx)
            gtsam_last_frame = gtsam.StereoCamera(
                self.values.atPose3(gtsam.symbol('c', last_frame_id)), self.k)
            if last_frame_id - self.start_kf_idx <= 2:
                continue
            # create the point by projecting the track from the last frame
            location_id = last_frame_id - t.first_frame_id
            xl, y = t.locations[location_id][0]
            xr = t.locations[location_id][1][0]
            gtsam_stereo_point = gtsam.StereoPoint2(xl, xr, y)
            gtsam_point_3d = gtsam_last_frame.backproject(gtsam_stereo_point)
            if gtsam_point_3d[2] < 0:
                global NEGATIVE_POINTS
                NEGATIVE_POINTS += 1
                continue
            if gtsam_point_3d[2] > 100 or gtsam_point_3d[2] < 1:
                continue
            self.landmarks_idx.append(i)
            sym_point = gtsam.symbol("q", i)
            self.values.insert(sym_point, gtsam_point_3d)

            # add the projection factor between the point and frame for all the relevant frames

            for frame_id in range(self.start_kf_idx, last_frame_id + 1):
                sym_frame = gtsam.symbol("c", frame_id)
                xl, y = t.locations[frame_id - t.first_frame_id][0]
                xr = t.locations[frame_id - t.first_frame_id][1][0]
                gtsam_stereo_point = gtsam.StereoPoint2(xl, xr, y)
                f = gtsam.GenericStereoFactor3D(gtsam_stereo_point,
                                                projection_noise_model,
                                                sym_frame,
                                                sym_point,
                                                self.k)
                self.graph.add(f)
                if self.debug and frame_id == self.test_stereo_camera_id and not debug_flag and \
                        last_frame_id == self.end_kf_idx:
                    self.test_point_3D_sym = sym_point
                    self.test_stereo_point = gtsam_stereo_point
                    self.test_measurement = (xl, xr, y)
                    self.test_factor = f
                    print(f'initial individual factor error: {f.error(self.values)}')
                    debug_flag = True

    def get_track_num(self):
        return len(self.landmarks_idx)

    def optimize(self):
        """
        Optimize the graph
        :return:
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        initial_error = self.graph.error(self.values)
        try:
            self.values = optimizer.optimize()
        except RuntimeError as e:
            print(
                f"Optimization failed due to error in bundle: start_kf: {self.start_kf_idx}, end_kf: {self.end_kf_idx}")
            print(e)
        if self.debug:
            print(f'optimized individual factor error: {self.test_factor.error(self.values)}')
            print("initial error = %d" % initial_error)
            print("final error = %d" % optimizer.error())

    def create_factor_graph(self):
        """
        create the factor graph
        """
        self.add_cameras()
        self.add_points_and_projection_factors()

    def project_test_point_on_test_camera(self):
        """
        project the test point on the test camera
        """
        stereo_camera = gtsam.StereoCamera(self.values.atPose3(self.test_camera_pose_sym), self.k)
        return stereo_camera.project(self.values.atPoint3(self.test_point_3D_sym))

    def get_test_measurement(self):
        """
        return the test measurement
        """
        return self.test_measurement

    def get_all_landmarks_point3_gen(self, with_sym=False, transformed=False):
        """
        return all the landmarks
        """
        if transformed:
            values = self.transformed_values
        else:
            values = self.values

        for i in self.landmarks_idx:
            if with_sym:
                sym = gtsam.symbol("q", i)
                yield sym, values.atPoint3(sym)
            else:
                yield values.atPoint3(gtsam.symbol("q", i))

    def get_all_cameras_pose3_gen(self, with_sym=False, transformed=False):
        """
        return all the camera poses
        """
        if transformed:
            values = self.transformed_values
        else:
            values = self.values
        for i in range(self.start_kf_idx, self.end_kf_idx + 1):
            if with_sym:
                sym = gtsam.symbol("c", i)
                yield sym, values.atPose3(sym)
            else:
                yield values.atPose3(gtsam.symbol("c", i))

    def get_error(self):
        """
        return the error of the graph
        """
        return self.graph.error(self.values)


class BundleAdjustment:

    def __init__(self, track_db, location_prior_sigma):
        self.track_db = track_db
        self.location_prior_sigma = location_prior_sigma
        self.bundles = []

    def set_bundles_debug(self):
        """
        Set the bundle by a constant length of 3 frames, for debugging purposes.
        :return:
        """
        for i in range(0, self.track_db.get_number_of_frames() - 3, 3):
            bundle = LocalBundle(self.track_db, i, i + 3, self.location_prior_sigma)
            bundle.create_factor_graph()
            self.bundles.append(bundle)

    def set_bundles(self):
        """
        set the local bundles using the following heuristic. Start from the first frame and add the
        median (or other percentile) track length to the bundle. Minimum bundle length is 5 frames,
        maximum bundle length is 20 frames.
        """
        start_kf = 0
        max_frame = self.track_db.get_number_of_frames() - 1
        while start_kf < max_frame:
            end_kf = start_kf + self.track_db.get_median_track_length_from_frame(start_kf)
            if end_kf < start_kf + 4:
                end_kf = start_kf + 4
            if end_kf > start_kf + 20:
                end_kf = start_kf + 20
            if end_kf > max_frame:
                end_kf = max_frame
            bundle = LocalBundle(self.track_db, start_kf, end_kf, self.location_prior_sigma)
            bundle.create_factor_graph()
            self.bundles.append(bundle)
            start_kf = end_kf

    def print_key_frames(self):
        """
        print the key frames of the bundles
        """
        prev_bundle = None
        for bundle in self.bundles:
            print(
                f"start kf: {bundle.start_kf_idx}, end kf:  {bundle.end_kf_idx}, number of landmarks: {bundle.get_track_num()}")
            if prev_bundle is not None:
                assert prev_bundle.end_kf_idx == bundle.start_kf_idx
            prev_bundle = bundle

    def bundle_length_hist(self):
        """
        plot the histogram of the bundle lengths
        """
        bundle_lengths = [bundle.end_kf_idx - bundle.start_kf_idx + 1 for bundle in self.bundles]
        plt.hist(bundle_lengths, bins=16)
        plt.show()
        plt.clf()

    def run_local_adjustments(self, print_final_bundle_properties=False):
        """
        run levenberg marquardt optimization on all the bundles
        :param print_final_bundle_properties: print the location of the last bundle's first frame
         after optimization
        """
        error_before = 0
        error_after = 0
        for bundle in tqdm.tqdm(self.bundles, total=len(self.bundles)):
            error_before += bundle.get_error()
            bundle.optimize()
            if bundle.end_kf_idx == self.track_db.get_number_of_frames() - 1 \
                    and print_final_bundle_properties:
                print(f'location of the first frame after optimization:'
                      f' {bundle.values.atPose3(gtsam.symbol("c", bundle.start_kf_idx)).translation()}')
                first_factor = bundle.graph.at(0)
                print(
                    f'the anchoring factor error of the first frame after optimization: {first_factor.error(bundle.values)}')
            error_after += bundle.get_error()
        print("error before: %f, error after: %f" % (error_before, error_after))
        print("all bundles are optimized")

    def set_global_coordinates(self):
        """
        set the global coordinates of the landmarks and cameras. This should be done after all the
         local adjustments
        """
        self.bundles[0].transformed_values = self.bundles[0].values
        for i in range(1, len(self.bundles)):
            new_values = gtsam.Values()
            prev_bundle = self.bundles[i - 1]
            cur_bundle = self.bundles[i]
            prev_last_pose = prev_bundle.transformed_values.atPose3(
                gtsam.symbol("c", prev_bundle.end_kf_idx))
            # transform cameras
            for j in range(self.bundles[i].start_kf_idx, self.bundles[i].end_kf_idx + 1):
                sym = gtsam.symbol("c", j)
                new_values.insert(sym, prev_last_pose * cur_bundle.values.atPose3(sym))
            # transform landmarks
            for j in cur_bundle.landmarks_idx:
                sym = gtsam.symbol("q", j)
                point = cur_bundle.values.atPoint3(sym)
                point = prev_last_pose.transformFrom(point)
                new_values.insert(sym, point)
            cur_bundle.transformed_values = new_values

    def get_all_camera_poses_global(self):
        """
        :return: a list of all camera poses in the global coordinates
        """
        poses = []
        for bundle in self.bundles:
            for i in range(bundle.start_kf_idx, bundle.end_kf_idx):
                poses.append(bundle.transformed_values.atPose3(gtsam.symbol("c", i)))
        poses.append(self.bundles[-1].transformed_values.atPose3(
            gtsam.symbol("c", self.bundles[-1].end_kf_idx)))
        return poses

    def get_all_landmarks_global(self):
        """
        :return: a list of all landmarks in the global coordinates
        """
        landmarks = []
        for bundle in self.bundles:
            for i in bundle.landmarks_idx:
                landmarks.append(bundle.transformed_values.atPoint3(gtsam.symbol("q", i)))
        return landmarks

    def get_key_frames_list(self):
        """
        :return: a list of all key frames indices
        """
        kf_list = []
        for bundle in self.bundles:
            kf_list.append(bundle.start_kf_idx)
        kf_list.append(self.bundles[-1].end_kf_idx)
        return kf_list


def run_5_1():
    track_db = TrackDB.deserialize('track_db_akaze_ratio=0.5_blur=10.pkl')
    track = track_db.get_track_with_len_at_least_10()
    values = gtsam.Values()
    k = gtsam_calib_mat()
    # find the last frame's R_t in the coordinates of the first frame and in the gtsam convention (camera to world):
    R_t0 = track_db.get_global_R_t(track.first_frame_id)  # W -> C0
    R_t0 = calculate_inverse_of_R_t(R_t0)  # C0 -> W
    R_tn = track_db.get_global_R_t(track.first_frame_id + len(track) - 1)  # W -> Cn
    R_tn = global_R_t_to_C0_R_t(R_tn, R_t0)  # Cn -> C0
    gtsam_pose_n = gtsam.Pose3(R_tn)
    gtsam_frame_n = gtsam.StereoCamera(gtsam_pose_n, k)
    # triangulate the track from the last frame
    x_l, y = track.locations[-1][0]
    x_r, _ = track.locations[-1][1]
    gtsam_point2d = gtsam.StereoPoint2(x_l, x_r, y)
    gtsam_point3d = gtsam_frame_n.backproject(gtsam_point2d)
    projected_point_symbol = gtsam.symbol("q", 0)
    values.insert(projected_point_symbol, gtsam_point3d)

    # project the point to all the frames, calculate the l2 norm of the reprojection error and the factor error.
    projections = []  # (x_l, x_r, y)
    factors = []
    for frame_id in range(track.first_frame_id, track.first_frame_id + len(track)):
        R_ti = track_db.get_global_R_t(frame_id)  # W -> Ci
        R_ti = global_R_t_to_C0_R_t(R_ti, R_t0)  # Ci -> C0
        gtsam_pose_i = gtsam.Pose3(R_ti)
        values.insert(gtsam.symbol("c", frame_id), gtsam_pose_i)
        gtsam_frame_i = gtsam.StereoCamera(gtsam_pose_i, k)
        gtsam_point2d = gtsam_frame_i.project(gtsam_point3d)
        projections.append((gtsam_point2d.uL(), gtsam_point2d.uR(), gtsam_point2d.v()))

        # add the factor
        print('frame_id: ', frame_id)
        x_l, y = track.locations[frame_id - track.first_frame_id][0]
        x_r, _ = track.locations[frame_id - track.first_frame_id][1]
        computed_point2d = gtsam.StereoPoint2(x_l, x_r, y)

        f = gtsam.GenericStereoFactor3D(computed_point2d,
                                        gtsam.noiseModel.Isotropic.Sigma(3, 1),
                                        gtsam.symbol("c", frame_id),
                                        projected_point_symbol,
                                        k)
        factors.append(f)

    left_projection_errors = [np.linalg.norm(
        np.array([projections[i][0], projections[i][2]]) - np.array(track.locations[i][0])) for i in
        range(len(track))]
    right_projection_errors = [np.linalg.norm(
        np.array([projections[i][1], projections[i][2]]) - np.array(track.locations[i][1])) for i in
        range(len(track))]
    # plot the reprojection error size (l2 norm)
    plt.figure()
    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)),
                left_projection_errors, label='left')
    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)),
                right_projection_errors, label='right')
    plt.legend()
    plt.title('l2 norm of the reprojection error')
    plt.xlabel('frame id')
    plt.ylabel('l2 norm of the reprojection error')
    # save the plot
    plt.savefig(utils.EX5_DOCS_PATH + '5_1_l2_error.png')
    plt.clf()

    # plot the factor error
    factor_errors = []
    for factor in factors:
        factor_errors.append(factor.error(values))
    plt.figure()
    plt.scatter(range(track.first_frame_id, track.first_frame_id + len(track)), factor_errors,
                label='factor error')

    plt.title('factor error')
    plt.xlabel('frame id')
    plt.ylabel('factor error')
    # save the plot
    plt.savefig(utils.EX5_DOCS_PATH + '5_1_factor_error.png')
    plt.clf()

    # plot the factor error as a function of the reprojection error
    plt.figure()
    x_values = []
    for i in range(len(left_projection_errors)):
        x_values.append((left_projection_errors[i] + right_projection_errors[i]) / 2)
    x_values = np.array(x_values)
    plt.plot(x_values, factor_errors, label='factor error')
    plt.plot(x_values, (x_values ** 2) * 0.5, label='1/2 reprojection error squared')
    plt.legend()
    plt.title('factor error as a function of the reprojection error')
    plt.xlabel('reprojection error')
    plt.ylabel('factor error')
    # save the plot
    plt.savefig(utils.EX5_DOCS_PATH + '5_1_factor_error_as_a_function_of_reprojection_error.png')


def run_5_2_1():
    track_db = TrackDB.deserialize('track_db_akaze_ratio=0.5_blur=10.pkl')
    start_kf_id = 0
    end_kf_id = 10
    first_location_prior_sigma = np.array([1, 1, 1, 1, 1, 1])
    bundle = LocalBundle(track_db, start_kf_id, end_kf_id, first_location_prior_sigma, debug=True)
    bundle.create_factor_graph()
    bundle.optimize()


def run_5_2_2():
    track_db = TrackDB.deserialize('track_db_akaze_ratio=0.5_blur=10.pkl')
    start_kf_id = 0
    end_kf_id = 20
    first_location_prior_sigma = np.array([1, 1, 1, 1, 1, 1])
    bundle = LocalBundle(track_db, start_kf_id, end_kf_id, first_location_prior_sigma, debug=True)
    bundle.create_factor_graph()
    projected_point = bundle.project_test_point_on_test_camera()
    initial_projected_x_l, initial_projected_x_r, initial_projected_y = projected_point.uL(), projected_point.uR(), projected_point.v()
    measured_x_l, measured_x_r, measured_y = bundle.get_test_measurement()
    bundle.optimize()
    projected_point = bundle.project_test_point_on_test_camera()
    print('projected point: ', initial_projected_x_l, initial_projected_x_r, initial_projected_y)
    print('measured point: ', measured_x_l, measured_x_r, measured_y)
    opt_projected_x_l, opt_projected_x_r, opt_projected_y = projected_point.uL(), projected_point.uR(), projected_point.v()
    print('optimized projected point: ', opt_projected_x_l, opt_projected_x_r, opt_projected_y)
    # overlay the left and right images with the projected point and the measurement, before and
    # after optimization:
    img_left, img_right = utils.read_images(bundle.start_kf_idx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('before optimization (green - measured point, red - projected point)')
    axes[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    axes[0].scatter(initial_projected_x_l, initial_projected_y, c='r', s=5,
                    label='initial projected point')
    axes[0].scatter(measured_x_l, measured_y, c='cyan', s=5, label='measured point')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    axes[1].scatter(initial_projected_x_r, initial_projected_y, c='r', s=5,
                    label='initial projected point')
    axes[1].scatter(measured_x_r, measured_y, c='cyan', s=5, label='measured point')
    axes[1].axis('off')
    plt.savefig(utils.EX5_DOCS_PATH + '5_2_2_before_optimization.png', dpi=300)
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('after optimization (green - measured point, red - projected point)')
    axes[0].imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    axes[0].scatter(opt_projected_x_l, opt_projected_y, c='r', s=2,
                    label='optimized projected point')
    axes[0].scatter(measured_x_l, measured_y, c='cyan', s=2, label='measured point')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    axes[1].scatter(opt_projected_x_r, opt_projected_y, c='r', s=2,
                    label='optimized projected point')
    axes[1].scatter(measured_x_r, measured_y, c='cyan', s=2, label='measured point')
    axes[1].axis('off')
    plt.savefig(utils.EX5_DOCS_PATH + '5_2_2_after_optimization.png', dpi=300)
    plt.clf()
    gtsam.utils.plot.plot_trajectory(fignum=0, values=bundle.values)
    gtsam.utils.plot.set_axes_equal(0)
    plt.savefig(utils.EX5_DOCS_PATH + '5_2_3_trajectory.png', dpi=300)
    plt.clf()
    # plot the trajectory and landmarks as a view from above of the scene:
    utils.plot_trajectory_and_landmarks_from_above(list(bundle.get_all_cameras_pose3_gen()),
                                                   list(bundle.get_all_landmarks_point3_gen()))


def run_5_3():
    db_path = 'akaze_ratio=0.5_blur=10_removed_close'
    localization.run_4_2(db_path)
    track_db = TrackDB.deserialize(db_path)
    first_location_prior_sigma = np.array([(1 * np.pi / 180),
                                           (1 * np.pi / 180),
                                           (1 * np.pi / 180),
                                           1e-3,
                                           1e-3,
                                           1e-3])
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    print('number of bundles: ', len(bundle_adjustment.bundles))
    print('number of negative points: ', NEGATIVE_POINTS)
    bundle_adjustment.print_key_frames()
    bundle_adjustment.bundle_length_hist()
    bundle_adjustment.run_local_adjustments(print_final_bundle_properties=True)
    bundle_adjustment.set_global_coordinates()
    optimized_poses = bundle_adjustment.get_all_camera_poses_global()
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.print_key_frames()
    bundle_adjustment.set_global_coordinates()
    points = bundle_adjustment.get_all_landmarks_global()
    initial_poses = bundle_adjustment.get_all_camera_poses_global()

    utils.plot_trajectory_and_landmarks_from_above(optimized_poses, points,
                                                   cameras_initial_estimate=initial_poses,
                                                   plot_ground_truth=True,
                                                   file_name=f'5_3_all cameras_{db_path}.png')

    # plot the localization error in mteres (euclidean distance) for each keyframe:

    gt_left_extrinsics = utils.get_gt_left_camera_matrices(2560)
    gt_locations = [localization.camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                    extrinsic_matrix in gt_left_extrinsics]
    gt_locations = np.array(gt_locations)
    optimized_locations = np.array([[pose.x(), pose.y(), pose.z()] for pose in optimized_poses])
    errors = np.linalg.norm(np.array(optimized_locations) - gt_locations, axis=1)
    key_frames_list = bundle_adjustment.get_key_frames_list()
    key_frames_errors = errors[key_frames_list]
    plt.plot(key_frames_errors)
    plt.title(f'localization error in meters for each keyframe ({len(key_frames_list)} keyframes)')
    plt.xlabel('keyframe index')
    plt.ylabel('error [m]')
    plt.savefig(utils.EX6_DOCS_PATH + f'5_3_keyframes_error_{db_path}.png', dpi=300)


if __name__ == '__main__':
    # run_5_2_1()
    # run_5_2_2()
    run_5_3()
    # localization.create_track_db("akaze_ratio=0.5_blur=10_removed_close")
