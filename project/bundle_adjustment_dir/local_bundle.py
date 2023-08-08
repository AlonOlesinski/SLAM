import gtsam
import numpy as np

from shared_utils import read_cameras, compose_affine_transformations,calculate_inverse_of_R_t,\
    gtsam_calib_mat


class LocalBundle:
    """
    Class for local bundle adjustment. Should be used on a small window of a long trajectory.
    """
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
        self.point_sym_cameras_sym_factor_dict = {} # {(point_sym, camera_sym): factor_idx}

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
        R_t0 = read_cameras()[1]  # W -> C0
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

                self.point_sym_cameras_sym_factor_dict[(sym_point, sym_frame)] = self.graph.size() - 1
                if self.debug and frame_id == self.test_stereo_camera_id and not debug_flag and \
                        last_frame_id == self.end_kf_idx:
                    self.test_point_3D_sym = sym_point
                    self.test_stereo_point = gtsam_stereo_point
                    self.test_measurement = (xl, xr, y)
                    self.test_factor = f
                    print(f'initial individual factor error: {f.error(self.values)}')
                    debug_flag = True

    def get_track_num(self):
        """
        Get the number of tracks in the graph
        """
        return len(self.landmarks_idx)

    def optimize(self):
        """
        Optimize the graph using Levenberg Marquardt optimizer
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

    def get_all_syms_of_cameras_with_factor_to_landmark(self, landmark_sym, length):
        """
        return all the camera poses that have factor to the given landmark.
        length is the minimal number of cameras that have factor to the landmark
        """
        camera_syms = []
        frames = []
        for i in range(self.start_kf_idx, self.end_kf_idx + 1):
            c_sym = gtsam.symbol("c", i)
            if (landmark_sym,c_sym) in self.point_sym_cameras_sym_factor_dict.keys():
                camera_syms.append(c_sym)
                frames.append(i)
        if len(camera_syms) >= length:
            return camera_syms, frames
        else:
            return None, None

    def reprojection_error_between_landmark_and_camera(self, camera_sym, landmark_sym):
        """
        return the reprojection error between the landmark and the camera
        """
        # get the camera pose
        camera_pose = self.values.atPose3(camera_sym)
        # get the landmark point
        landmark_point = self.values.atPoint3(landmark_sym)
        # get the camera
        stereo_camera = gtsam.StereoCamera(camera_pose, self.k)
        # project the landmark on the camera
        projected_point = stereo_camera.project(landmark_point)
        # get the measurement
        measurement = self.graph.at(self.point_sym_cameras_sym_factor_dict[(landmark_sym, camera_sym)]).measured()
        # calculate the l2 error
        return np.linalg.norm(measurement.vector() - projected_point.vector())

    def factor_error_between_landmark_and_camera(self, camera_sym, landmark_sym):
        """
        return the factor error between the landmark and the camera
        """
        return self.graph.at(self.point_sym_cameras_sym_factor_dict[(landmark_sym, camera_sym)]).error(self.values)

    def get_error(self):
        """
        return the error of the graph
        """
        return self.graph.error(self.values)