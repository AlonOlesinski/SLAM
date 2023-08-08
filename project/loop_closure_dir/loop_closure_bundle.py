import gtsam
import numpy as np

from shared_utils import gtsam_calib_mat, calculate_inverse_of_R_t, read_cameras,\
    compose_affine_transformations


class LoopClosureBundle:
    """
    Class for the loop closure bundle adjustment.
    """
    def __init__(self, c0_idx, c1_idx, relative_pose, first_location_prior_sigma,
                 inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1):
        self.c0_idx = c0_idx
        self.c1_idx = c1_idx
        self.sym_c0 = gtsam.symbol("c", c0_idx)
        self.sym_c1 = gtsam.symbol("c", c1_idx)
        self.relative_pose = relative_pose
        self.first_location_prior_sigma = first_location_prior_sigma
        self.inliers_xl_xr_y_frame0 = inliers_xl_xr_y_frame0
        self.inliers_xl_xr_y_frame1 = inliers_xl_xr_y_frame1
        self.values = gtsam.Values()
        self.graph = gtsam.NonlinearFactorGraph()
        self.k = gtsam_calib_mat()

    def add_cameras(self):
        """
        Add the cameras to the graph, relative to the first camera. Also adds a prior-factor on the
        first camera.
        :return:
        """
        R_t0 = read_cameras()[1]  # W -> C0
        first_camera_noise = gtsam.noiseModel.Diagonal.Sigmas(self.first_location_prior_sigma)
        pose = gtsam.Pose3(R_t0).inverse()  # C0 -> W
        self.graph.add(gtsam.PriorFactorPose3(self.sym_c0, pose, first_camera_noise))
        self.values.insert(self.sym_c0, pose)
        R_t1 = compose_affine_transformations(calculate_inverse_of_R_t(self.relative_pose),
                                              R_t0)  # C1 -> W
        pose = gtsam.Pose3(R_t1)
        self.values.insert(self.sym_c1, pose)

    def add_points_and_projection_factors(self):
        """
        Add the points and the projection factors to the graph.
        """
        sigma_value_x = 3e-2
        sigma_value_y = 3e-2
        projection_noise_model = gtsam.noiseModel.Diagonal.Sigmas([sigma_value_x, sigma_value_x,
                                                                   sigma_value_y])
        # decide which camera to project from by checking if the relative R_t has positive or negative z
        if self.relative_pose[2, 3] > 0:
            xl_xr_y_to_project_from = self.inliers_xl_xr_y_frame1
            last_frame_sym = self.sym_c1
        else:
            xl_xr_y_to_project_from = self.inliers_xl_xr_y_frame0
            last_frame_sym = self.sym_c0

        # create the projections
        for i, (xl, xr, y) in enumerate(xl_xr_y_to_project_from):
            gtsam_stereo_point = gtsam.StereoPoint2(xl, xr, y)
            gtsam_last_frame = gtsam.StereoCamera(
                self.values.atPose3(last_frame_sym), self.k)
            gtsam_point_3d = gtsam_last_frame.backproject(gtsam_stereo_point)
            if gtsam_point_3d[2] > 100 or gtsam_point_3d[2] < 1:
                continue
            sym_point = gtsam.symbol("q", i)
            self.values.insert(sym_point, gtsam_point_3d)

            # add the projection factor between the point and the two cameras
            frame0_xl, frame0_xr, frame0_y = self.inliers_xl_xr_y_frame0[i]
            frame1_xl, frame1_xr, frame1_y = self.inliers_xl_xr_y_frame1[i]

            gtsam_stereo_point_frame0 = gtsam.StereoPoint2(frame0_xl, frame0_xr, frame0_y)
            gtsam_stereo_point_frame1 = gtsam.StereoPoint2(frame1_xl, frame1_xr, frame1_y)

            f0 = gtsam.GenericStereoFactor3D(gtsam_stereo_point_frame0,
                                             projection_noise_model,
                                             self.sym_c0,
                                             sym_point,
                                             self.k)
            f1 = gtsam.GenericStereoFactor3D(gtsam_stereo_point_frame1,
                                             projection_noise_model,
                                             self.sym_c1,
                                             sym_point,
                                             self.k)
            self.graph.add(f0)
            self.graph.add(f1)

    def optimize(self):
        """
        Optimize the graph
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        initial_error = self.graph.error(self.values)
        print(f"Initial error: {initial_error}")
        try:
            self.values = optimizer.optimize()
            final_error = self.graph.error(self.values)
            print(f"Final error: {final_error}")
        except RuntimeError as e:
            print(
                f"Optimization failed due to error between frames: start: {self.c0_idx}, end: {self.c1_idx}")
            print(e)

    def get_optimized_pose_and_cond_cov(self):
        """
        Get the optimized pose and covariance matrix of the second camera
        :return: pose, covariance matrix
        """
        pose = self.values.atPose3(self.sym_c1)
        marginals = gtsam.Marginals(self.graph, self.values)
        keys = gtsam.KeyVector()
        keys.append(self.sym_c0)
        keys.append(self.sym_c1)

        information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1],
                                                                                   keys[-1])
        cond_cov_mat = np.linalg.inv(information_mat_first_second)
        return pose, cond_cov_mat

    def create_factor_graph(self):
        """
        Create the factor graph and add the cameras and the points to it.
        """
        self.add_cameras()
        self.add_points_and_projection_factors()