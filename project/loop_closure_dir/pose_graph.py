import gtsam
import numpy as np
import tqdm
import cv2

from shared_utils import read_cameras, RESULTS_PATH
from loop_closure_dir.closure_graph import ClosureGraph
from consensus_matching_dir.consensus_matching_localization import consensus_matching
from loop_closure_dir.loop_closure_bundle import LoopClosureBundle

class PoseGraph:

    def __init__(self, relative_poses, covs, keyframe_frame_indices, stop_closure_at_kf=420):
        self.relative_poses = relative_poses
        self.covs = covs
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.min_gap = 50
        self.closure_graph = None
        self.optimized_estimate = None
        self.keyframe_frame_indices = keyframe_frame_indices
        self.plot_dir = RESULTS_PATH
        self.create_graph(stop_closure_at_kf)
        self.uncertainties = []

    def create_graph(self, stop_closure_at_kf):
        """
        Create the pose graph and perform the loop closure bundle adjustment.
        :param stop_closure_at_kf: the keyframe index to stop the loop closure at.
        """
        # prior for loop closure bundle:
        loop_closure_counter = 0
        plot_matches_flag = True
        first_location_prior = np.array([(1 * np.pi / 180),
                                         (1 * np.pi / 180),
                                         (1 * np.pi / 180),
                                         1e-3,
                                         1e-3,
                                         1e-3])
        # Create first camera symbol
        cur_global_pose = gtsam.Pose3() # 0,0,0
        prev_sym = gtsam.symbol('c', 0)
        self.closure_graph = ClosureGraph(0)
        # Create first camera's pose factor
        sigmas = np.array([0.002,
                                           0.002,
                                           0.002,
                                           0.01,
                                           0.01,
                                           0.01])

        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
        factor = gtsam.PriorFactorPose3(prev_sym, cur_global_pose, pose_noise)
        self.graph.add(factor)
        self.initial_estimate.insert(prev_sym, cur_global_pose)

        # detector and matcher for loop closure consensus matching:
        k, left0_extrinsic, right0_extrinsic = read_cameras()
        detector = cv2.AKAZE_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        for i in tqdm.tqdm(range(len(self.relative_poses))):
            cur_sym = gtsam.symbol('c', i + 1)
            # add initial estimate
            cur_global_pose = cur_global_pose * (self.relative_poses[i])
            self.initial_estimate.insert(cur_sym, cur_global_pose)

            # add relative pose factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(self.covs[i])
            factor = gtsam.BetweenFactorPose3(prev_sym, cur_sym, self.relative_poses[i], noise_model)
            self.graph.add(factor)
            prev_sym = cur_sym

            # add to closure graph
            self.closure_graph.add_node(i+1)
            self.closure_graph.add_edge(i, i+1, np.sqrt(np.linalg.det(self.covs[i])), self.covs[i])

            # search for closures:
            candidates = []
            if i < self.min_gap:
                continue

            if i > stop_closure_at_kf:
                continue

            # loop closure:

            for j in range(i-self.min_gap):
                relative_cov = self.closure_graph.get_sum_cov(j, i+1)
                candidates.append((gtsam.symbol('c', j), cur_sym , relative_cov))

            candidate_indices = np.argsort([self.candidate_score(x) for x in candidates])

            # consensus matching for each candidate:
            frame0_idx = self.keyframe_frame_indices[i + 1]

            best_candidate = (None, None, None, None, None)
            best_candidate_inliers_percentage = 0
            candidates_num = min(5, len(candidate_indices))
            for ci in candidate_indices[:candidates_num]:
                frame1_idx = self.keyframe_frame_indices[ci]
                R_t, inliers_percentage, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1, _ =\
                    consensus_matching(detector=detector,matcher=bf, frame0_id=frame0_idx,
                                       frame1_id=frame1_idx,left0_extrinsic=left0_extrinsic,
                                       right0_extrinsic= right0_extrinsic,k=k, get_inliers=True)
                if inliers_percentage > 0.8:
                    print('found loop closure between frame {} and {}'.format(frame1_idx, frame0_idx))
                    if inliers_percentage > best_candidate_inliers_percentage:
                        best_candidate = (ci, frame1_idx, R_t, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1)
                        best_candidate_inliers_percentage = inliers_percentage

            if best_candidate_inliers_percentage == 0:
                continue


            # create local bundle for the two frames:
            loop_closure_counter += 1
            ci, frame1_idx, R_t, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1 = best_candidate
            if plot_matches_flag:
                consensus_matching(detector=detector, matcher=bf, frame0_id=frame0_idx,
                                   frame1_id=frame1_idx, left0_extrinsic=left0_extrinsic,
                                   right0_extrinsic=right0_extrinsic, k=k,
                                   get_inliers=True,
                                   plot_target_dir=self.plot_dir)
                plot_matches_flag = False
            loop_closure_bundle = LoopClosureBundle(ci,
                                                    i+1,
                                                    R_t,
                                                    first_location_prior,
                                                    inliers_xl_xr_y_frame0,
                                                    inliers_xl_xr_y_frame1)
            loop_closure_bundle.create_factor_graph()
            # optimize the local bundle:
            loop_closure_bundle.optimize()
            R_t, cov = loop_closure_bundle.get_optimized_pose_and_cond_cov()
            # add loop closure factor:
            loop_closure_factor = gtsam.BetweenFactorPose3(
                gtsam.symbol('c', i+1),
                gtsam.symbol('c', ci),
                R_t,
                gtsam.noiseModel.Gaussian.Covariance(cov))

            self.graph.add(loop_closure_factor)
            self.closure_graph.add_edge(ci, i+1, np.sqrt(np.linalg.det(cov)), cov)

            # optimize the (gtsam) graph:
            self.optimize()

        print('loop closure counter: ', loop_closure_counter)


    def candidate_score(self, candidate):
        """
        The score of a candidate is the error of the factor between the two frames
        :param candidate: (sym_j, sym_n, cov) where sym_j is the symbol of the first frame,
        sym_n is the symbol of the second frame, and cov is the approximated covariance of the
        relative pose between the two frames.
        :return: the error of the factor between the two frames
        """
        sym_j = candidate[0]
        sym_n = candidate[1]
        cov = candidate[2]
        pose = gtsam.BetweenFactorPose3(sym_j, sym_n, gtsam.Pose3(),
                                        gtsam.noiseModel.Gaussian.Covariance(cov))
        values = gtsam.Values()
        values.insert(sym_j, self.initial_estimate.atPose3(sym_j))
        values.insert(sym_n, self.initial_estimate.atPose3(sym_n))
        return pose.error(values)

    def optimize(self):
        """
        optimize the graph
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        print('error before optimization: ', optimizer.error())
        self.optimized_estimate = optimizer.optimize()
        print('error after optimization: ', optimizer.error())

    def get_initial_poses(self):
        """
        :return: a list of the initial poses
        """
        poses = []
        for i in range(len(self.relative_poses) + 1):
            poses.append(self.initial_estimate.atPose3(gtsam.symbol('c', i)))
        return poses

    def get_optimized_poses(self):
        """
        :return: a list of the optimized poses
        """
        poses = []
        for i in range(len(self.relative_poses) + 1):
            poses.append(self.optimized_estimate.atPose3(gtsam.symbol('c', i)))
        return poses

    def get_marginal_covariances(self):
        """
        :return: a list of the marginal covariances of the optimized poses
        """
        return gtsam.Marginals(self.graph, self.optimized_estimate)

    def get_optimized_values(self):
        """
        :return: the optimized values
        """
        return self.optimized_estimate

    def get_initial_values(self):
        """
        :return: the initial values
        """
        return self.initial_estimate

    def get_optimized_values_2d(self):
        """
        :return: the optimized values in 2d (birds eye view)
        """
        values = gtsam.Values()
        for i in range(len(self.relative_poses) + 1):
            values.insert(gtsam.symbol('c', i), self.optimized_estimate.atPose3(gtsam.symbol('c', i)).translation())
        return values

    def get_uncertainties(self, values):
        """
        calculate the uncertainties of the optimized poses as the square root of the determinant of
        the conditioned marginal covariance matrix from the first pose to the current pose.
        :param values: list of gtsam values
        :return: list of uncertainties
        """
        marginals = gtsam.Marginals(self.graph, values)
        sym_0 = gtsam.symbol('c', 0)
        uncertainties = []
        for i in range(1, len(self.relative_poses) + 1):
            sym_i = gtsam.symbol('c', i)
            # calculate the covariance between the first and the current pose:
            keys = gtsam.KeyVector()
            keys.append(sym_0)
            keys.append(sym_i)
            information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1],
                                                                                       keys[-1])
            cond_cov_mat = np.linalg.inv(information_mat_first_second)
            uncertainties.append(np.sqrt(np.linalg.det(cond_cov_mat)))

        return uncertainties