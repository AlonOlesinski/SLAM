from gtsam.utils import plot
import matplotlib.pyplot as plt

import localization
from bundle_adjustment import *
import numpy as np
import utils
from typing import Iterable
import dijkstar

class ClosureGraph:
    def __init__(self, c0: int):
        self.graph = dijkstar.Graph()
        self.graph.add_node(c0)

    def add_node(self, c: int):
        self.graph.add_node(c)

    def add_edge(self, c1: int, c2: int, weight: float):
        self.graph.add_edge(c1, c2, weight)

    def shortest_path(self, ci: int, cn: int):
        return dijkstar.find_path(self.graph, ci, cn)


class PoseGraph:

    def __init__(self, relative_poses, covs, keyframe_indices):
        self.relative_poses = relative_poses
        self.covs = covs
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.min_gap = 50
        self.closure_graph = None
        self.optimized_estimate = None
        self.create_graph()
        self.keyframe_indices = keyframe_indices

    def create_graph(self):
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

        print('the number of covs is: ', len(self.covs))
        for i in range(len(self.relative_poses)):

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
            self.closure_graph.add_edge(i, i+1, np.sqrt(np.linalg.det(self.covs[i])))

            # search for closures:
            candidates = []

            if i < self.min_gap: # TODO: try different gaps or a clever way to decide when to search for loop closures
                continue

            for j in range(i-self.min_gap): # TODO: try i - k from some k.
                shortest_path_info = self.closure_graph.shortest_path(j, i+1)
                relative_cov = self.sum_of_covs(shortest_path_info.nodes)
                candidates.append((cur_sym, gtsam.symbol('c', j), relative_cov))

            candidate_indices = np.argsort([self.candidate_score(x) for x in candidates])

            # consensus matching for each candidate:
            frame1_idx = i + 1
            for k in candidate_indices[:2]: # TODO: try different number of candidates
                frame0_idx = k
                R_t, inliers_percentage = consensus_matching(frame0_idx, frame1_idx)
                if inliers_percentage > 0.0:
                    print('found inliers percentage: ', inliers_percentage, ' between keyframe {} and {}'.format(frame0_idx, frame1_idx))
                if inliers_percentage > 0.7: # TODO: try different thresholds
                    print('found loop closure between keyframe {} and {}'.format(frame0_idx, frame1_idx))




    # add loop closure factor
    def candidate_score(self, candidate):
        pose = gtsam.BetweenFactorPose3(candidate[0], candidate[1], gtsam.Pose3(),
                                        gtsam.noiseModel.Gaussian.Covariance(candidate[2]))
        values = gtsam.Values()
        values.insert(candidate[0], self.initial_estimate.atPose3(candidate[0]))
        values.insert(candidate[1], self.initial_estimate.atPose3(candidate[1]))
        return pose.error(values)


    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        print('error before optimization: ', optimizer.error())
        self.optimized_estimate = optimizer.optimize()
        print('error after optimization: ', optimizer.error())

    def get_initial_poses(self):
        poses = []
        for i in range(len(self.relative_poses) + 1):
            poses.append(self.initial_estimate.atPose3(gtsam.symbol('c', i)))
        return poses

    def get_optimized_poses(self):
        poses = []
        for i in range(len(self.relative_poses) + 1):
            poses.append(self.optimized_estimate.atPose3(gtsam.symbol('c', i)))
        return poses

    def get_marginal_covariances(self):
        return gtsam.Marginals(self.graph, self.optimized_estimate)

    def get_optimized_values(self):
        return self.optimized_estimate

    def get_optimized_values_2d(self):
        values = gtsam.Values()
        for i in range(len(self.relative_poses) + 1):
            values.insert(gtsam.symbol('c', i), self.optimized_estimate.atPose3(gtsam.symbol('c', i)).translation())
        return values

    def sum_of_covs(self, indices):
        covs = []
        prev_idx = indices[0]
        for cur_idx in indices[1:]:
            values = self.optimized_estimate
            if values is None:
                values = self.initial_estimate
            _, cov = relative_poses_of_kfs(self.graph, values, prev_idx, cur_idx)
            covs.append(cov)
            prev_idx = cur_idx
        return np.sum(covs, axis=0)


def relative_poses_of_kfs_from_bundle(bundle, idx1, idx2):

    return relative_poses_of_kfs(bundle.graph, bundle.transformed_values, idx1, idx2)

def relative_poses_of_kfs(graph, values, idx1, idx2):
    marginals = gtsam.Marginals(graph, values)
    keys = gtsam.KeyVector()
    first_sym = gtsam.symbol('c', idx1)
    last_sym = gtsam.symbol('c', idx2)
    keys.append(first_sym)
    keys.append(last_sym)

    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    # compute the relative pose of the first and last keyframe
    relative_pose = values.atPose3(first_sym).between(values.atPose3(last_sym))

    return relative_pose, cond_cov_mat

def consensus_matching(frame0_id, frame1_id):
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left0_img, right0_img = utils.read_images(frame0_id)
    left1_img, right1_img = utils.read_images(frame1_id)
    blur_factor = 10
    left0_img = cv2.blur(left0_img, (blur_factor, blur_factor))
    right0_img = cv2.blur(right0_img, (blur_factor, blur_factor))
    left1_img = cv2.blur(left1_img, (blur_factor, blur_factor))
    right1_img = cv2.blur(right1_img, (blur_factor, blur_factor))
    localizer = Localizer(frame0_id, feature_detector, bf, left0_img, right0_img, left1_img,
                          right1_img, left0_extrinsic, right0_extrinsic, k)
    R_t, inliers_percent = localizer.pnp_ransac()
    return R_t, inliers_percent


def run_6_1():
    db_path = 'track_db_akaze_ratio=0.5_blur=10.pkl'
    track_db = TrackDB.deserialize(db_path)
    first_location_prior_sigma = np.array([(5 * np.pi / 180) ** 2,
                                           (1 * np.pi / 180) ** 2,
                                           (1 * np.pi / 180) ** 2,
                                           5e-3,
                                           1e-3,
                                           2e-3])
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    print('number of bundles: ', len(bundle_adjustment.bundles))
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()
    bundle = bundle_adjustment.bundles[0]
    marginals = gtsam.Marginals(bundle.graph, bundle.transformed_values)

    gtsam.utils.plot.set_axes_equal(2)
    gtsam.utils.plot.plot_trajectory(0, bundle.transformed_values, marginals=marginals, scale=2.5)
    plt.savefig(utils.EX6_DOCS_PATH + '6_1_trajectory.png')

    relative_pose_start_end_kfs, cov_start_end_kfs = relative_poses_of_kfs_from_bundle(
        bundle, bundle.start_kf_idx, bundle.end_kf_idx)

    print('relative pose of the first and last keyframe: \n', relative_pose_start_end_kfs)
    print('covariance of the relative pose of the first and last keyframe: \n', cov_start_end_kfs)

def run_6_2():
    db_path = 'akaze_ratio=0.5_blur=10_removed_close'
    track_db = TrackDB.deserialize(db_path)
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()

    relative_poses = []
    pose_graph_covs = []
    for bundle in bundle_adjustment.bundles:
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx, bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs)
    initial_poses = pose_graph.get_initial_poses()
    utils.plot_trajectory_and_landmarks_from_above(initial_poses,[] , file_name='6_2_initial_poses.png')

    pose_graph.optimize()
    optimized_poses = pose_graph.get_optimized_poses()
    utils.plot_trajectory_and_landmarks_from_above(optimized_poses,[] , file_name='6_2_optimized_poses_without_cov.png')


    marginals = pose_graph.get_marginal_covariances()
    optimized_values = pose_graph.get_optimized_values()
    gtsam.utils.plot.plot_trajectory(0, optimized_values, marginals=marginals, scale=10)
    gtsam.utils.plot.set_axes_equal(0)
    plt.show()
    # plt.savefig(utils.EX6_DOCS_PATH + '6_2_optimized_poses_with_cov.png')

def run_7_1():
    db_path = 'akaze_ratio=0.5_blur=10_removed_close'
    track_db = TrackDB.deserialize(db_path)
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()

    relative_poses = []
    pose_graph_covs = []
    keyframe_indices = [0]
    for bundle in bundle_adjustment.bundles:
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx, bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices)


if __name__ == '__main__':
    # run_6_1()
    run_7_1()