from gtsam.utils import plot
import matplotlib.pyplot as plt
import tqdm

import bundle_adjustment
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
        self.edge_dict = {}

    def add_node(self, c: int):
        self.graph.add_node(c)

    def add_edge(self, c1: int, c2: int, weight: float, cov_matrix: np.ndarray):
        self.graph.add_edge(c1, c2, weight)
        self.edge_dict[(c1, c2)] = cov_matrix

    def shortest_path(self, ci: int, cn: int):
        return dijkstar.find_path(self.graph, ci, cn)

    def get_sum_cov(self, ci: int, cn: int):
        shortest_path_info = self.shortest_path(ci, cn)
        prev_node_idx = ci
        sum_cov = np.zeros((6,6))
        for i in range(1, len(shortest_path_info.nodes)):
            cur_node_idx = shortest_path_info.nodes[i]
            sum_cov += self.edge_dict[(prev_node_idx, cur_node_idx)]
            prev_node_idx = cur_node_idx
        return sum_cov


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
        self.create_graph(stop_closure_at_kf)
        self.uncertainties = []

    def create_graph(self, stop_closure_at_kf):
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
            if i < self.min_gap: # TODO: try different gaps or a clever way to decide when to search for loop closures
                continue

            if i > stop_closure_at_kf:
                continue

            # loop closure:

            for j in range(i-self.min_gap): # TODO: try i - k from some k.
                relative_cov = self.closure_graph.get_sum_cov(j, i+1)
                candidates.append((gtsam.symbol('c', j), cur_sym , relative_cov))

            candidate_indices = np.argsort([self.candidate_score(x) for x in candidates])

            # consensus matching for each candidate:
            frame0_idx = self.keyframe_frame_indices[i + 1]

            best_candidate = (None, None, None, None, None)
            best_candidate_inliers_percentage = 0
            candidates_num = min(5, len(candidate_indices))
            for k in candidate_indices[:candidates_num]: # TODO: try different number of candidates
                frame1_idx = self.keyframe_frame_indices[k]
                R_t, inliers_percentage, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1 = consensus_matching(frame0_idx, frame1_idx)
                if inliers_percentage > 0.8: # TODO: try different thresholds
                    print('found loop closure between frame {} and {}'.format(frame1_idx, frame0_idx))
                    if inliers_percentage > best_candidate_inliers_percentage:
                        best_candidate = (k, frame1_idx, R_t, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1)
                        best_candidate_inliers_percentage = inliers_percentage

            if best_candidate_inliers_percentage == 0:
                continue


            # create local bundle for the two frames:
            loop_closure_counter += 1
            k, frame1_idx, R_t, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1 = best_candidate
            if plot_matches_flag:
                consensus_matching(frame0_idx, frame1_idx, plot_flag=True)
                plot_matches_flag = False
            loop_closure_bundle = bundle_adjustment.LoopClosureBundle(k,
                                                                      i+1,
                                                                      R_t,
                                                                      first_location_prior, # TODO this might need to be changed for plots
                                                                      inliers_xl_xr_y_frame0,
                                                                      inliers_xl_xr_y_frame1)
            loop_closure_bundle.create_factor_graph()
            # optimize the local bundle:
            loop_closure_bundle.optimize()
            R_t, cov = loop_closure_bundle.get_optimized_pose_and_cond_cov()
            # add loop closure factor:
            loop_closure_factor = gtsam.BetweenFactorPose3(
                gtsam.symbol('c', i+1),
                gtsam.symbol('c', k),
                R_t,
                gtsam.noiseModel.Gaussian.Covariance(cov))

            self.graph.add(loop_closure_factor)
            self.closure_graph.add_edge(k, i+1, np.sqrt(np.linalg.det(cov)), cov)

            # optimize the (gtsam) graph:
            self.optimize()

        print('loop closure counter: ', loop_closure_counter)

    # add loop closure factor
    def candidate_score(self, candidate):
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

    def get_initial_values(self):
        return self.initial_estimate

    def get_optimized_values_2d(self):
        values = gtsam.Values()
        for i in range(len(self.relative_poses) + 1):
            values.insert(gtsam.symbol('c', i), self.optimized_estimate.atPose3(gtsam.symbol('c', i)).translation())
        return values

    def get_uncertainties(self, values):
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
            # pose = gtsam.BetweenFactorPose3(sym_0, sym_i, gtsam.Pose3(),
            #                                 gtsam.noiseModel.Gaussian.Covariance(cond_cov_mat))
            # temp_values = gtsam.Values()
            # temp_values.insert(sym_i, values.atPose3(sym_i))
            # temp_values.insert(sym_0, values.atPose3(sym_0))
            # uncertainties.append(pose.error(temp_values))
            uncertainties.append(np.sqrt(np.linalg.det(cond_cov_mat)))

        return uncertainties



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

def consensus_matching(frame0_id, frame1_id, plot_flag = False):
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k, left0_extrinsic, right0_extrinsic = utils.read_cameras()
    left0_img, right0_img = utils.read_images(frame0_id)
    left1_img, right1_img = utils.read_images(frame1_id)
    blur_factor = 10
    if not plot_flag:
        left0_img = cv2.blur(left0_img, (blur_factor, blur_factor))
        right0_img = cv2.blur(right0_img, (blur_factor, blur_factor))
        left1_img = cv2.blur(left1_img, (blur_factor, blur_factor))
        right1_img = cv2.blur(right1_img, (blur_factor, blur_factor))
    localizer = Localizer(frame0_id, feature_detector, bf, left0_img, right0_img, left1_img,
                          right1_img, left0_extrinsic, right0_extrinsic, k)
    R_t, inliers_percent, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1 = localizer.pnp_ransac(get_inliers=True, plot_flag=plot_flag)
    return R_t, inliers_percent, inliers_xl_xr_y_frame0, inliers_xl_xr_y_frame1


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
    pose_graph.optimize()
    optimized_poses = pose_graph.get_optimized_poses()
    utils.plot_trajectory_and_landmarks_from_above(optimized_poses,[] , file_name='after_loop_closure.png')

def run_7_5_3():
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
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx,
                                                               bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices, stop_closure_at_kf=385)
    pose_graph.optimize()
    marginals = pose_graph.get_marginal_covariances()
    optimized_values = pose_graph.get_optimized_values()
    gtsam.utils.plot.plot_trajectory(0, optimized_values, marginals=marginals, scale=10)
    gtsam.utils.plot.set_axes_equal(0)
    plt.show()
    plt.clf()

def run_7_5_4():
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
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx,
                                                               bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices)
    initial_poses = pose_graph.get_initial_poses()
    pose_graph.optimize()
    optimized_poses = pose_graph.get_optimized_poses()
    utils.plot_trajectory_and_landmarks_from_above(optimized_poses, [], initial_poses,
                                                   file_name='before_and_after_loop_closure.png', plot_ground_truth=True
                                                   , key_frames_to_frames=keyframe_indices)

def run_7_5_5():
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
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx,
                                                               bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices)
    pose_graph.optimize()
    gt_left_extrinsics = utils.get_gt_left_camera_matrices(keyframe_indices)
    gt_locations = [localization.camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                    extrinsic_matrix in gt_left_extrinsics]
    gt_locations = np.array(gt_locations)
    optimized_poses = pose_graph.get_optimized_poses()
    initial_poses = pose_graph.get_initial_poses()
    optimized_locations = np.array([[pose.x(), pose.y(), pose.z()] for pose in optimized_poses])
    initial_locations = np.array([[pose.x(), pose.y(), pose.z()] for pose in initial_poses])
    optimized_errors = np.linalg.norm(np.array(optimized_locations) - gt_locations, axis=1)
    initial_errors = np.linalg.norm(np.array(initial_locations) - gt_locations, axis=1)
    key_frames_list = bundle_adjustment.get_key_frames_list()
    plt.clf()
    plt.plot(optimized_errors, label='optimized')
    plt.plot(initial_errors, label='initial')
    plt.title(f'localization error in meters for each keyframe ({len(key_frames_list)} keyframes)')
    plt.xlabel('keyframe index')
    plt.ylabel('error [m]')
    plt.legend()
    plt.savefig(utils.EX7_DOCS_PATH + f'keyframes_errors.png', dpi=300)

def run_7_5_6():
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
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx,
                                                               bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices, stop_closure_at_kf=420)
    pose_graph.optimize()
    optimized_uncertainties = pose_graph.get_uncertainties(pose_graph.get_optimized_values())
    pose_graph = PoseGraph(relative_poses, pose_graph_covs, keyframe_indices,
                           stop_closure_at_kf=0)
    initial_uncertainties = pose_graph.get_uncertainties(pose_graph.get_initial_values())
    plt.clf()
    plt.plot(optimized_uncertainties, label='after')
    plt.plot(initial_uncertainties, label='before')
    plt.title('Uncertainties before/after loop closure')
    plt.xlabel('keyframe index')
    plt.ylabel('uncertainty')
    # set the scale to logaritmic
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(utils.EX7_DOCS_PATH + f'uncertainties.png', dpi=300)



if __name__ == '__main__':
    # run_6_1()
    # run_7_5_3()
    # run_7_5_4()
    # run_7_5_5()
    run_7_5_6()