from gtsam.utils import plot
import matplotlib.pyplot as plt
from bundle_adjustment import *
import numpy as np
import utils
from typing import Iterable

class PoseGraph:
    def __init__(self, relative_poses, covs):
        self.relative_poses = relative_poses
        self.covs = covs
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.create_graph()
        self.optimized_estimate = None

    def create_graph(self):
        # Create first camera symbol
        cur_global_pose = gtsam.Pose3() # 0,0,0
        prev_sym = gtsam.symbol('c', 0)

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


def relative_poses_of_kfs(bundle, idx1, idx2):

    marginals = gtsam.Marginals(bundle.graph, bundle.transformed_values)
    keys = gtsam.KeyVector()
    first_sym = gtsam.symbol('c', idx1)
    last_sym = gtsam.symbol('c', idx2)
    keys.append(first_sym)
    keys.append(last_sym)

    information_mat_first_second = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    cond_cov_mat = np.linalg.inv(information_mat_first_second)

    # compute the relative pose of the first and last keyframe
    relative_pose = bundle.values.atPose3(first_sym).between(bundle.values.atPose3(last_sym))

    return relative_pose, cond_cov_mat


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

    relative_pose_start_end_kfs, cov_start_end_kfs = relative_poses_of_kfs(
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
        relative_pose, cov = relative_poses_of_kfs(bundle, bundle.start_kf_idx, bundle.end_kf_idx)
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

def plot_trajectory(
        fignum: int,
        values: gtsam.Values,
        scale: float = 1,
        marginals: gtsam.Marginals = None,
        title: str = "Plot Trajectory",
        axis_labels: Iterable[str] = ("X axis", "Y axis", "Z axis"),
) -> None:
    """
    Plot a complete 2D/3D trajectory using poses in `values`.

    Args:
        fignum: Integer representing the figure number to use for plotting.
        values: Values containing some Pose2 and/or Pose3 values.
        scale: Value to scale the poses by.
        marginals: Marginalized probability values of the estimation.
            Used to plot uncertainty bounds.
        title: The title of the plot.
        axis_labels (iterable[string]): List of axis labels to set.
    """
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])
    # axes.view_init(0, -90)

    # Plot 2D poses, if any
    poses = gtsam.utilities.allPose2s(values)
    for key in poses.keys():
        pose = poses.atPose2(key)
        if marginals:
            covariance = marginals.marginalCovariance(key)
        else:
            covariance = None

        gtsam.utils.plot.plot_pose2_on_axes(axes,
                           pose,
                           covariance=covariance,
                           axis_length=scale)

    # Then 3D poses, if any
    poses = gtsam.utilities.allPose3s(values)
    for key in poses.keys():
        pose = poses.atPose3(key)
        if marginals:
            covariance = marginals.marginalCovariance(key)
        else:
            covariance = None

        gtsam.utils.plot.plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

    fig.suptitle(title)
    fig.canvas.set_window_title(title.lower())




if __name__ == '__main__':
    # run_6_1()
    run_6_2()