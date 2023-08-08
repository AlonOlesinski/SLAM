import numpy as np

from shared_utils import TRACK_DB_PATH, relative_poses_of_kfs_from_bundle
from track_db_dir.track_db import TrackDB
from bundle_adjustment_dir.bundle_adjustment import BundleAdjustment
from run.utils import plot_trajectory_and_landmarks_from_above
from loop_closure_dir.pose_graph import PoseGraph

def run_bundle_adjustment_with_loop_closure(db_path):
    """
    run bundle adjustment on the track database, plot the trajectory before and after the optimization
    """
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
    plot_trajectory_and_landmarks_from_above(optimized_poses, [], initial_poses,
                                                   file_name='before_and_after_loop_closure.png',
                                                   plot_ground_truth=True,
                                                   title='before and after loop closure',
                                                   key_frames_to_frames=keyframe_indices)

if __name__ == '__main__':
    run_bundle_adjustment_with_loop_closure(db_path=TRACK_DB_PATH)