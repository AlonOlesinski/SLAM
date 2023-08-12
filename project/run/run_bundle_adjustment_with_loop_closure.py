import numpy as np

from shared_utils import TRACK_DB_PATH
from track_db_dir.track_db import TrackDB
from bundle_adjustment_dir.bundle_adjustment import BundleAdjustment
from run.utils import plot_trajectory_and_landmarks_from_above
from loop_closure_dir.pose_graph import create_pose_graph_from_ba

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

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    keyframe_frame_indices = pose_graph.get_keyframe_frame_indices()
    initial_poses = pose_graph.get_initial_poses()
    pose_graph.optimize()
    optimized_poses = pose_graph.get_optimized_poses()
    plot_trajectory_and_landmarks_from_above(optimized_poses, [], initial_poses,
                                                   file_name='before_and_after_loop_closure.png',
                                                   plot_ground_truth=True,
                                                   title='before and after loop closure',
                                                   key_frames_to_frames=keyframe_frame_indices)

if __name__ == '__main__':
    run_bundle_adjustment_with_loop_closure(db_path=TRACK_DB_PATH)