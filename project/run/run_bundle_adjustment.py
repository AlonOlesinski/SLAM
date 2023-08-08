import numpy as np

from track_db_dir.track_db import TrackDB
from bundle_adjustment_dir.bundle_adjustment import BundleAdjustment
import utils
from shared_utils import TRACK_DB_PATH


def run_bundle_adjustment(db_path):
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
    print('number of bundles: ', len(bundle_adjustment.bundles))
    bundle_adjustment.run_local_adjustments(print_final_bundle_properties=True)
    bundle_adjustment.set_global_coordinates()
    optimized_poses = bundle_adjustment.get_all_camera_poses_global()
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()
    points = bundle_adjustment.get_all_landmarks_global()
    initial_poses = bundle_adjustment.get_all_camera_poses_global()

    key_frames_to_frames = bundle_adjustment.get_key_frames_indices()

    utils.plot_trajectory_and_landmarks_from_above(optimized_poses, points,
                                                   cameras_initial_estimate=initial_poses,
                                                   plot_ground_truth=True,
                                                   file_name=f'trajectory_bundle_adjustment.png',
                                                   key_frames_to_frames=key_frames_to_frames)

if __name__ == '__main__':
    run_bundle_adjustment(db_path=TRACK_DB_PATH)


