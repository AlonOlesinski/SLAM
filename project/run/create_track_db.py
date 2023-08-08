import time
import tqdm
import cv2

from shared_utils import TRACK_DB_PATH, read_cameras, compose_affine_transformations, camera_location_from_extrinsic_matrix, get_gt_left_camera_matrices
from consensus_matching_dir.consensus_matching_localization import consensus_matching
from track_db_dir.track_db import TrackDB
from matplotlib import pyplot as plt
import numpy as np

ITERATIONS = 2559


def plot_pnp_trajectory():
    """
    Plot the trajectory of the camera according to the localization algorithm, and the ground truth trajectory.
    :return: the computed locations of the camera.
    """
    db = create_track_db(TRACK_DB_PATH)
    ex_mats = db.get_all_global_R_t()
    locations = []
    gt_left1_extrinsics = get_gt_left_camera_matrices(ITERATIONS + 1)
    first_gt_location = camera_location_from_extrinsic_matrix(gt_left1_extrinsics[0])
    gt_locations = [first_gt_location]
    for i in range(ITERATIONS):
        locations.append(camera_location_from_extrinsic_matrix(compose_affine_transformations(ex_mats[i], ex_mats[i + 1])))
        gt_left1_extrinsic = gt_left1_extrinsics[i + 1]
        gt_location = camera_location_from_extrinsic_matrix(np.array(gt_left1_extrinsic))
        gt_locations.append(gt_location)

    gt_locations = np.array(gt_locations)
    computed_locations = np.array(locations)
    plt.plot(gt_locations[:, 0], gt_locations[:, 2], label='gt')
    plt.plot(computed_locations[:, 0], computed_locations[:, 2], label='computed')
    plt.legend()
    plt.show()
    return computed_locations

def create_track_db(track_db_path):
    """
    Create a track db from the images in the dataset, using the localization algorithm.
    :param track_db_path: path to save the track db to.
    """
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k, left0_extrinsic, right0_extrinsic = read_cameras()
    prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches = None, None, None, None, None
    track_db = TrackDB()

    start = time.time()
    for i in tqdm.tqdm(range(ITERATIONS)):
        _, _, frame1_descs_and_matches = \
            consensus_matching(detector=feature_detector, matcher=bf,
                                frame0_id=i, frame1_id=i + 1,
                                left0_extrinsic=left0_extrinsic,
                                right0_extrinsic=right0_extrinsic,
                                k=k, track_db=track_db,
                                prev_desc_and_matches=(prev_kp_left1, prev_desc_left1,
                                                       prev_kp_right1, prev_desc_right1,
                                                       prev_frame1_matches),
                                get_desc_and_matches=True)

        prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches = frame1_descs_and_matches


    end = time.time()
    print("time took for db creation: {}".format(end - start))
    track_db.calc_global_R_t()
    TrackDB.serialize(track_db, track_db_path)
    return track_db

if __name__ == '__main__':
    create_track_db(TRACK_DB_PATH)
