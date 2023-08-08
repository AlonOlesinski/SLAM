import time
import tqdm
import cv2

from shared_utils import TRACK_DB_PATH, read_cameras
from consensus_matching_dir.consensus_matching_localization import consensus_matching
from track_db_dir.track_db import TrackDB


def create_track_db(track_db_path):
    """
    Create a track db from the images in the dataset, using the localization algorithm.
    :param track_db_path: path to save the track db to.
    """
    ITERATIONS = 2559
    feature_detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k, left0_extrinsic, right0_extrinsic = read_cameras()
    prev_kp_left1, prev_desc_left1, prev_kp_right1, prev_desc_right1, prev_frame1_matches = None, None, None, None, None
    start = time.time()
    track_db = TrackDB()

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

if __name__ == '__main__':
    create_track_db(TRACK_DB_PATH)
