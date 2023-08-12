import matplotlib.pyplot as plt
import numpy as np
import tqdm
import cv2

from track_db_dir.track_db import TrackDB
from run.create_track_db import create_track_db
from shared_utils import TRACK_DB_PATH, RESULTS_PATH, relative_poses_of_kfs_from_bundle,\
    plot_trajectory_from_above, read_cameras, compose_affine_transformations, calculate_inverse_of_R_t
from bundle_adjustment_dir.bundle_adjustment import BundleAdjustment
from loop_closure_dir.pose_graph import create_pose_graph_from_ba
from consensus_matching_dir.utils import project
from shared_utils import yield_sequence_length_of_gt_trajectory, global_R_ti_to_R_tj, \
get_gt_left_camera_matrices, yield_sequence_length_of_gt_trajectory_by_kf

from run.utils import plot_trajectory_and_landmarks_from_above

FIRST_LOCATION_SIGMA = np.array([0.02,
                                    0.002,
                                    0.002,
                                    1,
                                    0.01,
                                    1])

def track_db_statistics(res_dir):
    """
    print the total number of tracks, number of frames, means track length and mean number of frame
     links of the track database. In addition, plot a graph of the number of matches per frame and
     a graph of the percentage of inliers per frame.
    """
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    num_of_tracks, num_of_frames, mean_track_length, mean_num_of_frame_links = track_db.get_statistics()
    print('number of tracks: {}'.format(num_of_tracks))
    print('number of frames: {}'.format(num_of_frames))
    print('mean track length: {}'.format(mean_track_length))
    print('mean number of frame links: {}'.format(mean_num_of_frame_links))

    # plot the number of matches per frame
    plot_path = res_dir + 'num_of_matches_per_frame.png'
    track_db.plot_matches_per_frame(plot_path)
    track_db.plot_inliers_percentage_per_frame(res_dir + 'inliers_percentage_per_frame.png')

def plot_connectivity(plot_path):
    """
    plot the connectivity of the track database
    :param plot_path:  path to save the plot
    """
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    connectivities = []
    frames = list(range(2559))
    for frame in frames:
        connectivities.append(track_db.get_frame_connectivity(frame))

    # make the font large
    plt.rcParams.update({'font.size': 22})
    plt.plot(frames, connectivities)
    plt.xlabel('frame')
    plt.ylabel('outgoing tracks')
    plt.title('connectivity')
    plt.xlim(0, 2559)
    plt.yticks(np.arange(0, 700, 50))
    # make the plot wider
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    # draw the mean as a horizontal line
    plt.axhline(y=np.mean(connectivities), color='g', linestyle='-')
    plt.savefig(plot_path)
    plt.clf()

def plot_track_length_hist(plot_path):
    """
    plot the track length histogram of the track database
    :param plot_path: path to save the plot
    """
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    lengths = []
    for track in track_db.get_all_unique_tracks():
        lengths.append(len(track))

    # make the font large
    plt.rcParams.update({'font.size': 22})
    hist, bins = np.histogram(lengths, bins=range(0, 137, 1))
    plt.plot(bins[:-1], hist)
    plt.xlabel('track length')
    plt.ylabel('number of tracks')
    plt.title('track length histogram')
    plt.xlim(0, 135)
    plt.xticks(np.arange(0, 135, 7))
    # make the plot wider
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(plot_path)
    plt.clf()

def plot_trajectory(plot_path):
    """
    plot the trajectory of the track database
    :param plot_path: path to save the plot
    """
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()
    poses_pnp = bundle_adjustment.get_all_camera_poses_global()

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()
    poses_bundle_adjustment = bundle_adjustment.get_all_camera_poses_global()

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    keyframe_frame_indices = pose_graph.get_keyframe_frame_indices()
    pose_graph.optimize()
    poses_loop_closure = pose_graph.get_optimized_poses()

    trajectories = [(poses_pnp, 'PnP'), (poses_bundle_adjustment, 'BA'), (poses_loop_closure, 'LC')]
    plot_trajectory_from_above(trajectories, plot_path, title='Trajectory Comparison',
                               key_frames_to_frames=keyframe_frame_indices)

def plot_optimization_error(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    mean_error_per_bundle_before_opt, median_error_per_bundle_before_opt, \
    mean_error_per_bundle_after_opt, median_error_per_bundle_after_opt = \
    bundle_adjustment.run_local_adjustments(return_error_per_bundle=True)

    plt.plot(np.log(mean_error_per_bundle_before_opt), label='before')
    plt.plot(np.log(mean_error_per_bundle_after_opt), label='after')
    plt.xlabel('keyframe')
    plt.ylabel('mean factor error (log scale)')
    plt.title('mean factor error per keyframe before and after optimization')
    plt.legend()
    plt.savefig(plot_dir_path + 'mean_factor_error_per_keyframe.png')
    plt.clf()

    plt.plot(median_error_per_bundle_before_opt, label='before')
    plt.plot(median_error_per_bundle_after_opt, label='after')
    plt.xlabel('keyframe')
    plt.ylabel('median projection error')
    plt.title('median projection error per keyframe before and after optimization')
    plt.legend()
    plt.savefig(plot_dir_path + 'median_projection_error_per_keyframe.png')
    plt.clf()


def plot_median_projection_error_pnp(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    k, _, right0_camera = read_cameras()
    distance_error_dict = {} # key: distance, value: 2 lists of errors, left and right
    for track in tqdm.tqdm(track_db.get_all_tracks_at_percentile_track_length(30)):
        p3d = track_db.triangulate_from_last_frame(track)
        _, features = track.get_consecutive_frame_ids_and_locations(len(track))
        for i in range(len(features) - 1):
            left_R_t = track_db.get_global_R_t(track.get_frame_from_relative_index(i))
            right_R_t = compose_affine_transformations(right0_camera, left_R_t)
            left_projected_point = project(p3d, k @ left_R_t)
            right_projected_point = project(p3d, k @ right_R_t)
            left_error = np.linalg.norm(features[i][0] - left_projected_point)
            right_error = np.linalg.norm(features[i][1] - right_projected_point)
            distance = len(track) - i - 1
            # truncate the distance to the nearest 20 for a better comparison with the bundle adjustment graph
            if distance > 20:
                continue
            if distance not in distance_error_dict:
                distance_error_dict[distance] = ([],[])
            distance_error_dict[distance][0].append(left_error)
            distance_error_dict[distance][1].append(right_error)

    # calculate the median error for each distance
    left_median_error_per_distance = []
    right_median_error_per_distance = []
    for distance in distance_error_dict:
        left_median_error_per_distance.append((distance, np.median(distance_error_dict[distance][0])))
        right_median_error_per_distance.append((distance, np.median(distance_error_dict[distance][1])))
    left_median_error_per_distance.sort(key=lambda x: x[0])
    right_median_error_per_distance.sort(key=lambda x: x[0])

    # plot the medians against the distances
    plt.plot([x[0] for x in left_median_error_per_distance], [x[1] for x in
                                                              left_median_error_per_distance], label='left')
    plt.plot([x[0] for x in right_median_error_per_distance], [x[1] for x in
                                                                right_median_error_per_distance], label='right')
    plt.xlabel('distance')
    plt.ylabel('median error')
    plt.legend()
    plt.title('PnP median projection error vs distance from triangulation frame')
    plt.savefig(plot_dir_path)
    plt.clf()

def plot_median_projection_error_bundle_adjustment(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    distance_error_dict = bundle_adjustment.get_projection_errors_and_distances(min_distance=2)
    median_error_per_distance = []
    for distance in distance_error_dict:
        median_error_per_distance.append((distance, np.median(distance_error_dict[distance])))
    median_error_per_distance.sort(key=lambda x: x[0])
    plt.plot([x[0] for x in median_error_per_distance], [x[1] for x in median_error_per_distance])
    plt.xlabel('distance')
    plt.ylabel('median error')
    plt.ylim(0, 3)
    plt.title('bundle adjustment median projection error vs distance from the first frame')
    plt.savefig(plot_dir_path + 'median_projection_error_bundle_adjustment_per_distance.png')
    plt.clf()

    # plot the histogram of distances
    x = distance_error_dict.keys()
    y = [len(distance_error_dict[xi]) for xi in x]
    plt.bar(x, y)
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.title('bundle adjustment histogram of distances')
    plt.savefig(plot_dir_path + 'bundle_adjustment_histogram_of_distances.png')
    plt.clf()

def plot_median_factor_error_pnp_and_bundle_adjustment(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    pnp_distance_error_dict = bundle_adjustment.get_factor_errors(min_distance=5, reference_frame=-1)
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment_distance_error_dict = bundle_adjustment.get_factor_errors(min_distance=5, reference_frame=0)
    pnp_median_error_per_distance = []
    bundle_adjustment_median_error_per_distance = []
    for distance in pnp_distance_error_dict:
        pnp_median_error_per_distance.append((distance, np.median(pnp_distance_error_dict[distance])))
        bundle_adjustment_median_error_per_distance.append((distance, np.median(bundle_adjustment_distance_error_dict[distance])))
    pnp_median_error_per_distance.sort(key=lambda x: x[0])
    bundle_adjustment_median_error_per_distance.sort(key=lambda x: x[0])
    plt.plot([x[0] for x in pnp_median_error_per_distance], [x[1] for x in pnp_median_error_per_distance], label='pnp')
    plt.plot([x[0] for x in bundle_adjustment_median_error_per_distance], [x[1] for x in bundle_adjustment_median_error_per_distance], label='bundle adjustment')
    plt.xlabel('distance')
    plt.ylabel('median error')
    plt.legend()
    plt.title('median factor error vs distance - pnp and bundle adjustment')
    plt.savefig(plot_dir_path + 'median_factor_error_pnp_and_bundle_adjustment.png')
    plt.clf()

def absolute_estimation_error(plot_dir_path):
    video_len = 2560
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    gt_global_matrices = get_gt_left_camera_matrices(video_len)

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()
    pnp_global_poses = bundle_adjustment.get_all_camera_poses_global()

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()
    ba_global_poses = bundle_adjustment.get_all_camera_poses_global()

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    pose_graph.optimize()
    keyframe_frame_indices = pose_graph.get_keyframe_frame_indices()
    lc_global_poses = pose_graph.get_optimized_poses()

    pnp_x_error = []
    pnp_y_error = []
    pnp_z_error = []
    pnp_norm_error = []
    pnp_angle_error = []

    ba_x_error = []
    ba_y_error = []
    ba_z_error = []
    ba_norm_error = []
    ba_angle_error = []

    lc_x_error = []
    lc_y_error = []
    lc_z_error = []
    lc_norm_error = []
    lc_angle_error = []

    for frame_index in range(video_len):

        gt_R_t = gt_global_matrices[frame_index]
        pnp_pose = pnp_global_poses[frame_index]
        ba_pose = ba_global_poses[frame_index]

        gt_inv_R_t = calculate_inverse_of_R_t(gt_R_t)

        pnp_x_error.append(np.linalg.norm(gt_inv_R_t[0, 3] - pnp_pose.x()))
        pnp_y_error.append(np.linalg.norm(gt_inv_R_t[1, 3] - pnp_pose.y()))
        pnp_z_error.append(np.linalg.norm(gt_inv_R_t[2, 3] - pnp_pose.z()))
        pnp_norm_error.append(np.linalg.norm(gt_inv_R_t[0:3, 3] - pnp_pose.translation()))
        pnp_angle_error.append(
            np.linalg.norm(cv2.Rodrigues(gt_R_t[:3, :3] @ pnp_pose.rotation().matrix())[0]))

        ba_x_error.append(np.linalg.norm(gt_inv_R_t[0, 3] - ba_pose.x()))
        ba_y_error.append(np.linalg.norm(gt_inv_R_t[1, 3] - ba_pose.y()))
        ba_z_error.append(np.linalg.norm(gt_inv_R_t[2, 3] - ba_pose.z()))
        ba_norm_error.append(np.linalg.norm(gt_inv_R_t[0:3, 3] - ba_pose.translation()))
        ba_angle_error.append(
            np.linalg.norm(cv2.Rodrigues(gt_R_t[:3, :3] @ ba_pose.rotation().matrix())[0]))

    for kf in range(len(lc_global_poses)):
        frame_index = keyframe_frame_indices[kf]
        gt_R_t = gt_global_matrices[frame_index]
        gt_inv_R_t = calculate_inverse_of_R_t(gt_R_t)
        lc_pose = lc_global_poses[kf]

        lc_x_error.append(np.linalg.norm(gt_inv_R_t[0, 3] - lc_pose.x()))
        lc_y_error.append(np.linalg.norm(gt_inv_R_t[1, 3] - lc_pose.y()))
        lc_z_error.append(np.linalg.norm(gt_inv_R_t[2, 3] - lc_pose.z()))
        lc_norm_error.append(np.linalg.norm(gt_inv_R_t[0:3, 3] - lc_pose.translation()))
        lc_angle_error.append(
            np.linalg.norm(cv2.Rodrigues(gt_R_t[:3, :3] @ lc_pose.rotation().matrix())[0]))

    y_max_loc = max(max(pnp_norm_error), max(ba_norm_error), max(lc_norm_error))
    y_max_angle = max(max(pnp_angle_error), max(ba_angle_error), max(lc_angle_error))

    plt.clf()
    plt.plot(pnp_x_error, label='x')
    plt.plot(pnp_y_error, label='y')
    plt.plot(pnp_z_error, label='z')
    plt.plot(pnp_norm_error, label='norm')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.legend()
    plt.title('pnp absolute location error')
    plt.ylim(0, y_max_loc)
    plt.savefig(plot_dir_path + 'absolute_location_error_pnp.png')
    plt.clf()

    plt.plot(pnp_angle_error, label='angle')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.title('pnp absolute angle error')
    plt.ylim(0, y_max_angle)
    plt.savefig(plot_dir_path + 'absolute_angle_error_pnp.png')
    plt.clf()

    plt.plot(ba_x_error, label='x')
    plt.plot(ba_y_error, label='y')
    plt.plot(ba_z_error, label='z')
    plt.plot(ba_norm_error, label='norm')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.legend()
    plt.title('ba absolute location error')
    plt.ylim(0, y_max_loc)
    plt.savefig(plot_dir_path + 'absolute_location_error_ba.png')
    plt.clf()

    plt.plot(ba_angle_error, label='angle')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.title('ba absolute angle error')
    plt.ylim(0, y_max_angle)
    plt.savefig(plot_dir_path + 'absolute_angle_error_ba.png')
    plt.clf()


    plt.plot(keyframe_frame_indices, lc_x_error, label='x')
    plt.plot(keyframe_frame_indices, lc_y_error, label='y')
    plt.plot(keyframe_frame_indices, lc_z_error, label='z')
    plt.plot(keyframe_frame_indices, lc_norm_error, label='norm')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.legend()
    plt.title('lc absolute location error')
    plt.ylim(0, y_max_loc)
    plt.savefig(plot_dir_path + 'absolute_location_error_lc.png')
    plt.clf()

    plt.plot(keyframe_frame_indices, lc_angle_error, label='angle')
    plt.xlabel('frame index')
    plt.ylabel('error')
    plt.title('lc absolute angle error')
    plt.ylim(0, y_max_angle)
    plt.savefig(plot_dir_path + 'absolute_angle_error_lc.png')
    plt.clf()

def relative_estimation_error(plot_dir_path):
    video_len = 2560
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    gt_global_matrices = get_gt_left_camera_matrices(video_len)

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()
    pnp_global_poses = bundle_adjustment.get_all_camera_poses_global()

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()
    ba_global_poses = bundle_adjustment.get_all_camera_poses_global()

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    kf_frame_indices = pose_graph.get_keyframe_frame_indices()
    pose_graph.optimize()
    lc_global_poses = pose_graph.get_optimized_poses()


    pnp_location_error_list = []
    ba_location_error_list = []
    lc_location_error_list = []

    pnp_angle_error_list = []
    ba_angle_error_list = []
    lc_angle_error_list = []

    for seq_size in [100, 300, 800]:
        pnp_location_error_list.append([])
        ba_location_error_list.append([])
        pnp_angle_error_list.append([])
        ba_angle_error_list.append([])

        trajectory_length_generator = yield_sequence_length_of_gt_trajectory(seq_size)
        for i in tqdm.tqdm(range(video_len - seq_size + 1)):
            gt_trajectory_length = next(trajectory_length_generator)
            last_mat_idx = i + seq_size - 1
            # compute translation and rotation for gt
            gt_matrix_between = global_R_ti_to_R_tj(
                gt_global_matrices[last_mat_idx], gt_global_matrices[i]) # note that the order is consistent with the gtsam convention.
            gt_translation = gt_matrix_between[:3, 3]
            gt_rotation = gt_matrix_between[:3, :3]
            # compute translation and rotation for pnp
            pnp_pose_between = pnp_global_poses[i].between(pnp_global_poses[last_mat_idx])
            pnp_translation = pnp_pose_between.translation()
            pnp_rotation = pnp_pose_between.rotation().matrix()
            # compute translation and rotation for ba
            ba_between = ba_global_poses[i].between(ba_global_poses[last_mat_idx])
            ba_translation = ba_between.translation()
            ba_rotation = ba_between.rotation().matrix()

            # compute location error
            pnp_location_error_list[-1].append(np.linalg.norm(gt_translation - pnp_translation)/gt_trajectory_length)
            ba_location_error_list[-1].append(np.linalg.norm(gt_translation - ba_translation)/gt_trajectory_length)

            # compute angle error, by computing the norm of the Rodrigues representation of R.T * Q
            pnp_angle_error_list[-1].append(np.linalg.norm(cv2.Rodrigues(gt_rotation.T @ pnp_rotation)[0])/gt_trajectory_length)
            ba_angle_error_list[-1].append(np.linalg.norm(cv2.Rodrigues(gt_rotation.T @ ba_rotation)[0])/gt_trajectory_length)

    for seq_size_in_kf in [20, 60, 160]:
        lc_location_error_list.append([])
        lc_angle_error_list.append([])
        trajectory_length_generator = yield_sequence_length_of_gt_trajectory_by_kf(kf_frame_indices, seq_size_in_kf)

        for i in tqdm.tqdm(range(len(kf_frame_indices)-seq_size_in_kf+1)):
            start_kf_index = i
            end_kf_index = i + seq_size_in_kf - 1
            start_frame_index = kf_frame_indices[start_kf_index]
            end_frame_index = kf_frame_indices[end_kf_index]
            gt_trajectory_length = next(trajectory_length_generator)
            # compute translation and rotation for gt
            gt_matrix_between = global_R_ti_to_R_tj(
                gt_global_matrices[end_frame_index], gt_global_matrices[start_frame_index])
            gt_translation = gt_matrix_between[:3, 3]
            gt_rotation = gt_matrix_between[:3, :3]
            # compute translation and rotation for lc
            lc_between = lc_global_poses[start_kf_index].between(lc_global_poses[end_kf_index])
            lc_translation = lc_between.translation()
            lc_rotation = lc_between.rotation().matrix()
            # compute location error
            lc_location_error_list[-1].append(np.linalg.norm(gt_translation - lc_translation)/gt_trajectory_length)
            # compute angle error, by computing the norm of the Rodrigues representation of R.T * Q
            lc_angle_error_list[-1].append(np.linalg.norm(cv2.Rodrigues(gt_rotation.T @ lc_rotation)[0])/gt_trajectory_length)


    ylim_location = np.max(
        [np.max([np.max(pnp_location_error_list[i]) for i in range(len(pnp_location_error_list))]),
         np.max([np.max(ba_location_error_list[i])for i in range(len(ba_location_error_list))]),
         np.max([np.max(lc_location_error_list[i])for i in range(len(lc_location_error_list))])])

    plt.clf()
    # plot the pnp translation errors (one plot per sequence length)
    for i, seq_size in zip(range(len(pnp_location_error_list)), [100, 300, 800]):
        mean_error = np.round(np.mean(pnp_location_error_list[i]),4)
        plt.plot(pnp_location_error_list[i], label=f'{seq_size}, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative translation error')
    plt.legend()
    plt.title('relative translation error pnp')
    plt.ylim([0, ylim_location])
    plt.savefig(plot_dir_path + 'relative_translation_error_pnp.png')
    plt.clf()

    # plot the bundle adjustment translation errors (one plot per sequence length)
    for i, seq_size in zip(range(len(ba_location_error_list)), [100, 300, 800]):
        mean_error = np.round(np.mean(ba_location_error_list[i]),4)
        plt.plot(ba_location_error_list[i], label=f'{seq_size}, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative translation error')
    plt.legend()
    plt.title('relative translation error bundle adjustment')
    plt.ylim([0, ylim_location])
    plt.savefig(plot_dir_path + 'relative_translation_error_bundle_adjustment.png')
    plt.clf()

    # plot the loop closure translation errors (one plot per sequence length)
    for i, seq_size in zip(range(len(lc_location_error_list)), [20, 60, 160]):
        mean_error = np.round(np.mean(lc_location_error_list[i]),4)
        plt.plot(kf_frame_indices[:-seq_size+1], lc_location_error_list[i], label=f'{seq_size} keyframes, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative translation error')
    plt.legend()
    plt.title('relative translation error loop closure')
    plt.ylim([0, ylim_location])
    plt.savefig(plot_dir_path + 'relative_translation_error_loop_closure.png')
    plt.clf()

    ylim_angle = np.max(
        [np.max([np.max(pnp_angle_error_list[i]) for i in range(len(pnp_angle_error_list))]),
         np.max([np.max(ba_angle_error_list[i])for i in range(len(ba_angle_error_list))]),
         np.max([np.max(lc_angle_error_list[i])for i in range(len(lc_angle_error_list))])])

    # plot the pnp angle errors (one plot per sequence length)
    for i, seq_size in zip(range(len(pnp_angle_error_list)), [100, 300, 800]):
        mean_error = np.round(np.mean(pnp_angle_error_list[i]),4)
        plt.plot(pnp_angle_error_list[i], label=f'{seq_size}, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative angle error')
    plt.legend()
    plt.title('relative angle error pnp')
    plt.ylim([0, ylim_angle])
    plt.savefig(plot_dir_path + 'relative_angle_error_pnp.png')
    plt.clf()

    # plot the bundle adjustment angle errors (one plot per sequence length)
    for i, seq_size in zip(range(len(ba_angle_error_list)), [100, 300, 800]):
        mean_error = np.round(np.mean(ba_angle_error_list[i]), 4)
        plt.plot(ba_angle_error_list[i], label=f'{seq_size}, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative angle error')
    plt.legend()
    plt.title('relative angle error bundle adjustment')
    plt.ylim([0, ylim_angle])
    plt.savefig(plot_dir_path + 'relative_angle_error_bundle_adjustment.png')
    plt.clf()

    # plot the loop closure angle errors (one plot per sequence length)
    for i, seq_size in zip(range(len(lc_angle_error_list)), [20, 60, 160]):
        mean_error = np.round(np.mean(lc_angle_error_list[i]), 4)
        plt.plot(kf_frame_indices[:-seq_size+1], lc_angle_error_list[i], label=f'{seq_size} keyframes, average error: {mean_error}')
    plt.xlabel('frame')
    plt.ylabel('relative angle error')
    plt.legend()
    plt.title('relative angle error loop closure')
    plt.ylim([0, ylim_angle])
    plt.savefig(plot_dir_path + 'relative_angle_error_loop_closure.png')
    plt.clf()

def lc_matches_num_and_inliers_percent():
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = FIRST_LOCATION_SIGMA

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    pose_graph.optimize()

    lc_matches_num = pose_graph.get_lc_matches_num()
    lc_inliers_percent = pose_graph.get_lc_inliers_percent()

    print(f'min number of matches: {np.round(np.min(lc_matches_num),2)}')
    print(f'max number of matches: {np.round(np.max(lc_matches_num),2)}')
    print(f'average number of matches: {np.round(np.mean(lc_matches_num),2)}')
    print(f'min inliers percent: {np.round(np.min(lc_inliers_percent),2)}%')
    print(f'max inliers percent: {np.round(np.max(lc_inliers_percent),2)}%')
    print(f'average inliers percent: {np.round(np.mean(lc_inliers_percent),2)}%')


def plot_uncertainties(plot_dir_path):
    db_path = TRACK_DB_PATH
    track_db = TrackDB.deserialize(db_path)
    first_location_prior_sigma = FIRST_LOCATION_SIGMA
    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()

    relative_poses = []
    pose_graph_covs = []
    keyframe_indices = []
    for bundle in bundle_adjustment.bundles:
        relative_pose, cov = relative_poses_of_kfs_from_bundle(bundle, bundle.start_kf_idx,
                                                               bundle.end_kf_idx)
        relative_poses.append(relative_pose)
        pose_graph_covs.append(cov)
        keyframe_indices.append(bundle.end_kf_idx)

    pose_graph = create_pose_graph_from_ba(bundle_adjustment)
    pose_graph.optimize()
    opt_loc_uncertainties, opt_ori_uncertainties = pose_graph.get_uncertainties(pose_graph.get_optimized_values())
    pose_graph = create_pose_graph_from_ba(bundle_adjustment, stop_closure_index=0)
    init_loc_uncertainties, init_ori_uncertainties = pose_graph.get_uncertainties(pose_graph.get_initial_values())
    frame_indices = pose_graph.get_keyframe_frame_indices()[1:]
    plt.clf()
    plt.plot(frame_indices, init_loc_uncertainties, label='before')
    plt.plot(frame_indices, opt_loc_uncertainties, label='after')
    plt.title('Location uncertainties before/after loop closure')
    plt.xlabel('keyframe index')
    plt.ylabel('location uncertainty')
    # set the scale to logaritmic
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(plot_dir_path + 'loc_uncertainties.png')
    plt.clf()

    plt.plot(frame_indices, init_ori_uncertainties, label='before')
    plt.plot(frame_indices, opt_ori_uncertainties, label='after')
    plt.title('Orientation uncertainties before/after loop closure')
    plt.xlabel('keyframe index')
    plt.ylabel('orientation uncertainty')
    # set the scale to logaritmic
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(plot_dir_path + 'ori_uncertainties.png')
    plt.clf()




if __name__ == '__main__':
    try:
        track_db = TrackDB.deserialize(TRACK_DB_PATH)
    except FileNotFoundError:
        create_track_db(TRACK_DB_PATH)

    # track_db_statistics(RESULTS_PATH)
    # plot_connectivity(RESULTS_PATH + 'connectivity.png')
    # plot_track_length_hist(RESULTS_PATH + 'track_length_hist.png')
    # plot_trajectory(RESULTS_PATH + 'trajectory.png')
    # plot_optimization_error(RESULTS_PATH)
    # plot_median_projection_error_pnp(RESULTS_PATH + 'median_projection_error_per_distance_pnp.png')
    # plot_median_projection_error_bundle_adjustment(RESULTS_PATH)
    # plot_median_factor_error_pnp_and_bundle_adjustment(RESULTS_PATH)
    # absolute_estimation_error(RESULTS_PATH)
    # relative_estimation_error(RESULTS_PATH)
    # lc_matches_num_and_inliers_percent()
    plot_uncertainties(RESULTS_PATH)