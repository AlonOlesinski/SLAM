import matplotlib.pyplot as plt
import numpy as np
import tqdm

from track_db_dir.track_db import TrackDB
from run.create_track_db import create_track_db
from shared_utils import TRACK_DB_PATH, RESULTS_PATH, relative_poses_of_kfs_from_bundle,\
    plot_trajectory_from_above, read_cameras, compose_affine_transformations
from bundle_adjustment_dir.bundle_adjustment import BundleAdjustment
from loop_closure_dir.pose_graph import PoseGraph
from consensus_matching_dir.utils import project


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
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.set_global_coordinates()
    poses_pnp = bundle_adjustment.get_all_camera_poses_global()

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment.set_global_coordinates()
    poses_bundle_adjustment = bundle_adjustment.get_all_camera_poses_global()

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
    poses_loop_closure = pose_graph.get_optimized_poses()

    trajectories = [(poses_pnp, 'PnP'), (poses_bundle_adjustment, 'BA'), (poses_loop_closure, 'LC')]
    plot_trajectory_from_above(trajectories, plot_path, title='Trajectory Comparison',
                               key_frames_to_frames=keyframe_indices)

def plot_optimization_error(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    mean_error_per_bundle_before_opt, median_error_per_bundle_before_opt, \
    mean_error_per_bundle_after_opt, median_error_per_bundle_after_opt = \
    bundle_adjustment.run_local_adjustments(return_error_per_bundle=True)

    plt.plot(np.log(mean_error_per_bundle_before_opt), label='before')
    plt.plot(np.log(mean_error_per_bundle_after_opt), label='after')
    plt.xlabel('keyframe')
    plt.ylabel('mean error (log scale)')
    plt.title('mean error per keyframe before and after optimization')
    plt.legend()
    plt.savefig(plot_dir_path + 'mean_error_per_keyframe.png')
    plt.clf()

    plt.plot(np.log(median_error_per_bundle_before_opt), label='before')
    plt.plot(np.log(median_error_per_bundle_after_opt), label='after')
    plt.xlabel('keyframe')
    plt.ylabel('median error (log scale)')
    plt.title('median error per keyframe before and after optimization')
    plt.legend()
    plt.savefig(plot_dir_path + 'median_error_per_keyframe.png')
    plt.clf()


def plot_median_reprojection_error_pnp(plot_dir_path):
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
    plt.title('PnP median reprojection error vs distance')
    plt.savefig(plot_dir_path)
    plt.clf()

def plot_median_reprojection_error_bundle_adjustment(plot_dir_path):
    track_db = TrackDB.deserialize(TRACK_DB_PATH)
    # perform bundle adjustment
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    bundle_adjustment.run_local_adjustments()
    distance_error_dict = bundle_adjustment.get_reprojection_errors(min_distance=2)
    median_error_per_distance = []
    for distance in distance_error_dict:
        median_error_per_distance.append((distance, np.median(distance_error_dict[distance])))
    median_error_per_distance.sort(key=lambda x: x[0])
    plt.plot([x[0] for x in median_error_per_distance], [x[1] for x in median_error_per_distance])
    plt.xlabel('distance')
    plt.ylabel('median error')
    plt.title('bundle adjustment median reprojection error vs distance')
    plt.savefig(plot_dir_path + 'median_reprojection_error_bundle_adjustment.png')
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
    first_location_prior_sigma = np.array([0.02,
                                           0.002,
                                           0.002,
                                           1,
                                           0.01,
                                           1])

    bundle_adjustment = BundleAdjustment(track_db, first_location_prior_sigma)
    bundle_adjustment.set_bundles()
    pnp_distance_error_dict = bundle_adjustment.get_factor_errors(min_distance=5)
    bundle_adjustment.run_local_adjustments()
    bundle_adjustment_distance_error_dict = bundle_adjustment.get_factor_errors(min_distance=5)
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







if __name__ == '__main__':
    try:
        track_db = TrackDB.deserialize(TRACK_DB_PATH)
    except FileNotFoundError:
        create_track_db(TRACK_DB_PATH)

    # plot_connectivity(RESULTS_PATH + 'connectivity.png')
    # plot_track_length_hist(RESULTS_PATH + 'track_length_hist.png')
    # plot_trajectory(RESULTS_PATH + 'trajectory.png')
    # plot_optimization_error(RESULTS_PATH)
    # plot_median_reprojection_error_pnp(RESULTS_PATH + 'median_reprojection_error_pnp.png')
    # plot_median_reprojection_error_bundle_adjustment(RESULTS_PATH)
    plot_median_factor_error_pnp_and_bundle_adjustment(RESULTS_PATH)