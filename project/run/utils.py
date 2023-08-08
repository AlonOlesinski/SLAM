import gtsam
import matplotlib.pyplot as plt

from shared_utils import get_gt_left_camera_matrices, camera_location_from_extrinsic_matrix, RESULTS_PATH


def plot_trajectory_and_landmarks_from_above(cameras_optimized: list[gtsam.Pose3],
                                             landmarks: list[gtsam.Point3],
                                             cameras_initial_estimate: list[gtsam.Pose3] = None,
                                             plot_ground_truth=False,
                                             file_name='trajectory_and_landmarks_from_above.png',
                                             title='trajectory and landmarks from above',
                                             key_frames_to_frames=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    # make the x and y axes equal so that the plot is not distorted:
    # ax.set_aspect('equal', adjustable='box')

    if len(landmarks) != 0:
        landmark_x_z = [(landmark[0], landmark[2]) for landmark in landmarks]
        ax.scatter(*zip(*landmark_x_z), c='orange', s=0.1, label='landmark')

    if cameras_initial_estimate is not None:
        camera_x = [camera.x() for camera in cameras_initial_estimate]
        camera_z = [camera.z() for camera in cameras_initial_estimate]
        ax.scatter(camera_x, camera_z, c='blue', s=0.5, label='initial estimate')

    camera_x = [camera.x() for camera in cameras_optimized]
    camera_z = [camera.z() for camera in cameras_optimized]
    ax.scatter(camera_x, camera_z, c='r', s=0.5, label='optimized')

    if plot_ground_truth:
        print(len(camera_x))
        gt_left_extrinsics = get_gt_left_camera_matrices(key_frames_to_frames)
        gt_locations = [camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                        extrinsic_matrix in gt_left_extrinsics]
        camera_x = [location[0] for location in gt_locations]
        camera_z = [location[2] for location in gt_locations]
        ax.scatter(camera_x, camera_z, c='green',s=0.5, label='ground truth')



    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(title)
    ax.legend()
    plt.savefig(RESULTS_PATH + file_name, dpi=300)
    plt.clf()