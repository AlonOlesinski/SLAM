import numpy as np
import cv2
import matplotlib.pyplot as plt
import gtsam

DATA_PATH = '/mnt/c/Users/alono/OneDrive/desktop/studies/VAN_ex/dataset/sequences/05/'
RESULTS_PATH = '/mnt/c/Users/alono/OneDrive/desktop/studies/VAN_ex/docs/project/'
GT_LEFT_CAMERA_PATH = '/mnt/c/Users/alono/OneDrive/desktop/studies/VAN_ex/dataset/poses/05.txt'
PROJECT_PATH = '/mnt/c/Users/alono/OneDrive/desktop/studies/VAN_ex/code/project/'
TRACK_DB_PATH = '/mnt/c/Users/alono/OneDrive/desktop/studies/VAN_ex/code/project/track_db.pkl'

def read_images(idx):
    """
    read images from the dataset
    """
    img_name = '{:06d}.png'.format(idx)
    left_img = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    right_img = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return left_img, right_img

def read_cameras():
    """
    read the camera matrices from the dataset
    """
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2

def get_gt_left_camera_matrices(lines_num):
    """
    get the ground truth camera positions
    :return: the ground truth camera positions
    """
    with open(GT_LEFT_CAMERA_PATH) as f:
        lines = f.readlines()
    res = []
    if type(lines_num) == int:
        lines_num = [l for l in range(lines_num)]
    for i in lines_num:
        camera_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        res.append(camera_mat)
    return res

def camera_location_from_extrinsic_matrix(R_t):
    """
    The input is n+1 camera matrix [R|t], which transforms the n'th coordinate system to the
    n+1 coordinate system. The camera location (in the n'th coordinate system) is the negative of the inverse (=transpose)
    of R,  as seen in question 2.3.
    :param R_t: 3x4 matrix
    :return: camera location in world coordinate, according to the n'th coordinate system.
    """
    return (-R_t[:3, :3].T).dot(R_t[:3, 3])


def compose_affine_transformations(outer_affine, inner_affine):
    """
    compute the composition of the affine transformation represented by camera1 and camera2. The result transform takes
    first transforms to the coordinate system of camera1 and then transforms to the coordinate system of camera2.
    :param outer_affine: 3x4 matrix
    :param inner_affine: 3x4 matrix
    :return: 3x4 matrix
    """
    # we use the fact that [R2|t2] @ append_row([R1|t1], e4) = [R2@R1|t2+R2@t1]
    temp_mat = np.append(inner_affine, np.array([[0, 0, 0, 1]]), axis=0)
    return outer_affine @ temp_mat

def calculate_inverse_of_R_t(R_t):
    """
    calculate the inverse of R_t
    :param R_t: R_t
    :return: inverse of R_t
    """
    R = R_t[:3, :3]
    t = R_t[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return np.hstack((R_inv, t_inv.reshape(-1, 1)))


def gtsam_calib_mat():
    """
    convert the calibration matrix to gtsam format
    :return: gtsam calibration matrix
    """
    K, _, right_R_t = read_cameras()
    fx = K[0, 0]
    fy = K[1, 1]
    s = K[0, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    b = -right_R_t[0, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, b)


def gtsam_pose_from_global_R_t(R_t):
    """
    convert the global R_t to gtsam pose
    :param R_t: R_t
    :return: gtsam pose
    """
    R_t_inv = calculate_inverse_of_R_t(R_t)
    return gtsam.Pose3(gtsam.Rot3(R_t_inv[:3, :3]), gtsam.Point3(R_t_inv[:3, 3]))


def global_R_ti_to_R_tj(R_ti, R_tj):
    """
    Return the relative matrix between two cameras, given the global matrices of the two cameras.
    :param R_ti: as returned by TrackDB.get_global_R_t (W -> Ci)
    :param R_tj: as returned by TrackDB.get_global_R_t (W -> Ci)
    :return: R_ij (Ci -> Cj)
    """
    R_ti_inv = calculate_inverse_of_R_t(R_ti)
    return compose_affine_transformations(R_ti_inv, R_tj)


def bundle_length_hist(bundle_lengths):
    """
    plot the histogram of the bundle lengths
    """
    plt.hist(bundle_lengths, bins=16)
    plt.show()
    plt.clf()

def relative_poses_of_kfs_from_bundle(bundle, idx1, idx2):
    """
    Get the relative pose between two keyframes from a bundle
    :param bundle: bundle object
    :param idx1: index of the first keyframe
    :param idx2: index of the second keyframe
    :return: relative pose between the two keyframes
    """
    return relative_poses_of_kfs(bundle.graph, bundle.transformed_values, idx1, idx2)

def relative_poses_of_kfs(graph, values, idx1, idx2):
    """
    Get the relative pose between two keyframes
    :param graph: gtsam graph
    :param values: gtsam values
    :param idx1: index of the first keyframe
    :param idx2: index of the second keyframe
    :return: relative pose between the two keyframes
    """
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


def plot_trajectory_from_above(trajectories: list[tuple[list, str]],
                                             file_name,
                                             title='trajectory from above',
                                             key_frames_to_frames=None):
    """
    plot the trajectories from above (x-z plane) and save the plot to file_name. trajectories is
    a list of tuples (trajectory, trajectory_label) where trajectory is a list gtsam poses.
    :param trajectories: list of tuples (trajectory, trajectory_label)
    :param file_name: file name to save the plot to
    :param title: title of the plot
    :param key_frames_to_frames: if the length of the trajectories is not the same as the length of the
    ground truth, we need to provide a mapping from key frames to frames
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot()



    # plot every trajectory:
    for trajectory, trajectory_label in trajectories:
        x = [pose.x() for pose in trajectory]
        z = [pose.z() for pose in trajectory]
        ax.plot(x, z, label=trajectory_label)

    # plot gt:
    gt_left_extrinsics = get_gt_left_camera_matrices(key_frames_to_frames)
    gt_locations = [camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                    extrinsic_matrix in gt_left_extrinsics]
    camera_x = [location[0] for location in gt_locations]
    camera_z = [location[2] for location in gt_locations]
    ax.plot(camera_x, camera_z,label='GT', color='black')


    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.savefig(file_name, dpi=300)
    plt.clf()

def yield_sequence_length_of_gt_trajectory(sequence_size, video_len=2560):
    """
    compute the total length of the gt trajectory for each sequence of size sequence_size
    :param sequence_size: size of the sequence (int)
    :param video_len: length of the video (int)
    :return: generator of the total length of the gt trajectory for each sequence of size sequence_size
    """
    gt_left_extrinsics = get_gt_left_camera_matrices(video_len)
    gt_locations = [camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                    extrinsic_matrix in gt_left_extrinsics]
    norms = [np.linalg.norm(gt_locations[i]-gt_locations[i+1]) for i in range(video_len-1)]
    for j in range(0, video_len - sequence_size + 1):
        yield np.sum(norms[j:j+sequence_size])

def yield_sequence_length_of_gt_trajectory_by_kf(kf_frame_list, sequence_size_in_kfs, vidio_len=2560):
    """
    compute the total length of the gt trajectory for each sequence of size sequence_size (by keyframes).
    :param kf_frame_list: list that maps between keyframes and frames
    :param sequence_size_in_kfs: size of the sequence in number of keyframes (int)
    :param video_len: length of the video (int)
    :return: generator of the total length of the gt trajectory for each sequence of size sequence_size
    """
    gt_left_extrinsics = get_gt_left_camera_matrices(vidio_len)
    gt_locations = [camera_location_from_extrinsic_matrix(extrinsic_matrix) for
                    extrinsic_matrix in gt_left_extrinsics]
    norms = [np.linalg.norm(gt_locations[i]-gt_locations[i+1]) for i in range(len(gt_locations)-1)]
    for j in range(0, len(kf_frame_list) - sequence_size_in_kfs + 1):
        yield sum(norms[kf_frame_list[j]:kf_frame_list[j + sequence_size_in_kfs-1]])






if __name__ == "__main__":
    for tot_len in yield_sequence_length_of_gt_trajectory(20, 500):
        print(tot_len)