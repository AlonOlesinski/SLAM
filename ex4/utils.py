import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import imageio
import utils

DATA_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\dataset\sequences\05\\'
GT_LEFT_CAMERA_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\dataset\poses\05.txt'


def read_images(idx):
    """
    read images from the dataset
    """
    img_name = '{:06d}.png'.format(idx)
    left_img = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    right_img = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return left_img, right_img


def read_cameras():
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


#######################      plots for ex1      #######################


def set_figure_shape(images):
    """
    set the figure shape according to the number of images and detectors
    """
    if len(images) == 1:
        plt_shape = (1, 1)
    elif len(images) == 2:
        plt_shape = (2, 1)
    elif len(images) == 4:
        plt_shape = (2, 2)
    else:
        raise ValueError("The number of images must be 1,2 or 4")
    fig, axes = plt.subplots(*plt_shape)
    if len(images) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    return axes


def plot_kps(tracker, images, dest_path, print_desc=False):
    """
    plot images with keypoints
    :param tracker: tracker object
    :param images: list of images
    :param dest_path: path to save the plot
    :param print_desc: if True, print the first two descriptors of each image
    """
    axes = set_figure_shape(images)
    for i in range(len(images)):
        kp, desc = tracker.calculate_kps_and_descs(images[i])
        img_with_kp = cv2.drawKeypoints(images[i], kp, None, (0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        if print_desc:
            print(f'First descriptor of image number {i + 1}:')
            print(desc[0])
            print(f'Second descriptor of image number {i + 1}:')
            print(desc[1])

        axes[i].set_title(f'image {i + 1} with kp')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].imshow(img_with_kp)

    plt.savefig(dest_path)
    plt.clf()


def plot_matches(tracker, im1, im2, dest_path, matches_num=20):
    """
    plot matches between two images
    :param tracker: tracker object
    :param im1: first image
    :param im2: second image
    """
    kp1, desc1 = tracker.calculate_kps_and_descs(im1)
    kp2, desc2 = tracker.calculate_kps_and_descs(im2)
    matches = tracker.calculate_matches(desc1, desc2)[0]
    matches_img = cv2.drawMatches(im1, kp1, im2, kp2, matches[:matches_num], None, flags=2)
    # save image in dest_path
    cv2.imwrite(dest_path, matches_img)


def plot_false_negative_match(detector, matcher, im1, im2, dest_path):
    """
    code for 1.4.2 - show a false negative match.
    """
    """
        code for 1.4.2 - show a false negative match.
        """
    kp1, desc1 = detector.detectAndCompute(im1, None)
    kp2, desc2 = detector.detectAndCompute(im2, None)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    lower_ratio_space = np.linspace(0.2, 0.5, 30)
    upper_ratio_space = lower_ratio_space + 0.05
    for i, (m1, m2) in enumerate(matches):
        print(f'match number {i}')
        for low_ratio in lower_ratio_space:
            break_low_ratio_loop = False
            for high_ratio in upper_ratio_space:
                if m1.distance < high_ratio * m2.distance and m1.distance > low_ratio * m2.distance:
                    print(f'passed test with ratio {high_ratio} but failed with ratio {low_ratio}')
                    matches_img = cv2.drawMatches(im1, kp1, im2, kp2, [m1], None, flags=2)
                    # save the image
                    cv2.imwrite(dest_path, matches_img)
                    return


#######################      plots for ex2      #######################


def plot_deviation_histogram(deviations, dest_path):
    """
    plot a histogram of the deviations
    :param deviations: list of deviations
    """
    print(f'{round(100 * len(list(filter(lambda x: x > 2, deviations))) / len(deviations), 2)}% of the matches '
          f'deviate by more than 2 pixels')
    plt.hist(deviations, bins=max(deviations))
    plt.xlabel('Y axis deviation')
    plt.ylabel('Number of matches')
    plt.title('Y Axis Deviations Histogram')
    plt.savefig(dest_path)


def plot_geometric_rejection(tracker, im1, im2, dest_path):
    kp1, desc1 = tracker.calculate_kps_and_descs(im1)
    kp2, desc2 = tracker.calculate_kps_and_descs(im2)
    inliers, outliers = tracker.calculate_matches(desc1, desc2, kp1=kp1, kp2=kp2)
    print(f'number of discarded matches out of {len(inliers) + len(outliers)} matches: {len(outliers)}')

    inliers_kp1 = [kp1[m.queryIdx] for m in inliers]
    inliers_kp2 = [kp2[m.trainIdx] for m in inliers]
    inliers_kp = [inliers_kp1, inliers_kp2]
    outliers_kp1 = [kp1[m.queryIdx] for m in outliers]
    outliers_kp2 = [kp2[m.trainIdx] for m in outliers]
    outliers_kp = [outliers_kp1, outliers_kp2]

    # plot inliers in orange, outliers in cyan
    fig, axes = plt.subplots(2, 1)
    for ax, im, i, side in zip(axes, [im1, im2], [0, 1], "left right".split(" ")):
        for x in range(0, len(inliers_kp[i])):
            ax.add_patch(plt.Circle((int(inliers_kp[i][x].pt[0]), int(inliers_kp[i][x].pt[1])), 0.4, color='orange'))
        for x in range(0, len(outliers_kp[i])):
            ax.add_patch(
                plt.Circle((int(outliers_kp[i][x].pt[0]), int(outliers_kp[i][x].pt[1])), 0.4, color='cyan'))
            ax.set_title(f"{side} image")
        ax.axis('off')
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    fig.suptitle("Geometric Rejection, outliers (cyan), inliers(orange)")
    plt.savefig(dest_path)


def plot_triangulated_points(tracker, im1, im2, triangulation_func, dest):
    kp1, desc1 = tracker.calculate_kps_and_descs(im1)
    kp2, desc2 = tracker.calculate_kps_and_descs(im2)
    inliers, _, x_outliers = tracker.calculate_matches(desc1, desc2, kp1=kp1, kp2=kp2)
    k, m1, m2 = read_cameras()
    left_matrix = k @ m1
    right_matrix = k @ m2
    points_x, points_y, points_z = [], [], []
    inliers_xyz = [[], [], []]
    x_rejected_xyz = [[], [], []]
    for source, target in zip([inliers, x_outliers], [inliers_xyz, x_rejected_xyz]):
        for m in source:
            p = kp1[m.queryIdx].pt
            q = kp2[m.trainIdx].pt
            p4d = triangulation_func(left_matrix, right_matrix, p, q)
            p3d = p4d[:3] / p4d[3]
            target[0].append(p3d[0])
            target[1].append(p3d[1])
            target[2].append(p3d[2])
            points_x.append(p3d[0])
            points_y.append(p3d[1])
            points_z.append(p3d[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(inliers_xyz[0], inliers_xyz[1], inliers_xyz[2])
    ax.scatter3D(x_rejected_xyz[0], x_rejected_xyz[1], x_rejected_xyz[2], color='red')
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(dest)
    return points_x, points_y, points_z


###### plots for ex3 ######

def plot_left_right_camera_position(left_pos, right_pos):
    """
    plot the cameras from above in 2d
    :param left_pos: list of 3d positions of the left camera
    :param right_pos: list of 3d position of the right camera
    """
    fig, ax = plt.subplots()
    ax.scatter([p[0] for p in left_pos], [p[2] for p in left_pos], color='blue', label='left camera')
    ax.scatter([p[0] for p in right_pos], [p[2] for p in right_pos], color='red', label='right camera')
    ax.legend()
    # set the scale of the axes:
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    plt.show()



def get_gt_left_camera_matrices(lines_num):
    """
    get the ground truth camera positions
    :return: the ground truth camera positions
    """
    with open(GT_LEFT_CAMERA_PATH) as f:
        lines = f.readlines()
    res = []
    for i in range(lines_num):
        camera_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        res.append(camera_mat)
    return res


def plot_supporters_and_deniers(left0, left1, supporters, deniers):
    """
    scatter plot the supporters and deniers on a pair of left images
    :param left0: the first left image
    :param left1: the second left image
    :param supporters: list of supporters - each is a tuple of (left0_kp, left1_kp)
    :param deniers: list of deniers - each is a tuple of (left0_kp, left1_kp)
    """
    fig, axes = plt.subplots(2, 1)
    # plot the images
    axes[0].imshow(cv2.cvtColor(left0, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(left1, cv2.COLOR_BGR2RGB))
    # plot the supporters, each is a tuple of (left0_kp, left1_kp)
    for s in supporters:
        axes[0].scatter(s[0].pt[0], s[0].pt[1], color='red', s=1)
        axes[1].scatter(s[1].pt[0], s[1].pt[1], color='red', s=1)
    # plot the deniers, each is a tuple of (left0_kp, left1_kp)
    for d in deniers:
        axes[0].scatter(d[0].pt[0], d[0].pt[1], color='blue', s=1)
        axes[1].scatter(d[1].pt[0], d[1].pt[1], color='blue', s=1)
    # set the axis to be off
    axes[0].axis('off')
    axes[1].axis('off')
    # set the title
    axes[0].set_title('Left 0')
    axes[1].set_title('Left 1')
    # set the figure title to be the number of supporters and deniers and their colors
    fig.suptitle(f'# of supporters (red): {len(supporters)}, # of deniers (blue): {len(deniers)}')
    plt.show()


######################
#       ex4         ##
######################

EX4_DOCS_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\docs\ex4'

def plot_left_matches_from_locations(frame_ids, locations):
    """
    plot matches between the left images with frame ids frame_ids. Show a 100x100 square around each location (subject
    to image boundaries)
    :param frame_ids: list of frame ids
    :param locations: list of locations of the left images
    """
    # find the size to show the subplots, so it would be close to a square
    size_x = int(np.ceil(np.sqrt(len(frame_ids))))
    size_y = int(np.ceil(len(frame_ids) / size_x))

    fig, axes = plt.subplots(size_y, size_x)
    # plot the images
    for i, frame_id in enumerate(frame_ids):
        left_img = utils.read_images(frame_id)[0]
        height, width = left_img.shape
        # get the 100x100 square around the location
        left = max(0, int(locations[i][0] - 50))
        right = min(width, int(locations[i][0] + 50))
        top = max(0, int(locations[i][1] - 50))
        bottom = min(height, int(locations[i][1] + 50))
        left_img = left_img[top:bottom, left:right]
        # plot the image with a dot on the location:
        axes[i // size_x, i % size_x].imshow(left_img, cmap='gray')
        axes[i // size_x, i % size_x].scatter(locations[i][0] - left, locations[i][1] - top, color='red', s=1)
        axes[i // size_x, i % size_x].axis('off')
        # set the title to be the frame id
        axes[i // size_x, i % size_x].set_title(frame_id)

    # set the title
    fig.suptitle('Left matches from locations')

    # save the figure
    plt.savefig('left_matches_from_locations.png')
    plt.clf()

def plot_consecutive_matches_from_location(frame_ids, locations):
    """
    plot matches between the left and right images with frame ids frame_ids. Show a 100x100 square around each location
    (subject to image boundaries). Create a file for each left right couple with the frame_id in the file name and
    figure title.
    :param frame_ids:
    :param locations:
    :return:
    """
    size_x = 2
    size_y = 1

    for i in range(len(frame_ids)):
        fig, axes = plt.subplots(size_y, size_x)
        left_img, right_img = utils.read_images(frame_ids[i])
        height, width = left_img.shape
        # get the 100x100 square around the location in the left image
        left_left_img = max(0, int(locations[i][0][0] - 50))
        right_left_img = min(width, int(locations[i][0][0] + 50))
        top_left_img = max(0, int(locations[i][0][1] - 50))
        bottom_left_img = min(height, int(locations[i][0][1] + 50))
        left_img = left_img[top_left_img:bottom_left_img, left_left_img:right_left_img]
        # get the 100x100 square around the location in the right image
        left_right_img = max(0, int(locations[i][1][0] - 50))
        right_right_img = min(width, int(locations[i][1][0] + 50))
        top_right_img = max(0, int(locations[i][1][1] - 50))
        bottom_right_img = min(height, int(locations[i][1][1] + 50))
        right_img = right_img[top_right_img:bottom_right_img, left_right_img:right_right_img]
        # plot the image with a dot on the location:
        axes[0].imshow(left_img, cmap='gray')
        axes[0].scatter(locations[i][0][0] - left_left_img, locations[i][0][1] - top_left_img, color='red', s=1)
        axes[0].axis('off')

        axes[1].imshow(right_img, cmap='gray')
        axes[1].scatter(locations[i][1][0] - left_right_img, locations[i][1][1] - top_right_img, color='red', s=1)
        axes[1].axis('off')

        # set the title
        fig.suptitle('Frame: ' + str(frame_ids[i]))

        # save the figure
        plt.savefig(EX4_DOCS_PATH + '/' + f'4.3_frame_{frame_ids[i]}.png')
        plt.clf()

    # create a gif from the images
    image_paths = glob(EX4_DOCS_PATH + '/4.3_frame_*.png')
    image_paths.sort()
    images = []
    for image_path in image_paths:
        images.append(imageio.imread(image_path))
    imageio.mimsave(EX4_DOCS_PATH + '/4.3.gif', images, duration=1)

