import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_match_on_4_images(left0_img, right0_img, left1_img, right1_img, left0_kp, right0_kp,
                           left1_kp, right1_kp, title):
    """
    plot the matches on 4 images.
    :param left0_img: the first left image
    :param right0_img: the first right image
    :param left1_img: the second left image
    :param right1_img: the second right image
    :param left0_kp: the first left image keypoints
    :param right0_kp: the first right image keypoints
    :param left1_kp: the second left image keypoints
    :param right1_kp: the second right image keypoints
    :param title: the title of the plot
    """
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(left0_img, cmap='gray')
    ax[0, 0].scatter(left0_kp.pt[0], left0_kp.pt[1], c='r', s=3)
    ax[0, 1].imshow(right0_img, cmap='gray')
    ax[0, 1].scatter(right0_kp.pt[0], right0_kp.pt[1], c='r', s=3)
    ax[1, 0].imshow(left1_img, cmap='gray')
    ax[1, 0].scatter(left1_kp.pt[0], left1_kp.pt[1], c='r', s=3)
    ax[1, 1].imshow(right1_img, cmap='gray')
    ax[1, 1].scatter(right1_kp.pt[0], right1_kp.pt[1], c='r', s=3)
    plt.suptitle(title)
    plt.show()


def plot_supporters_and_deniers(left0, left1, supporters, deniers, dir_path):
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
    plt.savefig(dir_path + '/supporters_deniers.png')


def project(p3d, camera):
    """
    project the 3d point to the image plane.
    :param p3d: 3d point
    :param camera: 3x4 matrix (K @ [R|t])
    :return: 2d point
    """
    p3d = np.append(p3d, 1)
    p2d = camera @ p3d
    p2d = p2d[:2] / p2d[2]
    return p2d


def project_points(points, camera):
    """
    project the 3d points to the image plane.
    :param points: 3xn matrix
    :param camera: 3x4 matrix (K @ [R|t])
    :return: 2xn matrix
    """
    ones_row = np.ones((1, points.shape[1]))
    points = np.append(points, ones_row, axis=0)
    points = camera @ points
    points = points[:2] / points[2]
    return points
