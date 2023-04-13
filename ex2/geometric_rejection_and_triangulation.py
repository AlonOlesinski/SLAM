import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\dataset\sequences\05\\'
DOCS_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\docs\ex2\\'

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    left_img = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    right_img = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return left_img, right_img


def calc_deviations(detector, im1, im2, matcher):
    """
    compute the absolute rounded y value difference of every match of im1 and im2.
    """
    kp1, desc1 = detector.detectAndCompute(im1, None)
    kp2, desc2 = detector.detectAndCompute(im2, None)
    matches = matcher.match(desc1, desc2)
    deviations = []
    for m in matches:
        dv = abs(round(kp1[m.queryIdx].pt[1]) - round(kp2[m.trainIdx].pt[1]))
        deviations.append(dv)
    return kp1, kp2, desc1, desc2, matches, deviations


def deviations_hist(im1, im2, detector, matcher):
    """
    code for 2.1 - compute histogram of (rounded) y axis deviation of matches
    """
    _, _, _, _, _, deviations = calc_deviations(detector, im1, im2, matcher)

    print(
        f'{round(100 * len(list(filter(lambda x: x > 2, deviations))) / len(deviations), 2)}% of the matches '
        f'deviate by more than 2 pixels')
    plt.hist(deviations, bins=max(deviations))
    plt.xlabel('Y axis deviation')
    plt.ylabel('Number of matches')
    plt.title('Y Axis Deviations Histogram')
    plt.savefig(DOCS_PATH + '2.1.png')


def geometric_rejection(im1, im2, detector, matcher):
    """
    code for 2.2 - geometric rejection of outliers
    """
    kp1, kp2, _, _, matches, deviations = calc_deviations(detector, im1, im2, matcher)
    inliers = [[], []]
    outliers = [[], []]
    discarded_counter = 0
    for i, dv in enumerate(deviations):
        m = matches[i]
        target = outliers if dv > 2 else inliers
        target[0].append(kp1[m.queryIdx])
        target[1].append(kp2[m.trainIdx])
        if target == outliers:
            discarded_counter += 1

    fig, axes = plt.subplots(2, 1)
    for ax, im, i, side in zip(axes, [im1, im2], [0,1], "left right".split(" ")):
        for x in range(0, len(inliers[i])):
            ax.add_patch(plt.Circle((int(inliers[i][x].pt[0]), int(inliers[i][x].pt[1])),0.4,color='orange'))
        for x in range(0, len(outliers[i])):
            ax.add_patch(
                plt.Circle((int(outliers[i][x].pt[0]), int(outliers[i][x].pt[1])),0.4, color='cyan'))
            ax.set_title(f"{side} image")
        ax.axis('off')
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    print(discarded_counter)
    fig.suptitle("Geometric Rejection, outliers (cyan), inliers(orange)")
    plt.savefig(DOCS_PATH + '2.2.png')

def read_cameras():
    with open(DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:] # skip first token
        l2 = f.readline().split()[1:] # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2

def triangulate_point(P, Q, p, q):
    """
    calculate the location of a point in 3d based on location in two images
    :param P - 3x4 projection matrix of the first camera (to the center of the first camera)
    :param Q - 3x4 projection matrix of the second camera (to the center of the first camera)
    :param p - a point seen by the first camera
    :param q - a point seen by the second camera (hopefully the same point in the real world as p)
    return - a 4d (homogenous) vector representing the real world location of the point, (in
    relation to the first camera's position)
    """
    A_row_1 = P[2] * p[0] - P[0]
    A_row_2 = P[2] * p[1] - P[1]
    A_row_3 = Q[2] * q[0] - Q[0]
    A_row_4 = Q[2] * q[1] - Q[1]
    A = np.array([A_row_1, A_row_2, A_row_3, A_row_4])
    u, s, vh = np.linalg.svd(A)
    return vh[-1].reshape((4, 1))

def show_triangulated_points(im1, im2, detector, matcher, triangulation_func):
    """
    code for 2.3 - show the triangulated points
    :param im1: left image
    :param im2: right image
    :param detector: detector to use
    :param matcher: matcher to use
    :param triangulation_func: function to use for triangulation
    :return:
    """
    k, m1, m2 = read_cameras()
    kp1, kp2, _, _, matches, deviations = calc_deviations(detector, im1, im2, matcher)
    inliers_x, inliers_y, inliers_z = [], [], []
    outliers_x, outliers_y, outliers_z = [], [], []
    point_x, point_y, point_z = [], [], []

    left_matrix = k@m1
    right_matrix = k@m2
    for i, dv in enumerate(deviations):
        if dv > 2: # geometric rejection
            continue
        m = matches[i]
        p = kp1[m.queryIdx].pt
        q = kp2[m.trainIdx].pt
        p4d = triangulation_func(left_matrix, right_matrix, p, q)
        p3d = p4d[:3] / p4d[3]

        point_x.append(p3d[0])
        point_y.append(p3d[1])
        point_z.append(p3d[2])

        if p3d[2] < 0:
            outliers_x.append(p3d[0])
            outliers_y.append(p3d[1])
            outliers_z.append(p3d[2])
            # print the x value of the erroneous point in the left and right images
            print(f'l: {p[0]}, r: {q[0]}')
            continue

        assert p[0] >= q[0]

        inliers_x.append(p3d[0])
        inliers_y.append(p3d[1])
        inliers_z.append(p3d[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(inliers_x, inliers_y, inliers_z)
    ax.scatter3D(outliers_x, outliers_y, outliers_z, color='red')
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-100, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return point_x, point_y, point_z




if __name__ == "__main__":

    # # 2.1
    # left_1, right_1 = read_images(0)
    # akaze = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # deviations_hist(left_1, right_1, akaze, matcher)

    # 2.2
    # left_1, right_1 = read_images(0)
    # akaze = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # geometric_rejection(left_1, right_1, akaze, matcher)

    # 2.3
    left_1, right_1 = read_images(0)
    akaze = cv2.AKAZE_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    points1 = show_triangulated_points(left_1, right_1, akaze, matcher, triangulate_point)
    points2 = show_triangulated_points(left_1, right_1, akaze, matcher, cv2.triangulatePoints)
    print(f'median distance: {np.median(np.abs(np.array(points1) - np.array(points2)))}')
