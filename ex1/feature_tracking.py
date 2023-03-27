import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r'C:\Users\alono\OneDrive\desktop\studies\VAN_ex\dataset\sequences\05\\'


def set_figure_shape(detectors, imgs):
    """
    set the figure shape according to the number of images and detectors
    """
    assert len(imgs) == len(detectors) and len(imgs) <= 4
    if len(imgs) == 1:
        plt_shape = (1, 1)
    elif len(imgs) == 2:
        plt_shape = (2, 1)
    elif len(imgs) == 4:
        plt_shape = (2, 2)
    else:
        raise ValueError("The number of images must be 1,2 or 4")
    fig, axes = plt.subplots(*plt_shape)
    if len(imgs) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    return axes


def show_imgs_with_kp(imgs, detectors):
    """
    code for 1.1 and 1.2 - for imgae_i in imgs (list of 1,2 or 4 images)  and detector_i in
    detectors (list of cv2 detectors with same length as imgs), show the image with calculated
    kps. Also prints the first two descriptors of each image.
    """
    axes = set_figure_shape(detectors, imgs)
    for i in range(len(imgs)):
        kp, desc = detectors[i].detectAndCompute(imgs[i], None)
        print(f'First descriptor of image number {i + 1}:')
        print(desc[0])
        print(f'Second descriptor of image number {i + 1}:')
        print(desc[1])
        img_with_kp = cv2.drawKeypoints(imgs[i], kp, None, (0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        axes[i].set_title(f'image {i + 1} with kp')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        axes[i].imshow(img_with_kp)

    plt.savefig('1.1.png')


def show_imgs_with_matches(im1, im2, detector, matcher, ratio=0.0, find_false_negative=False):
    """
    code for 1.3 and 1.4 - show 20 matches on two images
    :param im1: first image
    :param im2: second image
    :param detector: cv2 detector to be used (orb,detector,etc..)
    :param matcher: cv2 matcher to be used (BFMatcher..)
    :param ratio: if > 0, conduct ratio test with this ratio
    :param find_false_negative: if true, output only one which does the test for a certain ratio
    """
    kp1, desc1 = detector.detectAndCompute(im1, None)
    kp2, desc2 = detector.detectAndCompute(im2, None)
    if ratio > 0 or find_false_negative:
        matches_to_keep = []
        matches = matcher.knnMatch(desc1, desc2, k=2)
        print(f'number of matches before ratio test: {len(matches)}')
        for m1, m2 in matches:
            if m1.distance < ratio * m2.distance:
                matches_to_keep.append(m1)
        print(F"{len(matches) - len(matches_to_keep)} were removed by ratio test")
    else:
        matches_to_keep = matcher.match(desc1, desc2)


    matches_to_keep = list(matches_to_keep)
    random.shuffle(matches_to_keep)
    matches_img = cv2.drawMatches(im1, kp1, im2, kp2, matches_to_keep[:20], None, flags=2)
    # save the image
    cv2.imwrite('matches.png', matches_img)


def show_false_negative(im1, im2, detector, matcher):
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
                    cv2.imwrite('false_negative.png', matches_img)
                    return



def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    left_img = cv2.imread(DATA_PATH + 'image_0/' + img_name, 0)
    right_img = cv2.imread(DATA_PATH + 'image_1/' + img_name, 0)
    return left_img, right_img


if __name__ == "__main__":
    # 1.1 + 1.2
    # left_1, right_1 = read_images(0)
    # detector = cv2.AKAZE_create()
    # show_imgs_with_kp([left_1, right_1], [detector] * 2)

    # 1.3
    # left_1, right_1 = read_images(0)
    # detector = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher()  # cv2.NORM_L2 by default
    # show_imgs_with_matches(left_1, right_1, detector, matcher)

    # 1.4 show 20 true positive
    # left_1, right_1 = read_images(0)
    # detector = cv2.AKAZE_create()
    # matcher = cv2.BFMatcher()  # cv2.NORM_L2 by default
    # show_imgs_with_matches(left_1, right_1, detector, matcher, ratio=0.5)

    # 1.4 show one false negative
    left_1, right_1 = read_images(0)
    detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher()
    show_false_negative(left_1, right_1, detector, matcher)
