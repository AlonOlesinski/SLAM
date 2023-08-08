import numpy as np


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


def triangulate_multiple_points(P, Q, p_array, q_array):
    """
    calculate the location of multiple points in 3d based on location in two images. Performed in a vectorized manner.
    :param P - 3x4 projection matrix of the first camera (to the center of the first camera)
    :param Q - 3x4 projection matrix of the second camera (to the center of the first camera)
    :param p_array - a list of points seen by the first camera
    :param q_array - a list of points seen by the second camera (hopefully the same point in the real world as p)
    return - a list of 4d (homogenous) vectors representing the real world location of the points, (in
    relation to the first camera's position)
    """
    A_row_1_stack = np.outer(p_array[:, 0] ,P[2]) - P[0]
    A_row_2_stack = np.outer(p_array[:, 1] ,P[2]) - P[1]
    A_row_3_stack = np.outer(q_array[:, 0] ,Q[2]) - Q[0]
    A_row_4_stack = np.outer(q_array[:, 1] ,Q[2]) - Q[1]
    # stack the above rows into a Nx4x4 matrix
    A = np.stack((A_row_1_stack, A_row_2_stack, A_row_3_stack, A_row_4_stack), axis=1)
    u, s, vh = np.linalg.svd(A)
    return vh[:, -1].T

