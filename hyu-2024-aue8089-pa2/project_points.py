import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (2xN)
    """

    # [TODO] get image coordinates
    x = points_3d[:, 0] / points_3d[:, 2]
    y = points_3d[:, 1] / points_3d[:, 2]
    r_square = x**2 + y**2

    # [TODO] apply distortion
    if len(D) == 2:
        k1, k2 = D
        k3, p1, p2 = 0, 0, 0
    if len(D) == 4:
        k1, k2, p1, p2 = D
        k3 = 0
    elif len(D) == 5:
        k1, k2, k3, p1, p2 = D

    r_d = (1 + k1*r_square + k2*r_square**2 + k3*r_square**3)
    x_d = x * r_d + 2*p1*x*y + p2*(r_square + 2*x**2)
    y_d = y * r_d + p1*(r_square + 2*y**2) + 2*p2*x*y

    projected_points = np.stack([K[0,0] * x_d + K[0,2],
                                 K[1,1] * y_d + K[1,2]], axis=-1)
    return projected_points
