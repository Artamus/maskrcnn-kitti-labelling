# KITTI dataset transformations provider,
# code modified from the https://github.com/charlesq34/frustum-pointnets repository
import numpy as np


def _inverse_rigid_transform(transform):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_transform = np.zeros_like(transform)
    inv_transform[0:3, 0:3] = np.transpose(transform[0:3, 0:3])
    inv_transform[0:3, 3] = np.dot(-np.transpose(transform[0:3, 0:3]), transform[0:3, 3])
    return inv_transform


def _cartesian_to_homogenous(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def _project_rect_to_image(p, pts_3d_rect):
    pts_3d_rect = _cartesian_to_homogenous(pts_3d_rect)

    pts_2d = np.dot(pts_3d_rect, np.transpose(p))
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def _project_velo_to_ref(v2c, pts_3d_velo):
    pts_3d_velo = _cartesian_to_homogenous(pts_3d_velo)
    return np.dot(pts_3d_velo, np.transpose(v2c))


def _project_ref_to_rect(r0, pts_3d_ref):
    return np.transpose(np.dot(r0, np.transpose(pts_3d_ref)))


def _project_velo_to_rect(r0, v2c, pts_3d_velo):
    pts_3d_ref = _project_velo_to_ref(v2c, pts_3d_velo)
    return _project_ref_to_rect(r0, pts_3d_ref)


def _project_velo_to_image(p, r0, v2c, pts_3d_velo):
    pts_3d_rect = _project_velo_to_rect(r0, v2c, pts_3d_velo)
    return _project_rect_to_image(p, pts_3d_rect)


def transform_velo_to_rect(pts_3d_velo: np.ndarray, transformations: dict) -> np.ndarray:
    """Transforms points in the velodyne coordinate system into the rect coordinate system

    Arguments:
        pts_3d_velo {np.ndarray} -- points in the velodyne coordinate system, nx3
        calibration_path {dict} -- dictionary containing necessary transformations (R0_rect, Tr_velo_to_cam)

    Returns:
        np.ndarray -- points in the rect coordinate system, nx3
    """
    return _project_velo_to_rect(transformations['R0_rect'], transformations['Tr_velo_to_cam'], pts_3d_velo)


def project_rect_to_image(pts_3d_rect: np.ndarray, transformations: dict) -> np.ndarray:
    """Projects points in the rect coordinate system onto the image plane

    Arguments:
        pts_3d_rect {np.ndarray} -- points in the rect coordinate system, nx3
        transformations {dict} -- dictionary containing all the transformations (P2)

    Returns:
        np.ndarray -- points on the image plane, nx2
    """
    return _project_rect_to_image(transformations['P2'], pts_3d_rect)


def project_velo_to_image(pts_3d_velo: np.ndarray, transformations: dict) -> np.ndarray:
    """Projects points in the velodyne coordinate system onto the image plane

    Arguments:
        pts_3d_velo {np.ndarray} -- points in the velodyne coordinate system, nx3
        transformations {dict} -- dictionary containing all the transformations (P2, R0_rect, Tr_velo_to_cam)

    Returns:
        np.ndarray -- points on the image plane, nx2
    """
    return _project_velo_to_image(transformations['P2'], transformations['R0_rect'], transformations['Tr_velo_to_cam'], pts_3d_velo)


def read_kitti_calibration_file(file_path: str) -> dict:
    """Read a KITTI-formatted calibration into a dictionary

    Arguments:
        file_path {str} -- path to the calibration file

    Returns:
        dict -- dictionary containing the transformations
    """
    calib = dict()

    with open(file_path, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue

            key, val = line.rstrip().split(': ')
            calib[key] = np.array([float(elem) for elem in val.split(' ')])

    return {
        'Tr_velo_to_cam': calib['Tr_velo_to_cam'].reshape((3, 4)),
        'Tr_cam_to_velo': _inverse_rigid_transform(calib['Tr_velo_to_cam'].reshape((3, 4))),
        'R0_rect': calib['R0_rect'].reshape((3, 3)),
        'P2': calib['P2'].reshape((3, 4))
    }
