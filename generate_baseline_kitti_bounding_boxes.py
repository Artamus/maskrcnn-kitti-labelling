import argparse
import os
import tqdm
import hdbscan
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from typing import Union
from MinimumBoundingBox import MinimumBoundingBox
from kitti_transformations import transform_velo_to_rect, project_rect_to_image, read_kitti_calibration_file


MASKRCNN_TO_CITYSCAPES = {
    'person': 12.0,
    'car': 14.0,
}
MASRKCNN_TO_KITTI_CLASSES = {
    'person': 'Pedestrian',
    'car': 'Car',
}
USED_CLASSES = set(MASRKCNN_TO_KITTI_CLASSES.keys())


def read_pointcloud(pointcloud_file_path: str) -> np.ndarray:
    points = np.fromfile(pointcloud_file_path, dtype=np.float32)
    return points.reshape(-1, 4)


def filter_forward_points(points: np.ndarray) -> np.ndarray:
    return points[points[:, 0] >= 0.0]


def filter_outlier_points(points: np.ndarray, method: str, params: dict, dimensions: int = 2, debug: bool = False) -> np.ndarray:
    """Divides input points into a binary category of keep/exclude"""

    if dimensions == 2:
        detector_points = points[:, [0, 2]]
    elif dimensions == 3:
        detector_points = points[:, :3]
    else:
        raise ValueError(f'Dimensions {dimensions} is not supported')

    if method == 'dbscan':
        # Only 2 parameters, epsilon(distance) and min_samples
        detector = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(detector_points)
    elif method == 'optics':
        # Parameters: min_samples, max_eps (default is infinity, but this could be useful),
        # min_cluster_size (default same as min_samples), xi (very specific parameter, unlikely to be useful)
        max_eps = params.get('eps', None)
        min_cluster_size = params.get('min_cluster_size', None)

        if max_eps:
            detector = OPTICS(min_samples=params['min_samples'], max_eps=max_eps, min_cluster_size=min_cluster_size).fit(detector_points)
        else:
            detector = OPTICS(min_samples=params['min_samples'], min_cluster_size=min_cluster_size).fit(detector_points)

    elif method == 'hdbscan':
        # Parameters: min_cluster_size, min_samples, alpha (very specific parameter, unlikely to be useful)
        min_samples = params.get('min_samples', None)

        if min_samples:
            detector = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=min_samples, allow_single_cluster=True).fit(detector_points)
        else:
            detector = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], allow_single_cluster=True).fit(detector_points)
    else:
        raise ValueError(f'No method named {method} available')

    # TODO: Can also choose cluster where the center is closest to the viewport
    mode = scipy.stats.mode(detector.labels_)[0]
    points_in_cluster = detector.labels_ == mode

    if debug:
        plt.scatter(points[points_in_cluster, 0], points[points_in_cluster, 2], s=50, linewidth=0, c='gray', alpha=0.25)
        plt.scatter(points[~points_in_cluster, 0], points[~points_in_cluster, 2], s=50, linewidth=0, c='red', alpha=0.5)
        plt.show()

    return points_in_cluster


def generate_3d_bounding_box(points: np.ndarray) -> np.ndarray:
    bounding_rectangle = MinimumBoundingBox(points[:, [0, 2]])

    length = bounding_rectangle.length_parallel
    width = bounding_rectangle.length_orthogonal

    x = bounding_rectangle.rectangle_center[0]
    z = bounding_rectangle.rectangle_center[1]

    bottom_y = points[:, 1].max()
    height = abs(bottom_y - points[:, 1].min())
    return [height, width, length, x, bottom_y, z, -1 * bounding_rectangle.unit_vector_angle]


def assemble_label(class_name: str, bbox_2d: list, bbox_3d: list, score: float) -> list:

    kitti_class = MASRKCNN_TO_KITTI_CLASSES.get(class_name, class_name)
    bbox_2d_ordered = [bbox_2d[1], bbox_2d[0], bbox_2d[3], bbox_2d[2]]

    # type trunc occl alpha left top right bottom height width length x y z rot_y score
    return [kitti_class, 0.0, 3, bbox_3d[-1]] + bbox_2d_ordered + bbox_3d + [score]


def format_floats(items: Union[list, str, int]) -> list:
    float_format = '{:.2f}'

    if type(items) != list:
        return [float_format.format(float(items))]

    float_items = [float(item) for item in items]
    return [float_format.format(item) for item in float_items]


def save_labels(labels: list, output_file_path: str):
    if len(labels) == 0:
        return

    formatted_labels = [[label[0]] + format_floats(label[1]) + [str(label[2])] + format_floats(label[3:]) for label in labels]

    string_labels = [' '.join(label) for label in formatted_labels]

    output_string = '\n'.join(string_labels)

    with open(output_file_path, 'w') as f:
        f.write(output_string)


def validate_parameters(method: str, params: dict):
    existing_params = set(params)

    method_parameters = {
        'dbscan': {'min_samples', 'eps'},
        'optics': {'min_samples'},
        'hdbscan': {'min_cluster_size'}
    }

    if method == 'dbscan' and {'min_samples', 'eps'} <= existing_params:
        return
    elif method == 'optics' and {'min_samples'} <= existing_params:
        return
    elif method == 'hdbscan' and {'min_cluster_size'} <= existing_params:
        return

    if method not in method_parameters:
        raise ValueError(f'Method {method} is unknown')

    raise ValueError(f'Missing one or more required parameters: {method_parameters[method]}')


def get_arguments() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'detections', type=str,
        help='Path to directory containing all applicable detections'
    )

    parser.add_argument(
        'kitti', type=str,
        help='Path to base KITTI data directory that contains folders for images, pointclouds, etc.'
    )

    parser.add_argument(
        'output', type=str,
        help='Path where to save KITTI format labels from input detections'
    )

    parser.add_argument(
        '-m', '--method', type=str, required=True,
        choices=['dbscan', 'optics', 'hdbscan'],
        default='dbscan',
        help='Method to use for outlier detection'
    )

    parser.add_argument(
        '-d', '--dimensions', type=int,
        default=2,
        help='Number of dimensions to use for outlier detection, 2 is top-down view, 3 is all available dimensions'
    )

    parser.add_argument(
        '-ms', '--min_samples', type=int,
        help='Min samples parameter for the outlier detection'
    )

    parser.add_argument(
        '-e', '--eps', type=float,
        help='Epsilon parameter for outlier detection, used as max_eps for OPTICS'
    )

    parser.add_argument(
        '-mcs', '--min_cluster_size', type=int,
        help='Min cluster size parameter for OPTICS and HDBSCAN'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    detections_dir = args.detections
    kitti_dir = args.kitti
    output_dir = args.output
    output_points_dir = os.path.join(output_dir, 'points')
    method = args.method
    dimensions = args.dimensions

    params = dict()
    if args.min_samples:
        params['min_samples'] = args.min_samples
    if args.eps:
        params['eps'] = args.eps
    if args.min_cluster_size:
        params['min_cluster_size'] = args.min_cluster_size

    if not os.path.isdir(output_points_dir):
        os.makedirs(output_points_dir, exist_ok=True)

    pointcloud_dir = os.path.join(kitti_dir, 'velodyne')
    calibration_dir = os.path.join(kitti_dir, 'calib')

    detection_files = sorted(os.listdir(detections_dir))

    validate_parameters(method, params)

    for scene in tqdm.tqdm(detection_files):

        detections_file_path = os.path.join(detections_dir, scene)
        detections = np.load(detections_file_path)

        masks = detections['masks']
        classes = detections['classes']
        bboxes = detections['bboxes']
        scores = detections['scores']
        if len(classes) == 0:
            continue

        scene_id = scene.split('.')[0]

        calibration_file_path = os.path.join(calibration_dir, scene_id + '.txt')
        calibration_data = read_kitti_calibration_file(calibration_file_path)

        pointcloud_file_path = os.path.join(pointcloud_dir, scene_id + '.bin')
        points = read_pointcloud(pointcloud_file_path)
        points = filter_forward_points(points)
        points = points[:, :3]
        points = transform_velo_to_rect(points, calibration_data)

        labelled_points = np.c_[points, np.zeros(len(points))]  # VisDebug

        projected_points = project_rect_to_image(points, calibration_data)  # Points are x, y
        projected_points = np.rint(projected_points).astype(np.int32)

        original_point_indices = np.arange(0, len(points))

        image_x_len = masks.shape[1]
        image_y_len = masks.shape[0]

        points_on_image = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_x_len) \
            & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_y_len)
        projected_points = projected_points[points_on_image]
        original_point_indices = original_point_indices[points_on_image]

        labels = list()
        for ind in range(masks.shape[2]):
            class_name = classes[ind]
            if class_name not in USED_CLASSES:
                continue

            mask = masks[:, :, ind]
            bbox_2d = bboxes[ind]

            projected_points_on_mask = mask[projected_points[:, 1], projected_points[:, 0]]
            mask_points_ind = original_point_indices[projected_points_on_mask]

            if len(mask_points_ind) <= 20:
                continue

            points_on_mask = points[mask_points_ind]
            filtered_points_mask = filter_outlier_points(points_on_mask, method, params, dimensions, debug=False)
            filtered_points_ind = mask_points_ind[filtered_points_mask]

            object_points = points[filtered_points_ind]

            if len(object_points) <= 5:
                continue

            labelled_points[filtered_points_ind, 3] = MASKRCNN_TO_CITYSCAPES[class_name]  # VisDebug

            bbox_3d = generate_3d_bounding_box(object_points)
            label = assemble_label(class_name, bbox_2d, bbox_3d, scores[ind])

            labels.append(label)

        output_file_path = os.path.join(output_dir, scene_id + '.txt')
        save_labels(labels, output_file_path)

        points_output_path = os.path.join(output_points_dir, scene_id)
        np.save(points_output_path, labelled_points, allow_pickle=False)

    print('Done!')
