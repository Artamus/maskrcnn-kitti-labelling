import argparse
import os
import tqdm
import numpy as np
from kitti_transformations import read_kitti_calibration_file, project_velo_to_image


MASRKCNN_TO_KITTI_CLASSES = {
    'person': 'Pedestrian',
    'car': 'Car'
}
USED_CLASSES = set(MASRKCNN_TO_KITTI_CLASSES.keys())
VELO_X_UNIT_VECTOR = np.array([1.0, 0.0])


def read_pointcloud(pointcloud_file_path: str) -> np.ndarray:
    points = np.fromfile(pointcloud_file_path, dtype=np.float32)
    return points.reshape(-1, 4)


def filter_forward_points(points: np.ndarray) -> np.ndarray:
    return points[points[:, 0] >= 0.0]


def save_frustum_points(labelled_points: np.ndarray, class_name: str, output_path: str):
    np.savez(output_path, points=labelled_points, class_name=class_name)


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

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    detections_dir = args.detections
    kitti_dir = args.kitti
    output_dir = args.output

    pointcloud_dir = os.path.join(kitti_dir, 'velodyne')
    calibration_dir = os.path.join(kitti_dir, 'calib')

    detection_files = os.listdir(detections_dir)

    running_detection_ind = 0

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

        original_point_indices = np.arange(0, len(points))

        projected_points = project_velo_to_image(points, calibration_data)  # Points are x, y
        projected_points = np.rint(projected_points).astype(np.int32)

        points = np.c_[points, np.zeros(len(points))]

        image_x_len = masks.shape[1]
        image_y_len = masks.shape[0]

        points_on_image = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_x_len) \
            & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_y_len)
        projected_points = projected_points[points_on_image]
        original_point_indices = original_point_indices[points_on_image]

        for ind in range(masks.shape[2]):
            class_name = classes[ind]
            if class_name not in USED_CLASSES:
                continue

            current_points = points.copy()
            mask = masks[:, :, ind]
            bbox_2d = bboxes[ind]

            # Label points for which the projection is on the mask
            points_matching_mask = mask[projected_points[:, 1], projected_points[:, 0]]
            mask_points_original_ind = original_point_indices[points_matching_mask]
            current_points[mask_points_original_ind, 3] = 1.0

            # Keep only the points for which the projection falls into the bounding box
            y_min, x_min, y_max, x_max = bbox_2d
            projected_points_in_bbox = (projected_points[:, 0] >= x_min) & (projected_points[:, 0] <= x_max) & \
                (projected_points[:, 1] >= y_min) & (projected_points[:, 1] <= y_max)

            bbox_points_original_ind = original_point_indices[projected_points_in_bbox]
            frustum_points = current_points[bbox_points_original_ind]

            if y_max - y_min < 25 or len(frustum_points) < 100:
                continue

            # Sanity check
            assert len(projected_points_in_bbox) == len(points_matching_mask), 'Different amount of mask points in projected points data'
            assert projected_points_in_bbox.sum() >= points_matching_mask.sum(), 'Bounding box has fewer points than mask'
            assert (points_matching_mask & projected_points_in_bbox).sum() == points_matching_mask.sum(), 'Mask is not subset of 2D bounding box'

            output_filename = f'{scene_id}_{ind}'
            output_filepath = os.path.join(output_dir, output_filename)
            save_frustum_points(frustum_points, class_name, output_filepath)

    print('Done!')
