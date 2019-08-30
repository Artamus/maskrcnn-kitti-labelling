import os
import sys
import random
import math
import argparse
import tqdm
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

"""Code based on the MaskRCNN repository's own examples"""

# Root directory of the project
ROOT_DIR = os.path.abspath('./Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class_names_np = np.array(class_names)


def read_split(file_path):
    scene_ids = set()

    with open(file_path, 'r') as f:
        for line in f:
            scene_ids.add(line.rstrip())

    return scene_ids


def initialize_model():

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model


def get_arguments():
    parser = argparse.ArgumentParser(description='Perform MaskRCNN detections and save results')

    parser.add_argument(
        'image_dir', type=str,
        help='Directory of images to perform detection on'
    )

    parser.add_argument(
        'output_dir', type=str,
        help='Directory to save data about detections made'
    )

    parser.add_argument(
        '-s', '--split', type=str,
        help='Path to file containing scene IDs to keep for detection'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    image_dir = args.image_dir
    output_dir = args.output_dir
    split = args.split

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    image_file_names = sorted(os.listdir(image_dir))

    if split:
        split_scenes = read_split(split)

        image_file_names = [image_file for image_file in image_file_names if image_file.split('.')[0] in split_scenes]

    model = initialize_model()

    for file_name in tqdm.tqdm(image_file_names):
        scene_id = file_name.split('.')[0]
        image = skimage.io.imread(os.path.join(image_dir, file_name))

        results = model.detect([image], verbose=0)

        r = results[0]

        classes = class_names_np[r['class_ids']]

        output_file_name = os.path.join(output_dir, scene_id + '.npz')
        np.savez(output_file_name, bboxes=r['rois'], masks=r['masks'], classes=classes, scores=r['scores'])
