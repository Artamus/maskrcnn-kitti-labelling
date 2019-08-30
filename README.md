# maskrcnn-kitti-labelling
Code to label the pointcloud of the KITTI dataset using MaskRCNN. The goal of this is to check if acquiring labels using a good 2D detector and then projecting those onto the pointcloud can be a substitute for spending money on labelling pointcloud data with 3D bounding boxes. While projecting these labels produces a sort of "bleeding" of labels onto background points as well, the goal is to also see if some deep learning methods learn to ignore the labels that bled onto the background.

As a baseline, there is also an implementation of using clustering to try and filter out bleeding outlier points and then forming a 3D bounding using the Convex Hull, as well as some scripts to demonstrate benchmarking that against ground truth.

# Installation
TODO

# Data format
TODO

# Usage
## Baseline scripts
Run these scripts from the main folder, as they assume that for the proper path.
All of the scripts require a path to the offline KITTI evaluator as an argument, I recommend using one from https://github.com/charlesq34/frustum-pointnets in the `train/kitti_eval` folder.


## Generating frustums
TODO