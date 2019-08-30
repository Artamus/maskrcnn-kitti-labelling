#!/bin/bash

min_samples=(20 40 60 80 100)
epsilons=(0.3 0.4 0.5)

for i in "${min_samples[@]}"
do 
    for j in "${epsilons[@]}"
    do
        echo "### Min samples: $i epsilon: $j ###"
        foldername="minsamp${i}_eps${j}_2d"
        python generate_baseline_kitti_bounding_boxes.py kitti_detections/ ~/KITTI/object/training/ kitti_labels_experiments/dbscan/$foldername -m dbscan -ms $i -e $j -d 2

        cd "kitti_labels_experiments/dbscan/$foldername"
        mkdir data
        mv *.txt data/

        # ~/frustum-pointnets/train/kitti_eval/evaluate_object_3d_offline
        # Example path to use frustum-pointnets offline evaluator for KITTI
        $1 ~/KITTI/object/training/label_2/ . > ./output.txt
        car_ap=$(grep "car_detection_3d AP" output.txt)
        pedestrian_ap=$(grep "Pedestrian_detection_3d AP" output.txt)
        echo "$car_ap" > results.txt
        echo "$pedestrian_ap" >> results.txt
        
        echo " ----> $car_ap"
        echo " ----> $pedestrian_ap"

        cd ../../../
    done
done