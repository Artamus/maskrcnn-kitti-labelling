#!/bin/bash

min_cluster_size=(2 3 4 5 10)
min_samples=(0 1 2 3 4 5 10)

for i in "${min_cluster_size[@]}"
do 
    for j in "${min_samples[@]}"
    do
        if (( j > i )); then
            continue
        fi

        echo "### Min cluster size: $i min samples: $j ###"
        foldername="mcs${i}_ms${j}_2d"
        python generate_baseline_kitti_bounding_boxes.py kitti_detections/ ~/KITTI/object/training/ kitti_labels_experiments/hdbscan_2/$foldername -m hdbscan -mcs $i -ms $j -d 2

        cd "kitti_labels_experiments/hdbscan/$foldername"
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
