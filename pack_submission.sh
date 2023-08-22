#!/bin/bash

python task1.py --input_path validation_folder/images --output ./result_task1_val.json
python task1.py --input_path test_folder/images --output ./result_task1.json
python task2.py --input_path faceCluster_5 --num_cluster 5

python utils.py --ubit $1