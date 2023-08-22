'''
Please do NOT make any changes to this file.
'''
# Author: DeeKay Goswami

from UB_Face import cluster_faces
import cv2
import numpy as np
import argparse
import json
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 3.")
    parser.add_argument(
        "--input_path", type=str, default="faceCluster_5",
        help="path to faceCluster K folder")
    parser.add_argument(
        "--num_cluster", type=str, default="5",
        help="number of clusters")
    parser.add_argument(
        "--output", type=str, default="./result_task2.json",
        help="path to the characters folder")

    args = parser.parse_args()
    return args

def save_results(result_dict, filename):
    results = []
    results = result_dict
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def read_images(img_dir):
    res = {}
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        res[img_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return res

def check_output_format(output, imgs, K):
    if not isinstance(output, list):
        print('Wrong output type! Should be a %s, but you get %s.' % (list, type(output)))
        return False
    if len(output) != K:
        print('Wrong cluster size! Should have %d clusters, but you get %d clusters.' % (K, len(output)))
        return False
    for i, cluster in enumerate(output):
        if not isinstance(cluster, list):
            print('Wrong cluster type in the %dth cluster! Should be a %s, but you get %s.' % (i, list, type(cluster)))
            return False
        for j, img_name in enumerate(cluster):
            if not isinstance(img_name, str):
                print('Wrong image name type for the %dth cluster %dth image slot! Should be a %s, but you get %s.' % (i, j, list, type(img_name)))
                return False
            if img_name not in imgs:
                print('Image name %s are not provide in the input dictionary!' % img_name)
                return False
    return True

def main():
    if cv2.__version__ != '4.5.4':
        print("Please use OpenCV 4.5.4")
        sys.exit(1)
    args = parse_args()
    path, filename = os.path.split(args.output)
    os.makedirs(path, exist_ok=True)
    imgs = read_images(args.input_path)
    result_list = cluster_faces(imgs, K=int(args.num_cluster))
    if not check_output_format(result_list, imgs, int(args.num_cluster)):
        print('Wrong output format!')
        sys.exit(2)
    save_results(result_list, args.output)

if __name__ == "__main__":
    main()

    