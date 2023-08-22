'''
Please do NOT make any changes to this file.
'''
# Author: DeeKay Goswami

import os
import cv2
import json
import zipfile
import argparse
import numpy as np
from typing import Dict, List


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()



def check_submission(py_file, check_list=['cv2.imshow(', 'cv2.imwrite(', 'cv2.imread(', 'open(']):
    res = True
    with open(py_file, 'r') as f:
        lines = f.readlines()
    for nline, line in enumerate(lines):
        for string in check_list:
            if line.find(string) != -1:
                print('You submitted code (in line %d) cannot have %s (Even if it is commented). Please remove that and zip again.' % (nline + 1, string[:-1]))
                res = False
    return res

def files2zip(files: list, zip_file_name: str):
    res = True
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                if name == 'UB_Face.py':
                    if not check_submission(file):
                        print('Zipping error!')
                        res = False
                zf.write(file, arcname=name)
            else:
                print('Zipping error! Your submission must have file %s, even if you does not change that.' % name)
                res = False
    return res

def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project Face submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file_list = ['UB_Face.py', 'result_task1.json', 'result_task1_val.json', 'result_task2.json']
    res = files2zip(file_list, 'submission_' + args.ubit + '.zip')
    if not res:
        print('Zipping failed.')
    else:
        print('Zipping succeed.')