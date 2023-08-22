import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds")
    parser.add_argument("--groundtruth")
    parser.add_argument("--iou", default=0.5)
    parser.add_argument("--beta", default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.preds) as file:
        preds = json.load(file)

    with open(args.groundtruth) as file:
        annos = json.load(file)

    judges = []
    for i, a_value in enumerate(annos):
        iname = a_value["iname"]
        bbox_gt = a_value["bbox"]

        detected = False
        if iname in preds:
            for bbox_pred in preds[iname]:
                iou = compute_iou(bbox_pred, bbox_gt)
                if iou > args.iou:
                    detected = True
                    break
        judges.append(detected)
        # for j, p_value in enumerate(preds):
        #     if detected:
        #         break

        #     if p_value["iname"] == iname:
        #         iou = compute_iou(p_value["bbox"], bbox)
        #         if iou > args.iou:
        #             detected = True
        # judges.append(detected)
    

    ntp = 0

    for i, j in enumerate(judges):
        if j:
            ntp += 1

    if ntp == 0 :
        return 0
    nfn = len(judges) - ntp
    nfp = len(preds) - ntp

    precision = ntp / len(preds)
    recall = ntp / len(judges)
    fbeta = (1 + args.beta ** 2) * precision * recall / ((args.beta ** 2 * precision) + recall)
    return fbeta

def compute_iou(bbox1, bbox2):
    #ensure each element in the first bounding box is not negative
    if any(t < 0 for t in bbox1): return 0.

    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_max = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    #the area of intersection
    intersection_area = max(0, x_max-x_min)*max(0, y_max-y_min)
    #bbox1 area
    bbox1_area = bbox1[2]*bbox1[3]
    #bbox2 area 
    bbox2_area = bbox2[2]*bbox2[3]
    #compute IOU
    iou = intersection_area / (bbox1_area+bbox2_area - intersection_area)
    return iou

fbeta = main()
print('F1 score: %.4f' % fbeta)
