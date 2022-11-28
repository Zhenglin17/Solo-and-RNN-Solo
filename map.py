import torch
from collections import Counter


def mean_average_precision(pred_masks, true_masks, iou_threshold = 0.5, num_classes = 1):
    average_precision = 0
    epsilon = 1e-6
    
    detections = []
    ground_truths = []
    
    for detection in pred_masks:
        detections.append(detection)
    for true_mask in true_masks:
        ground_truths.append(true_mask)
    #img 0 has 3 masks
    #img 1 has 5 masks
    #amount_masks = {0:3, 1:5}
    amount_masks = Counter([gt[0] for gt in ground_truths])
    for key, val in amount_masks.items():
        #amount_boxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        amount_masks[key] = torch.zeros(val)
        detections.sort(key = lambda x: x[1], reverse = True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_masks = len(ground_truths)
    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
        num_gts = len(ground_truth_img)
        best_iou = 0
        detection_post = torch.nn.functional.interpolate(detection, size=(1250, 1250), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        for idx, gt in enumerate(ground_truth_img):
            iou = torch.sum((torch.mul(detection_post[2], gt[2]))[2])/(torch.sum(detection_post[2]) + torch.sum(gt[2]) - torch.mul(detection_post[2], gt[2]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            if amount_masks[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_masks[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else: 
            FP[detection_idx] = 1
    # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    TP_cumsum = torch.cumsum(TP, dim = 0)
    FP_cumsum = torch.cumsum(FP, dim = 0)
    recalls = TP_cumsum / (total_true_masks + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    average_precision = torch.trapz(precisions, recalls)
    
    return average_precision


def mean_average_precision_15(pred_masks, true_masks, iou_threshold = 0.5, num_classes = 1):
    average_precision = 0
    epsilon = 1e-6
    
    detections = []
    ground_truths = []
    flags = []
    for index, detection in enumerate(pred_masks):
        if torch.sum(detection[2]) > 25:
            detections.append(detection)
            flags.append(index)
    for flag in flags:
        ground_truths.append(true_masks[flag])
    #img 0 has 3 masks
    #img 1 has 5 masks
    #amount_masks = {0:3, 1:5}
    amount_masks = Counter([gt[0] for gt in ground_truths])
    for key, val in amount_masks.items():
        #amount_boxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        amount_masks[key] = torch.zeros(val)
        detections.sort(key = lambda x: x[1], reverse = True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_masks = len(ground_truths)
    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
        num_gts = len(ground_truth_img)
        best_iou = 0
        detection_post = torch.nn.functional.interpolate(detection, size=(1250, 1250), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        for idx, gt in enumerate(ground_truth_img):
            iou = torch.sum((torch.mul(detection_post[2], gt[2]))[2])/(torch.sum(detection_post[2]) + torch.sum(gt[2]) - torch.mul(detection_post[2], gt[2]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            if amount_masks[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_masks[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else: 
            FP[detection_idx] = 1
    # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    TP_cumsum = torch.cumsum(TP, dim = 0)
    FP_cumsum = torch.cumsum(FP, dim = 0)
    recalls = TP_cumsum / (total_true_masks + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    average_precision = torch.trapz(precisions, recalls)
    
    return average_precision

def mean_average_precision_50(pred_masks, true_masks, iou_threshold = 0.5, num_classes = 1):
    average_precision = 0
    epsilon = 1e-6
    
    detections = []
    ground_truths = []
    flags = []
    for index, detection in enumerate(pred_masks):
        if torch.sum(detection[2]) > 50:
            detections.append(detection)
            flags.append(index)
    for flag in flags:
        ground_truths.append(true_masks[flag])
    #img 0 has 3 masks
    #img 1 has 5 masks
    #amount_masks = {0:3, 1:5}
    amount_masks = Counter([gt[0] for gt in ground_truths])
    for key, val in amount_masks.items():
        #amount_boxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        amount_masks[key] = torch.zeros(val)
        detections.sort(key = lambda x: x[1], reverse = True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_masks = len(ground_truths)
    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
        num_gts = len(ground_truth_img)
        best_iou = 0
        detection_post = torch.nn.functional.interpolate(detection, size=(1250, 1250), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        for idx, gt in enumerate(ground_truth_img):
            iou = torch.sum((torch.mul(detection_post[2], gt[2]))[2])/(torch.sum(detection_post[2]) + torch.sum(gt[2]) - torch.mul(detection_post[2], gt[2]))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            if amount_masks[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_masks[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else: 
            FP[detection_idx] = 1
    # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
    TP_cumsum = torch.cumsum(TP, dim = 0)
    FP_cumsum = torch.cumsum(FP, dim = 0)
    recalls = TP_cumsum / (total_true_masks + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    average_precision = torch.trapz(precisions, recalls)
    
    return average_precision