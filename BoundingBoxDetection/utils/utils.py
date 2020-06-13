from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def xyhwt2coord (cx,cy,w,h,angle) : 
    w = to_cpu(w).numpy()
    h = to_cpu(h).numpy()
    cx , cy = to_cpu(cx).numpy(), to_cpu(cy).numpy()
    angle = to_cpu(angle).numpy()
    r0 = np.vstack([np.dstack([w/2,-h/2]),np.dstack([w/2,h/2]),np.dstack([-w/2,h/2]),np.dstack([-w/2,-h/2])]).transpose(1,0,2)
    cA = np.cos(angle)
    sA = np.sin(angle)
    R1 = np.vstack([np.dstack([cA, -sA]),np.dstack([sA, cA])]).transpose(1,2,0)
    r1 = torch.bmm(torch.tensor(r0),torch.tensor(R1))
    r1[:,:,0] += torch.tensor(cx).view(cx.shape[0],1)
    r1[:,:,1] += torch.tensor(cy).view(cx.shape[0],1)
    return r1

def bbox_iou(box1,angle1, box2,angle2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    angle1 = angle1*10
    angle1[angle1==180] = 178  # for bin number 18[ 175-180 ] take 178 as the angle
    angle1 = angle1*np.pi/180

    angle2 = angle2*10
    angle2[angle2==180] = 178
    angle2 = angle2*np.pi/180
   
    single_box_comparison = box1.shape[0] == 1
    if x1y1x2y2 == False:
        cx1, cy1,w1,h1 = box1[:,0],box1[:,1],box1[:,2],box1[:,3]
        cx2, cy2,w2,h2= box2[:,0],box2[:,1],box2[:,2],box2[:,3]
        
        bbxes1 = xyhwt2coord(cx1, cy1,w1,h1,angle1)
        bbxes2 = xyhwt2coord(cx2, cy2,w2,h2,angle2)

    ious = []
    for i in range(len(bbxes2)):
        if single_box_comparison:
            a = Polygon(bbxes1[0]).convex_hull
        else:
            a = Polygon(bbxes1[i]).convex_hull
        b = Polygon(bbxes2[i]).convex_hull
        iou = a.intersection(b).area/(a.union(b).area+1e-20)

        ious.append(iou)

    if torch.cuda.is_available():
        return torch.tensor(ious).cuda()

    return torch.tensor(ious)
    
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, theta, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:14].max(1, keepdim=True)
        angle_confs, angle_preds = image_pred[:,14:].max(1,keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(), angle_confs.float(),angle_preds.float()), 1)
      
        # Perform non-maximum suppression
        keep_boxes = [] 
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0),detections[0,8], detections[:, :4],detections[:,8],x1y1x2y2=False) > nms_thres
            label_match = detections[0, -3] == detections[:, -3]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            if prediction.is_cuda:
                invalid = large_overlap.cuda() & label_match.cuda()
            else:
                invalid = to_cpu(large_overlap) & to_cpu(label_match) 
            keep_boxes += [detections[0]]
            detections = detections[~invalid]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def build_targets(pred_boxes, pred_cls,pred_angle_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nT = pred_angle_cls.size(-1) # Theta  
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    angle_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    ta = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    tacls = FloatTensor(nB, nA, nG, nG, nT).fill_(0)
    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    # Separate target values and angles
    b, target_labels = target[:, :2].long().t()
    target_angles = target[:,-1].long().t()

    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0


    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label and angles
    tcls[b, best_n, gj, gi, target_labels] = 1
    tacls[b, best_n, gj, gi, target_angles] = 1


    # Compute class label, angle label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    angle_mask[b, best_n, gj, gi] = (pred_angle_cls[b,best_n, gj, gi].argmax(-1) == target_angles).float()

    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi],pred_angle_cls[b,best_n, gj, gi].argmax(-1), target_boxes,target_angles, x1y1x2y2=False)
    tconf = obj_mask.float()
    return iou_scores, class_mask,angle_mask, obj_mask, noobj_mask, tx, ty, tw, th, tacls, tcls, tconf


#Input ({'bounding_box':bbxes, 'category':[int]}, 
#       {'bounding_box':bbxes, 'category':[int]})
#Output (all_bbxes, 7)
def compute_yolo_targets(target):
    b = torch.tensor((), dtype=torch.float64)
    img_size = 800
    bins = [0,5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,180]

    for t in range(len(target)):
        bbxes = target[t]['bounding_box'].clone()
        categories = target[t]['category'].clone().type(torch.DoubleTensor)
        world_coord = torch.stack([(bbxes[:,0,:] * 10 + 400), (-bbxes[:,1,:] * 10 + 400)],dim=2)
        IDX = torch.full(categories.shape,t, dtype=torch.double)
        centroid = world_coord.mean(dim=1)

        C_x = centroid[:,0]/img_size
        C_y = centroid[:,1]/img_size

        side_one = world_coord[:,0,:]- world_coord[:,1,:]
        H = torch.tensor(np.linalg.norm(side_one, axis=1))/img_size

        side_two = world_coord[:,0,:]-world_coord[:,2,:]
        W = torch.tensor(np.linalg.norm(side_two, axis=1))/img_size

        slope = (world_coord[:,2,1]- world_coord[:,0,1])/(world_coord[:,2,0]- world_coord[:,0,0] + 1e-12)
        degree = np.remainder((np.arctan(slope)*180/np.pi) + 180,180) # Convert from 0 to 180
        degree_categorize = np.digitize(degree,bins) - 1         
        theta = torch.tensor(degree_categorize,dtype=torch.double).view(-1) # Classification

        sample_target = torch.stack([IDX,categories, C_x, C_y, W, H,theta],dim=1)
        b = torch.cat([b, sample_target])
    return b.float() 