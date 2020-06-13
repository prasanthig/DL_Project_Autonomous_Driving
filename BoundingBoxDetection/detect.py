from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from torch.utils.data.sampler import SubsetRandomSampler
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, convert_map_to_road_map, convert_map_to_lane_map
from encoder_model import EncoderModel, DenseModel
from torchvision import transforms
import torchvision

def get_dataloader(batch_size,indices,data_dir):
    labeled_scene_index = np.arange(106, 134)

    val_sampler = SubsetRandomSampler(indices)

    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))    
                                              ])

    image_folder = data_dir
    annotation_csv =  f'{data_dir}/annotation.csv'
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True
                                     )

    val_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_size, num_workers=8,pin_memory =True, collate_fn=collate_fn,sampler=val_sampler)
    return val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/my_yolo3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=800, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--data_dir", type=str,default='../data',help="path to checkpoint model")
    opt = parser.parse_args()
    os.makedirs("output", exist_ok=True)

    print(opt)

    num_views = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_shape = np.array([3,256,304])
    encoder = EncoderModel(input_shape[0]).to(device)
    dense_models = [DenseModel([1,16,16,19]).to(device) for i in range(num_views)]
    darknet_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    load_model = True
    load_path=opt.checkpoint_model

    if load_model:
        checkpoint = torch.load(load_path,map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        darknet_model.load_state_dict(checkpoint['Darknet'])
        for i,model in enumerate(dense_models):
            model.load_state_dict(checkpoint['dense'+str(i+1)])

    models = [encoder]
    models.extend(dense_models)
    models.append(darknet_model)
    # Set in evaluation mode
    for model in models:
        model.eval()

    indices = [3024,3140,3400,3500]
    dataloader = get_dataloader(opt.batch_size,indices,opt.data_dir)
    classes = load_classes('./data/coco.names')  # Extracts class labels from file

    img_detections = []  # Stores detections for each image index
    road_images = []
    print("\nPerforming object detection:")
    prev_time = time.time()
    with torch.no_grad():
        for batch_i, (input_imgs,_,road_image_,_) in enumerate(dataloader):
            # Configure input
            road_image = road_image_[0]
            input_imgs = torch.stack(input_imgs)
            input_imgs = Variable(input_imgs.to(device))

            views =[torch.stack([img[i] for img in input_imgs]).to(device) for i in range(num_views)] # Get the 6 different views for all batch            
            encoded = [models[0](view) for view in views] # Pass views to encoder  
            flattened = [e.view(len(input_imgs),-1) for e in encoded] #flatten all views from encoder
            dense = [models[i+1](view) for i,view in enumerate(flattened)] # pass 6 views to 6 dense
            concatenated_views = torch.cat(dense,dim=1) # Concat all views channel wise

            # Get detections
            detections = models[-1](concatenated_views)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

            # Save image and detections
            img_detections.extend(detections)
            road_images.append(road_image)

    # Bounding-box colors
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i,(road_image,detections) in enumerate(zip(road_images,img_detections)):
        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(road_image,cmap='binary')
        ax.plot(400, 400, 'x', color="red")
        print("Image: ", img_i)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, [800,800])
            for d in detections:
                d1 = d.clone()
                x, y, w, h, conf, cls_conf, cls_pred, a_conf, a_pred = d1
                a_pred *=10
                if a_pred == 180:
                    a_pred = 178*np.pi/180
                else:
                    a_pred*=np.pi/180
                cosA = np.cos(a_pred)
                sinA = np.sin(a_pred)

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))  
                print("\t+ Angle: %s, Conf: %.5f" % (a_pred*180/np.pi, a_conf.item()))

                r0 = np.array([[(-w/2),(h/2)], [(-w/2),-(h/2)], [(w/2),-(h/2)],[(w/2),(h/2)]])
                R1 = np.array([[cosA, -sinA], [sinA, cosA]])
                r1 = torch.tensor(R1.dot(r0.T))

                r1[0,:] += x
                r1[1,:] += y
    
                point_squence = torch.stack([r1[:, 0], r1[:, 1], r1[:, 2], r1[:, 3], r1[:, 0]])
                ax.plot(point_squence.T[0] , point_squence.T[1] , color=color_list[cls_pred.long()])

               
        filename = 'ego'
        fig.savefig(f"output/{filename}_"+str(img_i)+".png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
