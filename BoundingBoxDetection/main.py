from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from terminaltables import AsciiTable
from test import evaluate

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, convert_map_to_road_map, convert_map_to_lane_map
from encoder_model import EncoderModel, DenseModel
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/my_yolo3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=800, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--run_specs", type=str, default="default", help="details of the model")
    parser.add_argument("--model",type=str, default="_70.pth", help="path to model state_dict")
    parser.add_argument("--data_dir",type=str, default="./data", help="path to data folder")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_run = datetime.datetime.now()
    saved_model_dir = './Models/Run_'+opt.run_specs+'_'+str(random_run)+'/'

    os.makedirs("output", exist_ok=True)
    os.makedirs(saved_model_dir,exist_ok=True)

    # Get data configuration
    class_names = load_classes('./data/coco.names')

    #Initiate encoder-dense model
    num_views = 6
    load_encoder_model = True
    load_full_model = False
    load_path = opt.model

    input_shape = np.array([3,256,304])
    encoded_shape = np.array([1,16,16,19])

    encoder = EncoderModel(input_shape[0]).to(device)
    dense_models = [DenseModel(encoded_shape).to(device) for i in range(num_views)]
    darknet_model = Darknet(opt.model_def,img_size=opt.img_size).to(device)

    assert(load_encoder_model or load_full_model == True)

    if load_encoder_model:
        checkpoint = torch.load(load_path,map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        for i,model in enumerate(dense_models):
            model.load_state_dict(checkpoint['dense'+str(i+1)])
        darknet_model.apply(weights_init_normal)
    else:
        checkpoint = torch.load(load_path,map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        darknet_model.load_state_dict(checkpoint['Darknet'])
        for i,model in enumerate(dense_models):
            model.load_state_dict(checkpoint['dense'+str(i+1)])


    # Get dataloader
    image_folder = opt.data_dir
    annotation_csv =  f'{opt.data_dir}/annotation.csv'
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))    
                                              ])

    sample_indices = np.arange(0,28*126)
    split = 3024  # 24*126 : 90% train-val split
    train_indices, val_indices = sample_indices[:split], sample_indices[split:]
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    train_sampler = SubsetRandomSampler(train_indices[:100])
    val_sampler = SubsetRandomSampler(val_indices[:12])
    labeled_scene_index = np.arange(106, 134)

    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True
                                     )

    dataloader     =  torch.utils.data.DataLoader(labeled_trainset, batch_size=opt.batch_size, \
                        num_workers=opt.n_cpu, pin_memory = True, collate_fn=collate_fn,sampler=train_sampler)
    val_dataloader =  torch.utils.data.DataLoader(labeled_trainset, batch_size=opt.batch_size, \
                        num_workers=opt.n_cpu, pin_memory = True, collate_fn=collate_fn,sampler=val_sampler)

    models = [encoder]
    models.extend(dense_models)
    models.append(darknet_model)
    all_params= []
    for m in models:
        all_params += list(m.parameters())

    optimizer = torch.optim.Adam(all_params,lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, verbose = True)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "angle_acc",
        "angle",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    best_val_loss = np.inf

    for epoch in range(opt.epochs):
        for model in models:
            model.train()
       
        start_time = time.time()
        train_loss = []
        for batch_i, (imgs,targets,_,_) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i
            targets = compute_yolo_targets(targets)
            imgs = torch.stack(imgs)
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            views =[torch.stack([img[i] for img in imgs]).to(device) for i in range(num_views)] # Get the 6 different views for all batch
            
            encoded = [encoder(view) for view in views] # Pass views to encoder  
            flattened = [e.view(len(imgs),-1) for e in encoded] #flatten all views from encoder

            dense = [dense_models[i](view) for i,view in enumerate(flattened)] # pass 6 views to 6 dense
            concatenated_views = torch.cat(dense,dim=1) # Concat all views channel wise

            loss, outputs = darknet_model(concatenated_views, targets)
            train_loss.append(loss.item())

            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(darknet_model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                formats["angle_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in darknet_model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            darknet_model.seen += imgs.size(0)
            if batch_i %20 == 0:
                print(log_str)

         #Tensorboard logging
            tensorboard_log = {}
            for j, yolo in enumerate(darknet_model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        if(f'Train-{name}' not in tensorboard_log):
                            tensorboard_log[f'Train-{name}'] = {}
                        tensorboard_log[f'Train-{name}'][f'Layer-{j+1}'] = metric
            logger.scalar_summary("Train loss", loss.item(),batches_done)
            logger.list_of_scalars_summary(tensorboard_log, batches_done)


        avg_train_loss = np.mean(train_loss)

        print('Avg train Loss: ', avg_train_loss, "Took ", time.time()-start_time)
        
        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            current_val_loss = evaluate(models, device, val_dataloader,epoch,logger)
            scheduler.step(current_val_loss)
            logger.list_of_scalars_summary({'Loss': {'Train loss':avg_train_loss,\
                                                      'Validation loss' : current_val_loss}}, epoch)
            if(current_val_loss < best_val_loss  or (epoch % opt.checkpoint_interval == 0)):
                torch.save({
                'encoder':models[0].state_dict(),
                'dense1':models[1].state_dict(),
                'dense2':models[2].state_dict(),
                'dense3':models[3].state_dict(),
                'dense4':models[4].state_dict(),
                'dense5':models[5].state_dict(),
                'dense6':models[6].state_dict(),
                'Darknet':models[7].state_dict(),
                'epoch':epoch,
                'optimizer':optimizer,
                },saved_model_dir+'_'+str(epoch)+'.pth')

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss