import torch
import time
import numpy as np
from models import *
from utils.logger import *
from utils.utils import *
from encoder_model import EncoderModel, DenseModel

num_views = 6
def evaluate(models, device, dataloader, epoch,logger):
    for model in models:
        model.eval()

    start_time = time.time()
    val_loss = []

    with torch.no_grad():   
        for batch_i, (imgs,targets,_,_) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i
            targets = compute_yolo_targets(targets)
            imgs = torch.stack(imgs)
            imgs = Variable(imgs.to(device),requires_grad = False)
            targets = Variable(targets.to(device), requires_grad=False)

            views =[torch.stack([img[i] for img in imgs]).to(device) for i in range(num_views)] # Get the 6 different views for all batch
            
            encoded = [models[0](view) for view in views] # Pass views to encoder  
            flattened = [e.view(len(imgs),-1) for e in encoded] #flatten all views from encoder

            dense = [models[i+1](view) for i,view in enumerate(flattened)] # pass 6 views to 6 dense
            concatenated_views = torch.cat(dense,dim=1) # Concat all views channel wise

            loss, outputs = models[-1](concatenated_views, targets)
            val_loss.append(loss.item())
            
            #Tensorboard logging
            tensorboard_log = {}
            for j, yolo in enumerate(models[-1].yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        if(f'Validation-{name}' not in tensorboard_log):
                            tensorboard_log[f'Validation-{name}'] = {}
                        tensorboard_log[f'Validation-{name}'][f'Layer-{j+1}'] = metric
            logger.scalar_summary("Val loss", loss.item(),batches_done)
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

    avg_val_loss = np.mean(val_loss)
    print('Average Validation Loss: ', avg_val_loss, "Took ", time.time()-start_time)
    return avg_val_loss