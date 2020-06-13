import cv2
import torch
import numpy as np
import torch.optim
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
num_views = 6
num_steps = 0

def train(models,device,criterion,train_loader,scheduler,optimizer,epoch,writer,tot_samples, log_interval = 100):
    global num_steps
    global train_sampler

    for model in models:
        model.train()

    train_loss = []
    for batch_idx, (data,target,road_image,extra) in enumerate(tqdm.tqdm(train_loader)):
        road_image = ~(torch.stack(road_image))
        road_image = road_image.to(device)
        optimizer.zero_grad()

        views =[torch.stack([d[i] for d in data]).to(device) for i in range(num_views)] # Get the 6 different views for all batches      
        encoded = [models[0](view) for view in views] # Pass views to encoder  
        flattened = [e.view(len(data),-1) for e in encoded] #flatten all views from encoder
        dense = [models[i+1](view) for i,view in enumerate(flattened)] # pass 6 views to 6 dense models
        concatenated_views = torch.cat(dense,dim=1) # Concat all views channel wise
        decoded_output = models[-1](concatenated_views)
        
        loss = criterion(decoded_output.type(torch.FloatTensor),road_image.type(torch.FloatTensor)).sum(dim=(0)).mean()
        train_loss.append(loss.item())

        writer.add_scalar('Training_loss',loss,num_steps)

        loss.backward()
        optimizer.step()
        num_steps+=1

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                    epoch, batch_idx * len(data), tot_samples, \
                    100. * batch_idx / len(train_loader), loss.item()))

    avg_train_loss = np.mean(train_loss)
    print('Average train loss: ',avg_train_loss)
    return avg_train_loss



def evaluate(models,device,criterion,test_loader,epoch,writer,save_dir,log_interval = 10):
    global num_steps

    for model in models:
        model.eval()

    test_loss = []

    with torch.no_grad():
        for batch_idx, (data,target,road_image,extra) in enumerate(tqdm.tqdm(test_loader)):
            #data : [batch,6,3,256,304]
            #road_image: [batch, 800,800]
            road_image = ~(torch.stack(road_image)) 
            road_image = road_image.to(device)

            views =[torch.stack([d[i] for d in data]).to(device) for i in range(num_views)]
            encoded = [models[0](view) for view in views]
            flattened = [e.view(len(data),-1) for e in encoded]
            dense = [models[i+1](view) for i,view in enumerate(flattened)]
            concatenated_views = torch.cat(dense,dim=1)
            decoded_output = models[-1](concatenated_views)

            loss = criterion(decoded_output.type(torch.FloatTensor),road_image.type(torch.FloatTensor)).sum(dim=(0)).mean()
            test_loss.append(loss.item())

            writer.add_scalar('Validation_loss',loss,num_steps)

            outputs = torch.stack([r.view(1,800,800).type(torch.FloatTensor) for r in decoded_output])       
            outputs = (torch.sigmoid(outputs.detach())).type(torch.FloatTensor)

    
    road_image = torch.stack([r.view(1,800,800).type(torch.FloatTensor) for r in road_image])
    grid_images = torch.cat([outputs, road_image],dim=0)
    grid_images = torchvision.utils.make_grid(grid_images*255,nrow = 6).numpy().astype(np.int).transpose(1,2,0) 
    cv2.imwrite(save_dir+'output_'+str(epoch)+'_'+str(batch_idx)+'.png',grid_images)

        
    avg_test_loss = np.mean(test_loss)
    print('Average test loss: ',avg_test_loss)
    return avg_test_loss