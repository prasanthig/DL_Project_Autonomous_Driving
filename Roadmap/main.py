import os
import time
import random
from datetime import datetime

import numpy as np

import torch
import torch.optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
writer = SummaryWriter()


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box, convert_map_to_road_map, convert_map_to_lane_map
from model import EncoderModel, DenseModel,DecoderModel
from train import train, evaluate

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

seed = 1008
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Set up your device
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
print(device)

if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--log_interval", type=int, default=20, help="interval between saving model weights")
    parser.add_argument("--run_specs", type=str, default="default", help="details of the model")
    parser.add_argument("--model",type=str, default="_121", help="path to model state_dict")
    parser.add_argument("--data_dir",type=str, default="./data", help="path to data folder")
    opt = parser.parse_args()
    print(opt)

    # All the images are saved in image_folder
    # All the labels are saved in the annotation_csv file
    image_folder = opt.data_dir
    annotation_csv =  f'{opt.data_dir}/annotation.csv'
    batch_size = opt.batch_size

    num_views = 6
    random_run = datetime.now()
    test_image_dir = './TestImages/'+opt.run_specs+str(random_run)+'/'
    saved_model_dir = './Models/Run_with_graph_'+opt.run_specs+str(random_run)+'/'

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)


    # The scenes from 106 - 133 are labeled
    # You should devide the labeled_scene_index into two subsets (training and validation)
    labeled_scene_index = np.arange(106, 134)

    #90 - 10 % split. Fir
    num_train = 24
    num_val = 4
    total_labeled = 28
    num_samples = 126

    sample_indices = np.arange(0,28*126) # Tot: 3528
    split = num_train * num_samples # Train-val split: 3024 - 504
    train_indices, val_indices = sample_indices[:split], sample_indices[split:]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    #Normalize images
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5563, 0.6024, 0.6325), (0.3195, 0.3271, 0.3282))    
    ])


    #Get dataset
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index,
                                      transform=transform,
                                      extra_info=True
                                     )

    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=opt.batch_size, num_workers=opt.n_cpu, collate_fn=collate_fn,sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=opt.batch_size, num_workers=opt.n_cpu, collate_fn=collate_fn,sampler=val_sampler)



    load_model = False
    load_path = opt.model

    tot_epochs = opt.epochs
    init_epochs = 0

    lr = 1e-3

    input_shape = np.array([3,256,304])
    encoded_shape = np.array([1,16,16,19])
    bottleneck_channels = 16*6
    encoder = EncoderModel(input_shape[0])
    dense_models = [DenseModel(encoded_shape) for i in range(num_views)]
    decoder = DecoderModel(bottleneck_channels)


    if load_model:
        checkpoint = torch.load(load_path)
        encoder.load_state_dict(checkpoint['encoder'])
        for i,model in enumerate(dense_models):
            model.load_state_dict(checkpoint['dense'+str(i+1)]) 
        decoder.load_state_dict(checkpoint['decoder'])
        init_epochs = checkpoint['epoch']


    models = [encoder]
    models.extend(dense_models)
    models.append(decoder)

    for model in models:
        model.to(device)


    all_params= []
    for model in models:
        all_params += list(model.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    best_loss = np.inf

    for epoch in range(init_epochs+1,init_epochs+tot_epochs+1):
        train_loss = train(models, device,criterion, trainloader,scheduler,optimizer,epoch,writer,len(train_sampler),log_interval=opt.log_interval)
        validation_loss = evaluate(models,device,criterion,val_loader,epoch,writer,test_image_dir,log_interval=opt.log_interval)

        scheduler.step(validation_loss)
        writer.add_scalars("Loss",{'Train-loss':train_loss, 'Validation-loss':validation_loss},epoch)

        if (validation_loss < best_loss or epoch % opt.checkpoint_interval == 0):
            torch.save({
                'encoder':models[0].state_dict(),
                'dense1':models[1].state_dict(),
                'dense2':models[2].state_dict(),
                'dense3':models[3].state_dict(),
                'dense4':models[4].state_dict(),
                'dense5':models[5].state_dict(),
                'dense6':models[6].state_dict(),
                'decoder':models[7].state_dict(),
                'epoch':epoch,
                'optimizer':optimizer,
                },saved_model_dir+'_'+str(epoch))

            if validation_loss < best_loss:
                best_loss = validation_loss

    writer.close()
