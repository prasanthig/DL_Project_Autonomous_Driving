import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModel(nn.Module):
    def __init__(self,input):
        super(EncoderModel,self).__init__()
        self.input_channel = input

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channel,128,kernel_size = 3, padding=1), # 2^7
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128,64,kernel_size = 3, padding=1), # 2^6
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64,32,kernel_size = 3, padding=1), # 2^5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32,16,kernel_size = 3, padding=1), # 2^4
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2) 
        )
        
    def forward(self,x):
        return self.encoder(x)
        
class DenseModel(nn.Module):
    def __init__(self,input):
        super(DenseModel,self).__init__()
        self.input_dim = input
        self.flattened_dim = self.input_dim[1]*self.input_dim[2]*self.input_dim[3]

        self.dense = nn.Sequential(

              nn.Linear(self.flattened_dim, 64),
              nn.ReLU(),
              nn.Dropout(),

              nn.Linear(64,64),
              nn.ReLU(),
              nn.Dropout(),

              nn.Linear(64, self.flattened_dim),
              nn.ReLU()
             )
        self.conv = nn.Conv2d(16, 16,kernel_size=3,padding=1)
        self.elu = nn.ReLU()

    def forward(self,x):
        output = self.dense(x)
        output = output.view(-1,self.input_dim[1],self.input_dim[2],self.input_dim[3])
        output = self.elu(self.conv(output))
        return output

class DecoderModel(nn.Module):
    def __init__(self,input):
        super(DecoderModel,self).__init__()
        self.input_channel = input
        self.decoder = nn.Sequential(
            nn.Conv2d(self.input_channel,16,kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(16,32,kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(32,64,kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(64,128,kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor = 2),

            nn.Conv2d(128,128,kernel_size = 3, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor = 2)
        )
        self.conv = nn.Conv2d(128,1,kernel_size=3,padding=1)
        
    def forward(self,x):
        output = self.decoder(x)
        #print(output.shape)
        output = nn.Upsample(size=(800,800))(output)
        #print(output.size())
        output = self.conv(output)
        return output.view(-1,800,800) 