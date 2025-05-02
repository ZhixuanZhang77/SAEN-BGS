# -*- coding: utf-8 -*-
"""
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from torch import nn, optim
import os
import time
from skimage import io
# from neural_networks import sCNN
from settings import SETTINGS

import cv2
from torch.optim.lr_scheduler import MultiStepLR,StepLR,ReduceLROnPlateau
from skimage.filters import threshold_otsu
from scipy import ndimage
import math
from Evaluation import compare,getStats#measure,readTemporalFile,
from torch.utils.data import DataLoader
from torchvision import transforms

from CDNet2014Dataset import CDNet2014Dataset#, Rescale, ToTensor
import matplotlib.pyplot as plt
device = torch.device(SETTINGS.training.device)
method ='Vgg16'
class_name = 'badWeather'#os.getcwd().split('/')[-1]
import copy
print(' Processor is %s' % (device))
print('method: ',method)
# import the video frames for the training part consisting 30000 frames
 # 10000  # number of samples to be loaded for the training
# loadPath = '/home/liuqi/Desktop/SNN_VAE/CDW_2014/Data/dataset2014/dataset/'+'/'.join(class_name) +'/input' # '../Data/Video_%03d/frames/' % vidNumber
output_path = './output'
os.makedirs(output_path,exist_ok=True)

output_path_weight = os.path.join(output_path,class_name)
os.makedirs(output_path_weight,exist_ok=True)



# VAE training parameters
batch_size =8#200
epoch_num = 100

traindataset = CDNet2014Dataset(root_dir='./dataset/CDNet2014',
                               category=class_name,
                               train=True,
                                valid=False,
                               transform=transforms.Compose([
                                       transforms.Resize((240, 320)),#Rescale((240, 320)),
                                       transforms.ToTensor()
                                       ]),
                                largeScale=True)

# valdataset = CDNet2014Dataset(root_dir='/home/liuqi/Desktop/SNN_VAE/CDW_2014/Data/dataset2014/dataset',
#                                category=class_name,
#                                train=False,
#                               valid=True,
#                                transform=transforms.Compose([
#                                        Rescale((240, 320)),
#                                        ToTensor()
#                                        ]))
testdataset = CDNet2014Dataset(root_dir='./dataset/CDNet2014',
                               category=class_name,
                               train=False,
                              valid=False,
                               transform=transforms.Compose([
                                       transforms.Resize((240, 320)),#Rescale((240, 320)),
                                       transforms.ToTensor()
                                       ]),
                               largeScale=True)
train_loader = DataLoader(traindataset, batch_size=batch_size,
                            shuffle=True, num_workers=8)

# val_loader = DataLoader(valdataset, batch_size=batch_size,
#                             shuffle=False, num_workers=4)

test_loader = DataLoader(testdataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)
# Path parameters

# if not os.path.exists(save_PATH):
#     os.makedirs(save_PATH)
#
# if not os.path.exists(save_rec_PATH):
#        os.mkdir(save_rec_PATH)
PATH_model= os.path.join(output_path_weight,'{0}_CDNet2014_{1}_weight.pth'.format(method,(class_name)))#save_PATH + '/{0}_CDNet2014_{1}'.format(method,(class_name))
PATH_result= os.path.join(output_path_weight,'{0}_CDNet2014_{1}_result.pth'.format(method,(class_name)))#save_PATH + '/{0}_CDNet2014_{1}'.format(method,(class_name))
# Restore
Trainflag =True#
Generate_bmp =True
print('class name:',class_name)
# load  Dataset

print('test:',len(testdataset))
# exit()
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 256, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 32, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 128, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 32, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 2, kernel_size=1, stride=1,padding=0),
            nn.LogSoftmax(1)
        )

        # self.energy =0

    def forward(self, x):
        energy = 0
        x = self.encoder(x)
        # energy += (3*64*9*240*320+64*128*9*120*160+128*256*9*60*80+256*512*9*30*40)
        x = self.decoder(x)
        # energy += (512*32*1*30*40+32*256*16*60*80+
        #            256*32*1*60*80+128*32*16*120*160+
        #            128*32*1*120*160+32*64*16*240*320+
        #            64*2*1*240*320)
        # energy *= x.size(0)
        # energy *=4.6
        # self.energy = energy
        return x


# initialize model
vae = VGG16()
# vae = sVAE(T)
vae.to(device)

# if class_name in ['nightVideos','cameraJitter']:
#     lr = 5e-4
# else:
#     lr = 2e-4
lr = 1e-3#3
# if class_name in ['PTZ']:
#     optimizer = optim.Adam
# else:
optimizer = optim.RMSprop

vae_optimizer = optimizer(vae.parameters(), lr=lr)

# if class_name == 'intermittentObjectMotion':
#     scheduler = MultiStepLR(vae_optimizer, milestones=[10, 40,60,80], gamma=0.9)
# elif class_name == 'PTZ':
# scheduler = ReduceLROnPlateau(vae_optimizer,'min',factor=0.5,patience=4,verbose=True)
# else:
# scheduler = ReduceLROnPlateau(vae_optimizer,'min',factor=0.5,patience=4,verbose=True)#
# # #MultiStepLR(vae_optimizer,milestones=[20,80,120,160],gamma=0.9)
# print(lr)
# loss function


criterion =nn.NLLLoss()# nn.BCELoss()


if Trainflag :
    best_weight = None
    best_fmeasure =0
    best_result = 0
    print("Training...")
    for i in range(epoch_num):
        time_start = time.time()
        loss_vae_value = 0.0
        vae.train()

        for batch_indx, (data,(label,_)) in enumerate(train_loader):
            # update VAE
            x = data.to(device)
            y = label.long().to(device)
            # data_vae=data #if using gpu comment this line!
            vae_optimizer.zero_grad()
            predict_y = vae.forward(x)

            loss_vae = criterion(predict_y,y.squeeze(1))
            loss_vae.backward()
            # print(loss_vae)
            loss_vae_value += loss_vae.item()  # data[0]

            vae_optimizer.step()
        # if class_name == 'PTZ':
        # scheduler.step(loss_vae_value)
        # else:
        # scheduler.step()

        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))
        print('====> Epoch: %d train_Loss : %0.8f' % ((i + 1), loss_vae_value))

        confusionMatrix = torch.zeros(4)
        loss_val_value=0
        vae.eval()
        with torch.no_grad():
            for batch_indx,(data,(label,roi)) in enumerate(test_loader):
                x = data.to(device)
                y = label.long().to(device)
                roi = roi.long().to(device)
                vae_optimizer.zero_grad()
                predict_y = vae.forward(x)
                loss_vae = criterion(predict_y,y.squeeze(1))
                _, predicted = torch.max(predict_y.data, 1)

                gt = y.squeeze(1).float()#*255
                predicted_gt = predicted.float()#*255
                # roi = roi.squeeze(1)
                confusionMatrix += compare(predicted_gt,gt,roi)#*roi
                # for indx in range(len(y)):
                #     gt = y[indx].squeeze().cpu().numpy().astype(np.float32)*255
                #     predicted_gt = predicted[indx].cpu().numpy().astype(np.float32)*255
                #     confusionMatrix+=compare(predicted_gt,gt)
        result = getStats(confusionMatrix)
        print(result,'\n')
        fmeasure = result['fmeasure']
        # scheduler.step(fmeasure)
        weight = vae.state_dict()
        if fmeasure>best_fmeasure:
            best_fmeasure = fmeasure
            best_result = result
            best_weight = copy.deepcopy(weight)
            torch.save(best_weight, PATH_model)
            torch.save(best_result, PATH_result)
        # print('====> Epoch: %d val_Loss : %0.8f' % ((i + 1), loss_vae_value))


    print('best result: ',best_result)
