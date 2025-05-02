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
import torchvision.transforms as transforms
import cv2
from torch.optim.lr_scheduler import MultiStepLR,StepLR,ReduceLROnPlateau
from skimage.filters import threshold_otsu
from scipy import ndimage
import math
from Evaluation import compare,getStats #measure,readTemporalFile,
from torch.utils.data import DataLoader
from torchvision import transforms
from snn import ConvBN2d, ConvTransposeBN2d,ConvBN2d_v3,ConvTransposeBN2d_v3
from functional import ZeroExpandInput_CNN,ZeroExpandInput_CNN_v2
from CDNet2014Dataset import CDNet2014Dataset#, Rescale, ToTensor
import torch.nn.functional as F

device = torch.device(SETTINGS.training.device)
method ='SAEN-BGS'   #[badWeather, dynamicBackground] [baseline, PTZ, badWeather, cameraJitter, dynamicBackground, intermittentObjectMotion, lowFramerate, nightVideos, shadow, thermal, turbulence]
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


epoch_num = 100#50

batch_size = 8  # 200

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
print('train',len(traindataset))
# print('val:',len(valdataset))
print('test:',len(testdataset))
# exit()

PATH_init= os.path.join(output_path_weight,'{0}_CDNet2014_{1}_weight.pth'.format('Vgg16',(class_name)))

def inint_weight(path,model):
    state_dict_ann = torch.load(path,map_location='cpu')
    state_dict_snn = model.state_dict()
    ak = list(state_dict_ann.keys())
    sk = list(state_dict_snn.keys())

    for i in range(len(ak)):
        state_dict_snn[sk[i]] = state_dict_ann[ak[i]]
    model.load_state_dict(state_dict_snn)
    return model

class sVGG16(nn.Module):
    def __init__(self, T = 10):
        super(sVGG16, self).__init__()
        # encoder
        self.T =T
        self.e1 = ConvBN2d_v3(3,64, kernel_size=3, stride=1,padding=1,pooling=2)
        self.e2 = ConvBN2d_v3(64, 128, kernel_size=3, stride=1,padding=1,pooling=2)
        self.e3 = ConvBN2d_v3(128, 256, kernel_size=3, stride=1,padding=1,pooling=2)
        self.e4 = ConvBN2d_v3(256, 512, kernel_size=3, stride=1,padding=1,pooling=1)

        # decoder
        self.d1 = ConvBN2d_v3(512, 32, kernel_size=1, stride=1,padding=0,pooling=1)
        self.d2 = ConvTransposeBN2d_v3(32, 256, kernel_size=4, stride=2,padding=1,pooling=1)

        self.d3 = ConvBN2d_v3(256, 32, kernel_size=1, stride=1,padding=0,pooling=1)
        self.d4 = ConvTransposeBN2d_v3(32, 128, kernel_size=4, stride=2,padding=1,pooling=1)

        self.d5 = ConvBN2d_v3(128, 32, kernel_size=1, stride=1,padding=0,pooling=1)
        self.d6 = ConvTransposeBN2d_v3(32, 64, kernel_size=4, stride=2,padding=1,pooling=1)

        # self.output=nn.Sequential(
        #     nn.Conv2d(64, 2, kernel_size=1, stride=1,padding=0),
        #     nn.LogSoftmax(1)
        # )
        self.output = ConvBN2d_v3(64, 2, kernel_size=1, stride=1, padding=0,last_layer=True)
        # nn.init.uniform_(self.output.conv2d.weight, -0.1, 0.1)
        # nn.init.uniform_(self.output.conv2d.bias, -0.1, 0.1)


        self.energy =0

    def forward(self, x):
        x, x_st, x_sc = ZeroExpandInput_CNN_v2.apply(x, self.T)

        x, x_st, x_sc = self.e1(x, x_st, x_sc)

        x, x_st, x_sc = self.e2(x, x_st, x_sc)
        x, x_st, x_sc = self.e3(x, x_st, x_sc)
        x, x_st, x_sc = self.e4(x, x_st, x_sc)

        x, x_st, x_sc = self.d1(x, x_st, x_sc)
        x, x_st, x_sc = self.d2(x, x_st, x_sc)
        x, x_st, x_sc = self.d3(x, x_st, x_sc)
        x, x_st, x_sc = self.d4(x, x_st, x_sc)
        x, x_st, x_sc = self.d5(x, x_st, x_sc)
        x, x_st, x_sc = self.d6(x, x_st, x_sc)
        x_out, _, x_sc= self.output(x, x_st, x_sc)
        return F.log_softmax(x_out,dim=1), F.log_softmax(x_sc,dim=1)


# initialize model
vae = sVGG16()

vae = inint_weight(PATH_init, vae)# initial weight by weight of counterpart ANN

vae.to(device)

# vae.load_state_dict(torch.load(PATH_model))
if class_name in ['nightVideos','cameraJitter']:
    lr = 5e-4
else:
    lr = 2e-4

# if class_name in ['PTZ']:
#     optimizer = optim.Adam
# else:
optimizer = optim.RMSprop

vae_optimizer = optimizer(vae.parameters(), lr=lr,momentum=0.9)

# scheduler = StepLR(vae_optimizer,5,0.9)#
print(lr)



criterion = nn.NLLLoss()# nn.BCELoss()
alpha=0.9

if Trainflag :
    best_weight = None
    best_fmeasure =0
    best_result = 0
    print("Training...")
    for i in range(epoch_num):
        time_start = time.time()
        loss_vae_value = 0.0
        vae.train()

        for batch_indx, (data,label) in enumerate(train_loader):
            # update VAE
            x = data.to(device)
            y = label.long().to(device)
            # data_vae=data #if using gpu comment this line!
            vae_optimizer.zero_grad()
            pred_y, pred_y_sc = vae.forward(x)

            loss_vae = (1 - alpha) * criterion(pred_y, y.squeeze(1)) + alpha * criterion(pred_y_sc, y.squeeze(1))

            loss_vae.backward()
            loss_vae_value += loss_vae.item()

            vae_optimizer.step()
        # if class_name == 'PTZ':
        #     scheduler.step(loss_vae_value)
        # else:
        #     scheduler.step()

        time_end = time.time()
        print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))
        print('====> Epoch: %d train_Loss : %0.8f' % ((i + 1), loss_vae_value))

        confusionMatrix = torch.zeros(4)
        loss_val_value=0
        vae.eval()
        with torch.no_grad():
            for batch_indx,(data,label) in enumerate(test_loader):
                x = data.to(device)
                y = label.long().to(device)
                vae_optimizer.zero_grad()
                pred_y, pred_y_sc = vae.forward(x)

                # loss_vae = criterion(predict_y,y.squeeze(1))
                _, predicted = torch.max(pred_y_sc.data, 1)

                gt = y.squeeze(1).float()*255
                predicted_gt = predicted.float()*255

                confusionMatrix += compare(predicted_gt,gt)#*roi

        result = getStats(confusionMatrix)
        print(result,'\n')
        fmeasure = result['fmeasure']
        weight = vae.state_dict()
        if fmeasure>best_fmeasure:
            best_fmeasure = fmeasure
            best_result = result
            best_weight = copy.deepcopy(weight)
            torch.save(best_weight, PATH_model)
            torch.save(best_result, PATH_result)

    print('best result: ',best_result)


