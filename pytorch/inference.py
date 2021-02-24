import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

import matplotlib.pyplot as plt

workers = 16
epochs = 25
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

global args, best_prec1, weight_decay, momentum

model = ITrackerModel()
model = torch.nn.DataParallel(model)
model.cuda()
imSize=(224,224)
cudnn.benchmark = True   

# Load model
saved = load_checkpoint()
if saved:
    print('Loading checkpoint for epoch %05d with loss %.5f' % (saved['epoch'], saved['best_prec1']))
        state = saved['state_dict']
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
        epoch = saved['epoch']
        best_prec1 = saved['best_prec1']
    else:
        print('Warning: Could not read checkpoint!')

model.eval()

# Get ready for inference
imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

image = image_loader("./eye.png")


# def show_results(img_face, img_leye, img_reye, gaze, output):
#     fig = plt.figure(figsize=(12,12))
#     ax1 = fig.add_subplot(3,3,(2,2))
#     ax1.imshow(img_face.permute(1, 2, 0).cpu())
#     ax2 = fig.add_subplot(3,3,3)
#     ax2.imshow(img_reye.permute(1, 2, 0).cpu())
#     ax3 = fig.add_subplot(3,3,6)
#     ax3.imshow(img_leye.permute(1, 2, 0).cpu())
    
#     # Plot gaze vectors
#     ax4 = fig.add_subplot(3,3,5)
#     gaze = np.array(gaze.cpu())
#     output = np.array(output.cpu())

#     V = np.array([gaze, output])
#     origin = np.array([[0, 0],[0, 0]]) # origin point

#     q_handle = ax4.quiver(*origin, gaze[0], gaze[1], color='r', angles='xy', scale_units='xy', scale=1, label='True')
#     q_handle = ax4.quiver(*origin, output[0], output[1], color='b', angles='xy', scale_units='xy', scale=1, label='Prediction')
#     ax4.set_xlim([-20,20])
#     ax4.set_ylim([-20,20])

#     ax4.legend()

# imIdx = 57

# imFace_ = imFace[imIdx, :, :, :]
# imEyeL_ = imEyeL[imIdx, :, :, :]
# imEyeR_ = imEyeR[imIdx, :, :, :]
# gaze_ = gaze[imIdx, :]
# output_ = output[imIdx, :]
# print(np.array(gaze_.cpu()))
# print(output_.cpu())
# imFace_.shape
# show_results(imFace_, imEyeL_, imEyeR_, gaze_, output_)