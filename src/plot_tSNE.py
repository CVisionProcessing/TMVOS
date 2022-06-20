import sys
sys.path.append("..")
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
# Tensorboard include
from models import vgg
from models import MPS2,MPS,MPS3,MPS4
import config
import glob
from PIL import Image
import datetime
import dataset_utils
from dataset_utils import DavisDataset_video_test
import utils
from torchvision.utils import make_grid
import numpy as np
import math
import cv2
import sklearn
from sklearn.manifold import TSNE 
from sklearn.datasets import load_digits # For the UCI ML handwritten digits dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

def plot(x, colors,save_dir,i,num):
    # Create a scatter plot.
    #f = plt.figure(figsize=(8, 8))
    type1 = []
    for i in range(3):
        type1.append(x[colors==i,:])
    #print(type1)
    g1 = plt.scatter(type1[0][:,0],type1[0][:,1],c="red",lw=0,s=40)
    g2 = plt.scatter(type1[1][:,0],type1[1][:,1],c="yellow",lw=0,s=40)
    g3 = plt.scatter(type1[2][:,0],type1[2][:,1],c="blue",lw=0,s=40)
    plt.legend(handles=[g1, g2, g3], labels=['0', '1', '2'])
    plt.savefig("./"+save_dir+"/"+str(i).zfill(5) + "_"+str(num)+".png")


def test(sequence, model1, model2,save_dir,writer=None,year=2017):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model1.eval()
    model2.eval()
    #end = time.time()
    if not os.path.exists(os.path.join(save_dir,sequence)):
        os.makedirs(os.path.join(save_dir,sequence))

    composed_transforms_ts=transforms.Compose([dataset_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),dataset_utils.ToTensor()])    
    datasetloader = DavisDataset_video_test(sequence=sequence,transform=composed_transforms_ts,year=year)

    test_data = torch.utils.data.DataLoader(datasetloader, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    end=time.time()
    #tmp = [None for i in range(0,datasetloader.cls_num +1)]
    for i, sample_batched in enumerate(test_data):
        img_input, first_img,first_mask,target = sample_batched["image"],sample_batched["first_image"],sample_batched["first_mask"],sample_batched["label"]

        input_var = torch.autograd.Variable(img_input).cuda()
        first_img_var= torch.autograd.Variable(first_img).cuda()
        first_mask = torch.autograd.Variable(first_mask).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        if i==0:
            output_mask1 = first_mask
            output_mask2 = first_mask
        else:
            with torch.no_grad():
                output1,embedding1= model1(input_var,first_img_var,first_mask,output_mask1,int(first_mask.max()),None)
                output2,embedding2 = model2(input_var,first_img_var,first_mask,output_mask2,int(first_mask.max()),None)

            target_var = F.interpolate(target_var,size=embedding1.size()[2:],mode="nearest",align_corners=None)

            label = np.array(target_var.unsqueeze(0).unsqueeze(0).view(-1))
            print(embedding1.shape,embedding2.shape)
            
            embedding1 = np.array(embedding1.permute(0,2,3,1).unsqueeze(0).view(-1,64))
            embedding2 = np.array(embedding2.permute(0,2,3,1).unsqueeze(0).view(-1,64))
            print(embedding1.shape,embedding2.shape)

            tSNE1 = TSNE(perplexity=30).fit_transform(embedding1)
            tSNE2 = TSNE(perplexity=30).fit_transform(embedding2)

            plot(tSNE1, label,save_dir,i,1)
            plot(tSNE2, label,save_dir,i,2)

            output_mask1 = torch.max(output1,1)[1].unsqueeze(1).float()
            output_mask2 = torch.max(output2,1)[1].unsqueeze(1).float()

    return (time.time()-end),len(datasetloader)#datasetloader.cls_num*len(datasetloader)

def main():
    global  best_prec1

    model1 = MPS.MPS()
    model2 = MPS2.MPS()
    #model =  VSHM.VSHM()
    # model.features = torch.nn.DataParallel(model.features)

    model1.cuda()
    model2.cuda()

    checkpoint_file1=os.path.join("save_dir1",'checkpoint_'+str(int(9))+'.pth')
    checkpoint1 = torch.load(checkpoint_file1)
    model1.load_state_dict(checkpoint1['state_dict'])

    checkpoint_file2=os.path.join("save_dir2",'checkpoint_'+str(int(10))+'.pth')
    checkpoint2 = torch.load(checkpoint_file2)
    model2.load_state_dict(checkpoint2['state_dict'])

    time_tmp,num=test("horsejump-high", model1,model2,"plot_result",year=2017)

if __name__ == '__main__':
    main()