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
from models import MPS
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
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

# model_names = sorted(name for name in vgg.__dict__
#     if name.islower() and not name.startswith("__")
#                      and name.startswith("vgg")
#                      and callable(vgg.__dict__[name]))

large_neg=-1e6
best_prec1 = 0  


def test(sequence, model,save_dir,writer=None,year=2017):
    """
    Run evaluation
    """
    # switch to evaluate mode
    model.eval()
    #end = time.time()
    if not os.path.exists(os.path.join(save_dir,sequence)):
        os.makedirs(os.path.join(save_dir,sequence))

    composed_transforms_ts=transforms.Compose([dataset_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),dataset_utils.ToTensor()])    
    datasetloader = DavisDataset_video_test(sequence=sequence,transform=composed_transforms_ts,year=year)

    test_data = torch.utils.data.DataLoader(datasetloader, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    end=time.time()
    #tmp = [None for i in range(0,datasetloader.cls_num +1)]
    for i, sample_batched in enumerate(test_data):
        img_input, first_img,first_mask = sample_batched["image"],sample_batched["first_image"],sample_batched["first_mask"]

        input_var = torch.autograd.Variable(img_input).cuda()
        first_img_var= torch.autograd.Variable(first_img).cuda()
        first_mask = torch.autograd.Variable(first_mask).cuda()
        if i==0:
            output_mask = first_mask
        else:
            with torch.no_grad():
                output,embedding= model(input_var,first_img_var,first_mask,output_mask,int(first_mask.max()),None)
            output_mask = torch.max(output,1)[1].unsqueeze(1).float()
            #embedding 结果保存
            if not os.path.exists(os.path.join(config.save_dir,'embedding results',sequence)):
                os.makedirs(os.path.join(config.save_dir,'embedding results',sequence))
            for j in range(int(first_mask.max())+1):
                embedding_temp = embedding[0,j]
                embedding_image = Image.fromarray(embedding_temp.data.cpu().numpy()*255).convert("L")
                embedding_image.save(os.path.join(config.save_dir,'embedding results',sequence, "embedding_"+str(i)+"_"+str(j) + '.png'))

        final_pil_image = Image.fromarray(np.uint8(output_mask.data.cpu().numpy()[0][0]))
        final_pil_image.putpalette(datasetloader.color_palette)
        final_pil_image.save(os.path.join(save_dir, sequence, str(i).zfill(5) + '.png'))
        
    return (time.time()-end),len(datasetloader)#datasetloader.cls_num*len(datasetloader)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global  best_prec1


    # Check the save_dir exists or not
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    model = MPS.MPS()
    #model =  VSHM.VSHM()
    # model.features = torch.nn.DataParallel(model.features)

    model.cuda()
    epoch=0
    # optionally resume from a checkpoint or begin at start_epoch
    if config.resume:
        checkpoint_file=''
        if config.start_epoch>=0 and os.path.isfile(os.path.join(config.save_dir,'checkpoint_'+str(int(config.start_epoch))+'.pth')):
            checkpoint_file=os.path.join(config.save_dir,'checkpoint_'+str(int(config.start_epoch))+'.pth')
        else:
            print('not found start_epoch: '+str(config.start_epoch))
            if glob.glob(os.path.join(config.save_dir,'*.pth')):
                checkpoint_file=sorted(glob.glob(os.path.join(config.save_dir,'*.pth')),key=lambda f: os.stat(f).st_mtime)[-1]
            else:
                print ("=> no checkpoint found at '{}'".format(config.save_dir))
        if checkpoint_file:
            # print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            config.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_file, checkpoint['epoch']))
        else:
            print("=> a new train... ")
    else:
        print("=> a new train... ")   

    cudnn.benchmark = True

    print("{0:-^80}".format("start training"))
    print('start time:',datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    total_time=0
    total_num=0
    if config.year==2017:
        #with open("/home/ljj/data/davis-2017/data/DAVIS/ImageSets/2017/test-dev.txt") as f:
        with open(config.davis_datadir+"ImageSets/2017/val.txt") as f:
            seqs=f.readlines()
    else:
        with open(config.davis_datadir+"ImageSets/2016/val.txt") as f:
            seqs=f.readlines()
    for sequence in seqs:
        print("processing sequence:",sequence.strip())
        time_tmp,num=test(sequence.strip(), model,config.res_savedir,year=config.year)
        #time_tmp,num=test("shooting", model,"result3",year=config.year)
        total_time+=time_tmp
        total_num +=num
        print(time_tmp/num)
    print(total_time/total_num)
if __name__ == '__main__':
    main()