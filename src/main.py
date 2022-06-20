import sys
sys.path.append("..")
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
# Tensorboard include
from tensorboardX import SummaryWriter
import torchvision.datasets as datasets
from models import MPS
import config
import glob
from PIL import Image


import datetime
import dataset_utils
from dataset_utils import COCO,DavisDataset_video1,DavisDataset_video_val,DavisDataset_per_video_val
import utils
from torchvision.utils import make_grid
import socket
import timeit
import numpy as np
import curses
import math
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

# model_names = sorted(name for name in vgg.__dict__
#     if name.islower() and not name.startswith("__")
#                      and name.startswith("vgg")
#                      and callable(vgg.__dict__[name]))


best_prec1 = 0  

def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    batch_losses = AverageMeter()
    prcurve = []
    prec=[]
    recall=[]
    if config.train_index['mae']:
        mae = AverageMeter()
    if config.train_index['iou']:
        top1 = AverageMeter()
    if config.train_index['map']:
        MAP = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    print(len(train_loader))
    for i, sample_batched in enumerate(train_loader):

        lr_t=0
        for param_group in optimizer.param_groups:
            lr_t = param_group['lr']
            break
        writer.add_scalar('learning_rate_steps', lr_t, (len(train_loader)*epoch+i)*config.batch_size)
        img_input, target ,first_img, spa_prior,target_first,spa_img= sample_batched['image'], sample_batched['label'], sample_batched['first_image'], sample_batched['spa_prior'],sample_batched['target_first'],sample_batched["spa_img"]
        #spa_prior = random_Gaussianfilter(spa_prior)
        # measure data loading time
        
        if target.numpy().sum()==0:
            continue
        if target_first.numpy().sum()==0:
            continue


        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(img_input).cuda()
        first_img_var= torch.autograd.Variable(first_img).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        spa_prior_var = torch.autograd.Variable(spa_prior).cuda()
        target_first_var = torch.autograd.Variable(target_first).cuda()
        spa_img_var = torch.autograd.Variable(spa_img).cuda()
        if config.half:
            input_var = input_var.half()
            first_img_var=first_img_var.half()
        # compute output

        output,embedding= model(input_var,first_img_var,target_first_var,spa_prior_var,int(target.max()),spa_img_var)
        #print(output.size())
        loss = criterion(output, target_var, size_average=True, batch_average=True)
        #print(loss)

       
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
            

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        if config.train_index['iou']:
            i_iou,num = utils.get_iou(output, target_var)
            top1.update(i_iou, img_input.size(0))
        if config.train_index['mae']:
            _mae = utils.get_mae(output, target_var)
            mae.update(_mae)
        if config.train_index['pr']:
            curPrec, curRecall = utils.get_prcurve(target_var, output)
            prec.append(curPrec)
            recall.append(curRecall)
        
        losses.update(loss.item(), img_input.size(0))
        batch_losses.update(loss.item(), img_input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.print_freq == 0:
            writer.add_scalar('train/total_loss_iter', loss.item(), i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            
            print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(train_loader)), end='\t')
            print('Time {batch_time.val:.4f} ({batch_time.sum:.4f})'.format(batch_time=batch_time), end='\t')
            print('Loss {batch_loss.avg:.4f} ({loss.avg:.4f})'.format(batch_loss=batch_losses, loss=losses), end='\t')
            if config.train_index['iou']:
                writer.add_scalar('train/train_IOU_iter', top1.val, i * config.batch_size + len(train_loader)*config.batch_size * epoch)
                print('IOU@1 {top1.val:.4f} ({top1.avg:.4f})'.format(top1=top1))
            if config.train_index['mae']:
                print('MAE {mae.val:.4f} ({mae.avg:.4f})'.format(mae=mae))
            '''
            embedding = embedding[0][0].detach().cpu().numpy()
            attention1 = attention1[0][0].detach().cpu().numpy()
            attention2 = attention2[0][0].detach().cpu().numpy()
            attention3 = attention3[0][0].detach().cpu().numpy()
            writer.add_image('{}'.format("train/embedding"), embedding, global_step=i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            writer.add_image('{}'.format("train/attention1"), attention1, global_step=i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            writer.add_image('{}'.format("train/attention2"), attention2, global_step=i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            writer.add_image('{}'.format("train/attention3"), attention3, global_step=i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            writer.add_image('{}'.format("train/target"), target_var[0][0].detach().cpu().numpy(), global_step=i * config.batch_size + len(train_loader)*config.batch_size * epoch)
            '''
            batch_losses.reset()

    writer.add_scalar('train/total_loss_epoch', losses.avg, epoch )
    if config.train_index['iou']:
        writer.add_scalar('train/train_IOU_epoch', top1.avg, epoch )
    if config.train_index['mae']:
        writer.add_scalar('train/train_MAE', mae.avg, epoch)
    if config.train_index['pr']:
        pr = utils.PR_Curve(prec, recall)
        print('F-max: ', '%.4f' % pr['curScore'], end='\t')
        writer.add_scalar('test/F-max', pr['curScore'], epoch)

    global_step=len(train_loader)*config.batch_size*(epoch+1)
    grid_image = make_grid(img_input[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image('train/Image', grid_image, global_step)
    grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                           range=(0, 255))
    writer.add_image('train/Predicted label', grid_image, global_step)
    grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(target_var[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
    writer.add_image('train/Groundtruth label', grid_image, global_step)

    print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    return losses.avg


def val_per_video(model, criterion, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    batch_losses = AverageMeter()
    batch_top1 = AverageMeter()
    seq_iou = AverageMeter()
    total_iou = AverageMeter()
    prcurve = []
    prec=[]
    recall=[]
    if config.train_index['mae']:
        mae = AverageMeter()
    if config.train_index['iou']:
        top1 = AverageMeter()
    if config.train_index['map']:
        MAP = AverageMeter()
    # switch to train mode
    model.eval()
    #################################################################################
    #dataloader
    total_dataloader=0
    composed_transforms = transforms.Compose([
        #dataset_utils.FixedResize(size=(config.width, config.height)),
        dataset_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        dataset_utils.ToTensor()])
    if config.year==2017:
        with open(config.davis_datadir+"ImageSets/2017/val.txt") as f:
            seqs=f.readlines()
    else:
        with open(config.davis_datadir+"ImageSets/2016/val.txt") as f:
            seqs=f.readlines()
    for seq in seqs:
        if config.year==2017:
            cls_num = np.array(Image.open(os.path.join(config.davis_datadir+'Annotations/480p',seq.strip(),"00000.png"))).max()
        else:
            cls_num = 1
        fine_data = DavisDataset_per_video_val(transform=composed_transforms,sequence=seq.strip(),obj_id=None,year=config.year)
        train_loader = torch.utils.data.DataLoader(fine_data, batch_size=1, shuffle=False)

#################################################################################
        result_list = []
        end = time.time()
        total_dataloader+=len(train_loader)
        for i, sample_batched in enumerate(train_loader):


            img_input, target ,first_img,target_first= sample_batched['image'], sample_batched['label'], sample_batched['first_image'],sample_batched['first_mask']
        
            # measure data loading time

            """
            if target.numpy().sum()==0:
                continue
            """

            data_time.update(time.time() - end)
        
            if i==0:
                output_tmp = torch.autograd.Variable(target_first).cuda()
                spa_img = torch.autograd.Variable(first_img).cuda()
            input_var = torch.autograd.Variable(img_input).cuda()
            first_img_var= torch.autograd.Variable(first_img).cuda()
            target_var = torch.autograd.Variable(target).cuda()
            target_first_var = torch.autograd.Variable(target_first).cuda()
            if config.half:
                input_var = input_var.half()
                first_img_var=first_img_var.half()
            # compute output
            with torch.no_grad():
                output,embedding= model(input_var,first_img_var,target_first_var,output_tmp,cls_num,spa_img)
                loss = criterion(output, target_var, size_average=True, batch_average=True)
                softmax = nn.Softmax(dim=1)
                output = softmax(output)

            spa_img = input_var
            output_tmp = torch.max(output,1)[1].float().unsqueeze(0)
            output = output.float()
            loss = loss.float()


            # measure accuracy and record loss
            if i!=len(train_loader)-1:
                if config.train_index['iou']:
                    i_iou,num= utils.get_iou(output, target_var)
                    top1.update(i_iou, img_input.size(0)*num)
                    batch_top1.update(i_iou,img_input.size(0)*num)
                    result_list.append(i_iou)
                if config.train_index['mae']:
                    _mae = utils.get_mae(output, target_var)
                    mae.update(_mae)
                if config.train_index['pr']:
                    curPrec, curRecall = utils.get_prcurve(target_var, output)
                    prec.append(curPrec)
                    recall.append(curRecall)
            
            losses.update(loss.item(), img_input.size(0))
            batch_losses.update(loss.item(), img_input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i==len(train_loader)-1:
                total_iou.update(batch_top1.avg,cls_num)
                print('epoch:',epoch,end='\t')
                print('seq:',seq,end='\t')
                print('Time {batch_time.val:.4f} ({batch_time.sum:.4f})'.format(batch_time=batch_time), end='\t')
                print('Loss {batch_loss.avg:.4f} ({loss.avg:.4f})'.format(batch_loss=batch_losses, loss=losses), end='\t')
                if config.train_index['iou']:
                    print('IOU@1 {batch_top1.avg:.4f} ({top1.avg:.4f},{total_iou.avg:.4f})'.format(batch_top1=batch_top1,top1=top1,total_iou=total_iou))
                if config.train_index['mae']:
                    print('MAE {mae.val:.4f} ({mae.avg:.4f})'.format(mae=mae))
                batch_losses.reset()
                batch_top1.reset()
                

    writer.add_scalar('test/total_loss_epoch', losses.avg, epoch )
    if config.train_index['iou']:
        writer.add_scalar('test/test_IOU_epoch', total_iou.avg, epoch )

    global_step=total_dataloader*config.batch_size*(epoch+1)
    grid_image = make_grid(img_input[:6].clone().cpu().data, 6, normalize=True)
    writer.add_image('test/Image', grid_image, global_step)
    grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(output[:6], 1)[1].detach().cpu().numpy()), 6, normalize=False,
                           range=(0, 255))
    writer.add_image('test/Predicted label', grid_image, global_step)
    grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(target_var[:6], 1).detach().cpu().numpy()), 6, normalize=False, range=(0, 255))
    writer.add_image('test/Groundtruth label', grid_image, global_step)

    print(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    return losses.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # lr = config.lr * (0.5 ** (epoch // 30))
    print("{0:~^40}".format(' adjust lr to half '))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] *0.9


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    global  best_prec1

    

    # Check the save_dir exists or not
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)


    model =MPS.MPS()
    #多gpu加速
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

    

    # Logging into Tensorboard
    log_dir = os.path.join(config.save_dir, 'log', datetime.datetime.now().strftime('%b%d_%H:%M:%S') + '_begin_epoch' + str(epoch))
    writer = SummaryWriter(log_dir=log_dir)

    cudnn.benchmark = True

    composed_transforms_tr = transforms.Compose([
        dataset_utils.RandomSized([config.width,config.height]),
        dataset_utils.RandomRotate(15),
        dataset_utils.RandomHorizontalFlip(),
        dataset_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #dataset_utils.Erode(),
        dataset_utils.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        dataset_utils.FixedResize(size=(config.width, config.height)),
        dataset_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        dataset_utils.ToTensor()])


    train_data = DavisDataset_video1(transform=composed_transforms_tr)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)
    
    train_data1 = COCO(transform=composed_transforms_tr)
    train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True)
    
    voc_val = DavisDataset_video_val(transform=composed_transforms_ts)
    val_loader = torch.utils.data.DataLoader(voc_val, batch_size=config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=True)
    

    
    # define loss function (criterion) and optimizer

    criterion = utils.cross_entropy2d

    if config.half:
        model.half()
        criterion.half()

    if config.optim=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    elif config.optim=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    else:
        raise ValueError('config.optim choose error!')

    if config.evaluate:
        for dataset in ['DUT-OMRON','DUTS-TE','HKU-IS','ECSSD','MSRA-B','PASCAL-S','SOD']:
            voc_val = Salientdataset(dataset, transform=composed_transforms_ts)
            val_loader = torch.utils.data.DataLoader(voc_val, batch_size=4, shuffle=False, num_workers=config.workers, pin_memory=True)
            print('>>== eval for '+dataset)
            validate(val_loader, model, criterion, epoch, writer = writer)
        print('eval done!')
        return
    
    print("{0:-^80}".format("start training"))
    print('start time:',datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    train_loss_min=float("inf")
    train_loss_delay_adjust=0
    val_loss_min=float("inf")
    val_loss_delay_adjust=0

    num_img_tr = len(train_loader)
    num_img_ts = len(val_loader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    prec1,val_loss = 0, float("inf")
    # for epoch in range(config.start_epoch, config.epochs):
    while True:
        # train for one epoch

        if epoch <3:
            train_loss = train(train_loader1, model, criterion, optimizer, epoch,writer = writer)
        else:
            train_loss = train(train_loader, model, criterion, optimizer, epoch,writer = writer)

        #train_loss = train(train_loader, model, criterion, optimizer, epoch,writer = writer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] *0.95

        val_loss =val_per_video(model, criterion, epoch, writer=writer)

        
        if config.optim=='SGD':
            adjust_learning_rate(optimizer, epoch )
        #学习率调整
        if train_loss<train_loss_min:
            train_loss_min=train_loss
            train_loss_delay_adjust=0
        else:
            train_loss_delay_adjust+=1

        # if train_loss_delay_adjust>3: 
        #     #loss连续n个epoch不下降就调整学习率
        #     adjust_learning_rate(optimizer, epoch, decay = 0.2 )
        #     train_loss_delay_adjust=0
        #验证loss监控
        if val_loss<val_loss_min:
            val_loss_min=val_loss
            val_loss_delay_adjust=0
        else:
            val_loss_delay_adjust+=1

        # if val_loss_delay_adjust>30: 
        #     print('val_loss did not decay for n epoches, stop training!')
        #     break

        # remember best prec@1 and save checkpoint
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(config.save_dir, 'checkpoint_{}.pth'.format(epoch)))
        
        epoch+=1
    writer.close()


if __name__ == '__main__':
    main()