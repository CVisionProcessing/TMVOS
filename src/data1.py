from __future__ import print_function, division
import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from utils import recursive_glob, decode_segmap
from torch.utils.data import Dataset
from config import Path
import json
import scipy.io
import math
import numbers
import random
from PIL import Image, ImageOps
import glob
import config
import torchvision.transforms as transforms
import cv2
import json


class COCO(data.Dataset):
    def __init__(self, base_dir=config.coco_datadir, img_dir=config.coco_datadir + "train2014",
                 transform=None,
                 ):
        """
        :param base_dir: path to Salient dataset directory
        :param transform: transform to apply
        """
        super().__init__()

        classes = os.listdir(os.path.join(base_dir, "org_masks"))
        self.img_path = []
        self.label_path = []
        self.first_path = []
        self.first_mask = []
        self.spa_prior_path = []
        for c in classes:
            img = os.listdir(os.path.join(base_dir, "org_masks", c))

            for i in img:
                '''
                self.img_path.append(os.path.join(base_dir,"jpgs",c,os.path.splitext(i)[0]+"_0.jpg"))
                self.label_path.append(os.path.join(base_dir,"gts",c,os.path.splitext(i)[0]+'_0.png'))
                self.first_path.append(os.path.join(img_dir,os.path.splitext(i)[0]+'.jpg'))
                self.first_mask.append(os.path.join(base_dir,"org_masks",c,i))
                self.spa_prior_path.append(os.path.join(base_dir,"gts",c,os.path.splitext(i)[0]+'_1.png'))
                '''
                self.first_path.append(os.path.join(base_dir, "jpgs", c, os.path.splitext(i)[0] + "_0.jpg"))
                self.first_mask.append(os.path.join(base_dir, "gts", c, os.path.splitext(i)[0] + '_0.png'))
                self.img_path.append(os.path.join(img_dir, os.path.splitext(i)[0] + '.jpg'))
                self.label_path.append(os.path.join(base_dir, "org_masks", c, i))
                tmp = random.sample([1, 2], 1)
                self.spa_prior_path.append(
                    os.path.join(base_dir, "Deformations", c, os.path.splitext(i)[0] + '_d' + str(tmp[0]) + '.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _target, _img_first, _spa_prior, _target_first = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'first_image': _img_first, 'spa_prior': _spa_prior,
                  'target_first': _target_first, "spa_img": _img}  # coco数据集没有前一帧的图片没办法训练

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        _target = Image.open(self.label_path[index]).convert('1')
        _target_first = Image.open(self.first_mask[index]).convert('1')
        _spa_prior = Image.open(self.spa_prior_path[index]).convert('1')
        # _img_first=extract_image(_img_first,_target_first)
        return _img, _target, _img_first, _spa_prior, _target_first


class DavisDataset_video1(data.Dataset):
    def __init__(self,
                 transform=None,
                 aim=None,
                 train=True
                 ):
        """
        :param base_dir: path to Salient dataset directory
        :param transform: transform to apply
        """
        super().__init__()
        dataset_prefix = config.davis_datadir + 'JPEGImages/480p'
        self.vedio_classes = []
        self.aim = aim
        if aim == 'single':
            with open(config.davis_datadir + "ImageSets/2016/train.txt") as f:
                seqs = f.readlines()
                for seq in seqs:
                    self.vedio_classes.append(seq.strip())
        else:
            with open(config.davis_datadir + "ImageSets/2017/train.txt") as f:
                seqs = f.readlines()
                for seq in seqs:
                    self.vedio_classes.append(seq.strip())
        self.img_path = []
        self.label_path = []
        self.first_path = []
        self.first_mask = []
        self.spa_prior_path = []
        self.spa_img = []
        for vc in self.vedio_classes:
            frame_list = os.listdir(os.path.join(dataset_prefix, vc))
            for i in range(len(frame_list)):
                frame1, frame2 = random.sample(frame_list, 2)
                self.img_path.append(os.path.join(dataset_prefix, vc, frame1))
                self.label_path.append(os.path.join(config.davis_datadir + 'Annotations/480p', vc,
                                                    os.path.splitext(frame1)[0] + '.png'))  # bug
                self.first_path.append(os.path.join(dataset_prefix, vc, frame2))
                self.first_mask.append(
                    os.path.join(config.davis_datadir + 'Annotations/480p', vc, os.path.splitext(frame2)[0] + '.png'))
                while (1):
                    n = random.sample([-3, -2, -1, 1, 2, 3], 1)
                    if os.path.exists(os.path.join(config.davis_datadir + "Annotations/480p", vc,
                                                   str(int(os.path.splitext(frame1)[0]) + n[0]).zfill(5) + '.png')):
                        self.spa_prior_path.append(os.path.join(config.davis_datadir + "Annotations/480p", vc,
                                                                str(int(os.path.splitext(frame1)[0]) + n[0]).zfill(
                                                                    5) + '.png'))
                        self.spa_img.append((os.path.join(dataset_prefix, vc,
                                                          str(int(os.path.splitext(frame1)[0]) + n[0]).zfill(
                                                              5) + '.jpg')))
                        break

        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _target, _img_first, _spa_prior, _target_first, _spa_img = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'first_image': _img_first, 'spa_prior': _spa_prior,
                  'target_first': _target_first, "spa_img": _spa_img}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        _spa_img = Image.open(self.spa_img[index]).convert('RGB')
        _target = Image.open(self.label_path[index])
        _target_first = Image.open(self.first_mask[index])
        _spa_prior = Image.open(self.spa_prior_path[index])
        return _img, _target, _img_first, _spa_prior, _target_first, _spa_img


def extract_image(image, target, mean=[0.485, 0.456, 0.406]):
    # extract the object from _img_first
    image = np.array(image).astype(np.float32).transpose((2, 0, 1))
    target = np.expand_dims(np.array(target).astype(np.float32), -1).transpose((2, 0, 1))

    target = target.repeat(3, axis=0)
    image = image * target

    for i in range(image.shape[0]):
        image[i, :, :][target[i, :, :] == 0] = mean[i] * 255
    image = Image.fromarray(image.transpose((1, 2, 0)).astype('uint8'))

    return image


class DavisDataset_per_video_val(data.Dataset):
    def __init__(self, sequence, obj_id,
                 transform=None,
                 aim=None,
                 train=True,
                 year=2017
                 ):
        super().__init__()
        self.year = year
        dataset_prefix = os.path.join(config.davis_datadir + 'JPEGImages/480p', sequence)
        target_prefix = os.path.join(config.davis_datadir + 'Annotations/480p', sequence)
        self.sequence = sequence
        self.img_path = []
        self.target_path = []
        self.first_path = []
        self.first_mask = []

        for i in range(1, len(os.listdir(dataset_prefix))):
            self.img_path.append(os.path.join(dataset_prefix, str(i).zfill(5) + ".jpg"))
            self.first_path.append(os.path.join(dataset_prefix, "00000.jpg"))
            self.target_path.append(os.path.join(target_prefix, str(i).zfill(5) + ".png"))
            self.first_mask.append(os.path.join(target_prefix, "00000.png"))
        self.transform = transform
        self.color_palette = Image.open(config.davis_datadir + "Annotations/480p/bear/00000.png").getpalette()

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _img_first, _first_mask, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'first_image': _img_first, 'first_mask': _first_mask, "label": _target}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        if self.year == 2017:
            _first_mask = Image.open(self.first_mask[index])
            _target = Image.open(self.target_path[index])
        else:
            _first_mask = Image.open(self.first_mask[index])
            _first_mask = np.uint8(np.array(_first_mask) != 0)
            _first_mask = Image.fromarray(_first_mask * 255).convert("1")

            _target = Image.open(self.target_path[index])
            _target = np.uint8(np.array(_target) != 0)
            _target = Image.fromarray(_target * 255).convert("1")
        return _img, _img_first, _first_mask, _target


class DavisDataset_video_val(data.Dataset):
    def __init__(self,
                 transform=None,
                 aim=None,
                 train=True
                 ):
        """
        :param base_dir: path to Salient dataset directory
        :param transform: transform to apply
        """
        super().__init__()
        dataset_prefix = config.davis_datadir + 'JPEGImages/480p'
        self.vedio_classes = []
        if aim == 'single':
            with open(config.davis_datadir + "ImageSets/2016/val.txt") as f:
                seqs = f.readlines()
                for seq in seqs:
                    self.vedio_classes.append(seq.strip())
        else:
            with open(config.davis_datadir + "ImageSets/2017/val.txt") as f:
                seqs = f.readlines()
                for seq in seqs:
                    self.vedio_classes.append(seq.strip())
        self.img_path = []
        self.label_path = []
        self.first_path = []
        self.first_mask = []
        self.spa_prior_path = []
        for vc in self.vedio_classes:
            vc_c = len(glob.glob(os.path.join(config.davis_datadir + "Annotations_binary/480p", vc, "00000_*.png")))
            for c in range(1, vc_c + 1):
                for frame in os.listdir(os.path.join(dataset_prefix, vc)):
                    if frame != "00000.jpg":
                        self.img_path.append(os.path.join(dataset_prefix, vc, frame))  # bug
                        self.label_path.append(os.path.join(config.davis_datadir + "Annotations_binary/480p", vc,
                                                            os.path.splitext(frame)[0] + '_' + str(c) + ".png"))
                        self.first_path.append(os.path.join(dataset_prefix, vc, "00000.jpg"))
                        self.first_mask.append(os.path.join(config.davis_datadir + "Annotations_binary/480p", vc,
                                                            "00000_" + str(c) + ".png"))
                        self.spa_prior_path.append(os.path.join(config.davis_datadir + 'Annotations_binary/480p',
                                                                vc, str(int(os.path.splitext(frame)[0]) - 1).zfill(
                                5) + '_' + str(c) + '.png'))
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _target, _img_first, _spa_prior, _target_first = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'first_image': _img_first, 'spa_prior': _spa_prior,
                  'target_first': _target_first}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        _target = Image.open(self.label_path[index]).convert('1')
        _spa_prior = Image.open(self.spa_prior_path[index]).convert('1')
        _target_first = Image.open(self.first_mask[index]).convert('1')
        # _img_first=extract_image(_img_first,_target_first)
        return _img, _target, _img_first, _spa_prior, _target_first


# datasetloaders
class DavisDataset_video_test(data.Dataset):
    def __init__(self, sequence,
                 transform=None,
                 aim=None,
                 train=True,
                 year=2017
                 ):
        super().__init__()
        self.year = year

        dataset_prefix = os.path.join(config.davis_datadir + 'JPEGImages/480p', sequence)
        target_prefix = os.path.join(config.davis_datadir + 'Annotations/480p', sequence)
        self.sequence = sequence
        self.img_path = []
        self.first_path = []
        self.first_mask = []
        #self.target_path = []
        for i in range(0, len(os.listdir(dataset_prefix))):
            self.img_path.append(os.path.join(dataset_prefix, str(i).zfill(5) + ".jpg"))  # bug
            #self.target_path.append(os.path.join(target_prefix, str(i).zfill(5) + ".png"))
            self.first_path.append(os.path.join(dataset_prefix, "00000.jpg"))
            self.first_mask.append(os.path.join(target_prefix, "00000.png"))
        self.transform = transform
        if year == 2017:
            self.color_palette = Image.open(self.first_mask[0]).getpalette()
            self.cls_num = np.array(Image.open(self.first_mask[0])).max()
        else:
            self.cls_num = 1
            self.color_palette = Image.open(config.davis_datadir + "Annotations/480p/bear/00000.png").getpalette()

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _img_first, _first_mask = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'first_image': _img_first, 'first_mask': _first_mask}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        if self.year == 2017:
            _first_mask = Image.open(self.first_mask[index])
            # _target = Image.open(self.target_path[index])
        else:
            _first_mask = Image.open(self.first_mask[index])
            _first_mask = np.uint8(np.array(_first_mask) != 0)
            _first_mask = Image.fromarray(_first_mask * 255).convert("1")

            # _target = Image.open(self.target_path[index])
            # _target = np.uint8(np.array(_target) != 0)
            # _target = Image.fromarray(_target * 255).convert("1")

        return _img, _img_first, _first_mask


class Youtube_VOS(data.Dataset):
    def __init__(self,
                 transform=None,
                 train=True
                 ):
        """
        :param base_dir: path to Salient dataset directory
        :param transform: transform to apply
        """
        super().__init__()
        dataset_prefix = config.youtube_datadir + '/train_zip/train'
        sequences = os.listdir(os.path.join(dataset_prefix, "JPEGImages"))

        self.img_path = []
        self.label_path = []
        self.first_path = []
        self.first_mask = []
        self.spa_prior_path = []
        self.spa_img_path = []

        for seq in sequences:
            vedio_classes = os.listdir(os.path.join(dataset_prefix, "Annotations_split", seq))
            frame_list = os.listdir(os.path.join(dataset_prefix, "JPEGImages", seq))
            for vc in vedio_classes:
                for i in range(len(frame_list)):
                    frame1, frame2 = random.sample(frame_list, 2)
                    self.img_path.append(os.path.join(dataset_prefix, "JPEGImages", seq, frame1))
                    self.label_path.append(os.path.join(dataset_prefix, "Annotations_split", seq, vc,
                                                        os.path.splitext(frame1)[0] + '.png'))
                    self.first_path.append(os.path.join(dataset_prefix, "JPEGImages", seq, frame2))
                    self.first_mask.append(os.path.join(dataset_prefix, "Annotations_split", seq, vc,
                                                        os.path.splitext(frame2)[0] + '.png'))
                    '''
                    if np.where(frame_list==frame1)==0:
                        self.spa_prior_path.append(os.path.join(dataset_prefix,"Annotations_split",seq,vc,os.path.splitext(frame_list[np.where(frame_list==frame1)+1])[0]+".png"))
                    else:
                        self.spa_prior_path.append(os.path.join(dataset_prefix,"Annotations_split",seq,vc,os.path.splitext(frame_list[np.where(frame_list==frame1)-1])[0]+".png"))
                    '''
                    if os.path.exists(os.path.join(dataset_prefix, "JPEGImages", seq,
                                                   str(int(os.path.splitext(frame1)[0]) - 5).zfill(5) + ".jpg")):
                        self.spa_prior_path.append(os.path.join(dataset_prefix, "Annotations_split", seq, vc,
                                                                str(int(os.path.splitext(frame1)[0]) - 5).zfill(
                                                                    5) + ".png"))
                    elif os.path.exists(os.path.join(dataset_prefix, "JPEGImages", seq,
                                                     str(int(os.path.splitext(frame1)[0]) + 5).zfill(5) + ".jpg")):
                        self.spa_prior_path.append(os.path.join(dataset_prefix, "Annotations_split", seq, vc,
                                                                str(int(os.path.splitext(frame1)[0]) + 5).zfill(
                                                                    5) + ".png"))
                    else:
                        frame3 = random.sample(frame_list, 1)[0]
                        self.spa_prior_path.append(os.path.join(dataset_prefix, "Annotations_split", seq, vc,
                                                                str(os.path.splitext(frame3)[0]).zfill(5) + ".png"))
                        self.spa_img_path.append(os.path.join(dataset_prefix, "JPEGImages", seq,
                                                              str(int(os.path.splitext(frame3)[0])).zfill(5) + ".jpg"))
                    '''
                    while(1):
                        n = random.sample([-1,1],1)
                        if os.path.exists(os.path.join(dataset_prefix,"JPEGImages",seq,str(int(os.path.splitext(frame1)[0])+int(n[0])*5).zfill(5)+".jpg")):
                            self.spa_prior_path.append(os.path.join(dataset_prefix,"Annotations_split",seq,vc,str(int(os.path.splitext(frame1)[0])+int(n[0])*5).zfill(5)+".png"))
                            break
                    '''
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        _img, _target, _img_first, _spa_prior, _target_first, _spa_img = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'first_image': _img_first, 'spa_prior': _spa_prior,
                  'target_first': _target_first, 'spa_img': _spa_img}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.img_path[index]).convert('RGB')
        _img_first = Image.open(self.first_path[index]).convert('RGB')
        _spa_img = Image.open(self.spa_img_path[index]).convert('RGB')
        _target = Image.open(self.label_path[index]).convert('1')
        _target_first = Image.open(self.first_mask[index]).convert('1')
        _spa_prior = Image.open(self.spa_prior_path[index]).convert('1')
        # _img_first=extract_image(_img_first,_target_first)
        return _img, _target, _img_first, _spa_prior, _target_first, _spa_img


# augmentation
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        th, tw = self.size
        x1 = -1
        y1 = -1
        for elem in sample.keys():
            tmp = sample[elem]
            if self.padding > 0:
                tmp = ImageOps.expand(tmp, border=self.padding, fill=0)
            w, h = tmp.size
            if w == tw and h == th:
                pass
            elif w < tw or h < th:
                if np.array(tmp).ndim == 2:
                    tmp = tmp.resize((tw, th), Image.NEAREST)
                else:
                    tmp = tmp.resize((tw, th), Image.BILINEAR)
            else:
                if x1 < 0 and y1 < 0:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                tmp = tmp.crop((x1, y1, x1 + tw, y1 + th))
            sample[elem] = tmp
        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        th, tw = self.size
        for elem in sample.keys():
            tmp = sample[elem]
            w, h = tmp.size
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            tmp = tmp.crop((x1, y1, x1 + tw, y1 + th))
            sample[elem] = tmp
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        t = random.random()
        for elem in sample.keys():
            tmp = sample[elem]
            if t < 0.5:
                tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
            sample[elem] = tmp
        return sample


class Erode(object):
    def __init__(self, kernel_size=(3, 3)):
        self.kernel = np.ones(kernel_size, np.uint8)

    def __call__(self, sample):
        for elem in sample.keys():
            if elem == "spa_prior":
                sample[elem] = cv2.erode(sample[elem], self.kernel)
        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = np.array(sample[elem]).astype(np.float32)
            if tmp.ndim >= 3:
                tmp /= 255.0
                tmp -= self.mean
                tmp /= self.std
            sample[elem] = tmp
        return sample


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for elem in sample.keys():
            tmp = sample[elem]
            if np.array(tmp).ndim == 2:
                tmp = np.expand_dims(np.array(tmp).astype(np.float32), -1).transpose((2, 0, 1))
                tmp[tmp == 255] = 0
            else:
                tmp = np.array(tmp).astype(np.float32).transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp).float()
            sample[elem] = tmp
        return sample


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        for elem in sample.keys():
            tmp = sample[elem]
            # assert img.size == mask.size
            if np.array(tmp).ndim == 2:
                tmp = tmp.resize(self.size, Image.NEAREST)
            else:
                tmp = tmp.resize(self.size, Image.BILINEAR)
            sample[elem] = tmp
        return sample


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        oh, ow = self.size
        for elem in sample.keys():
            tmp = sample[elem]
            # assert img.size == mask.size
            w, h = tmp.size
            if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
                sample[elem] = tmp
            else:
                if np.array(tmp).ndim == 2:
                    tmp = tmp.resize((ow, oh), Image.NEAREST)
                else:
                    tmp = tmp.resize((ow, oh), Image.BILINEAR)
                sample[elem] = tmp
        return sample


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        for elem in sample.keys():
            tmp = sample[elem]
            if np.array(tmp).ndim == 2:
                tmp = tmp.rotate(rotate_degree, Image.NEAREST)
            else:
                tmp = tmp.rotate(rotate_degree, Image.BILINEAR)
            sample[elem] = tmp
        return sample
        '''
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}
        '''


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        w = -1
        h = -1
        for elem in sample.keys():
            tmp = sample[elem]
            if w < 0 and h < 0:
                w = int(random.uniform(0.8, 2.5) * tmp.size[0])
                h = int(random.uniform(0.8, 2.5) * tmp.size[1])
            # assert img.size == mask.size
            if np.array(tmp).ndim == 2:
                tmp = tmp.resize((w, h), Image.NEAREST)
            else:
                tmp = tmp.resize((w, h), Image.BILINEAR)
            sample[elem] = tmp
        # sample = {'image': img, 'label': mask}
        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return {'image': img, 'label': mask}


if __name__ == '__main__':
    # from dataloaders import custom_transforms as tr
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader

    # from torchvision import transforms
    # import matplotlib.pyplot as plt
    '''
    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScale((0.5, 0.75)),
        tr.RandomCrop((512, 1024)),
        tr.RandomRotate(5),
        tr.ToTensor()])
    '''
    cityscapes_train = DavisDataset_video(train=False)

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)
    print(len(dataloader))
    '''
    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            tmp = np.squeeze(tmp, axis=0)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)
    '''
