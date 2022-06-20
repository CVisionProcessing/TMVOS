import os
import config
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import scipy.stats as scipystats


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='0'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, 'pascal')
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        n_classes = 2
        label_colours = np.array([
            [0, 0, 0],
            [255, 255, 255]])
        # sraise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def BCE(logit,target, ignore_index=255,weight=None,size_average=True,batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    logit = logit.squeeze(1)
    criterion = nn.BCELoss(weight=None, size_average=True, reduction='mean')
    loss = criterion(logit.float(),target.float())
    return loss


def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(),
                                        ignore_index=ignore_index, reduction='sum')
    loss = criterion(logit.float(), target.long())
    if size_average:
        loss /= (h * w)
    if batch_average:
        loss /= n

    return loss

def cross_entropy2d_weight(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    temp=target.data.cpu().numpy()
    freqCount = scipystats.itemfreq(temp)
    total = freqCount[0][1]+freqCount[1][1]
    perc_1 = freqCount[1][1]/total
    perc_0 = freqCount[0][1]/total
    weight_array= [perc_1,perc_0]
    weight_tensor=torch.FloatTensor(weight_array).cuda()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor,
                                        ignore_index=ignore_index, reduction='sum')
    loss = criterion(logit.float(), target.long())
    if size_average:
        loss /= (h * w)
    if batch_average:
        loss /= n

    return loss

class vLoss(nn.Module):
    def __init__(self, w=0.5):
        super(vLoss, self).__init__()
        self.w = w
        # self.mean_batch = []
        return

    def forward(self, output, target, target_var_for_cross, batch_average=True):  # mse：最小平方误差函数
        icloss = 0
        ocloss = 0
        n, c, w, h = output.size()
        mean_batch = []
        for ni in range(n):
            mean = []
            mea_tmp = torch.Tensor()
            for i in range(len(target[ni])):  # 0为背景
                gt = target[ni][i].unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
                a = output[ni, :, :, :] * gt
                mea = torch.mean(a, dim=2, keepdim=True)
                mea = torch.mean(mea, dim=3, keepdim=True)
                if i == 0:
                    mea_tmp = mea
                else:
                    mea_tmp = torch.cat([mea, mea_tmp], 0)
                # mean.append(mea.squeeze())
                # mea = mea.expand(1, c, w, h) * gt
                # icloss += F.mse_loss(a, mea) / (gt.sum())
            mean_batch.append(mea_tmp)
            # for i in range(len(target[ni])):
            #     for j in range(i + 1, len(target[ni])):
            #         ocloss += 1 / F.mse_loss(mean[i], mean[j])
        self.mean_batch = mean_batch

        predctions = get_pred(output, mean_batch)
        cross_loss = cross_entropy2d(predctions.float(), target_var_for_cross)
        return torch.autograd.Variable(cross_loss, requires_grad=True)

        # if batch_average:
        #     return (icloss + self.w * ocloss) / n
        # return icloss + self.w * ocloss


def slice_target(targetvar):  # 分割出不同实例mask,0为背景
    n, c, w, h = targetvar.size()
    assert c == 1
    target_arr = []

    for i in range(n):
        target_tmp = []
        for j in range(int(targetvar[i].max()) + 1):
            target_tmp.append(targetvar[i][0] == j)
        target_arr.append(target_tmp)
    return target_arr


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_mean_batch(output, target):  # 获取聚类中心
    mean_batch = []
    for ni in range(target.size()[0]):
        mean = []
        mea_tmp = torch.Tensor()
        for i in range(int(target[ni, :, :, :].max()) + 1):  # 0为背景
            gt = (target[ni, 0, :, :] == i).float().unsqueeze(dim=0).unsqueeze(dim=0)
            a = output[ni, :, :, :] * gt
            mea = torch.mean(a, dim=2, keepdim=True)
            mea = torch.mean(mea, dim=3, keepdim=True)
            if i == 0:
                mea_tmp = mea
            else:
                mea_tmp = torch.cat([mea, mea_tmp], 0)
            # mean.append(mea.squeeze())
            # mea = mea.expand(1, c, w, h) * gt
            # icloss += F.mse_loss(a, mea) / (gt.sum())
        mean_batch.append(mea_tmp)
    return mean_batch


def get_pred(output, sur):  # sur为多通道二值gt图
    pred = torch.Tensor()
    for i in range(len(sur)):  # batch
        cha = sur[i] - output[i, :, :, :]
        out = cha.mul(cha)
        out = torch.mean(out, dim=1, keepdim=False)
        out = torch.sqrt(out)
        min = out.min(dim=0)[0].unsqueeze(dim=0)
        out = (out == min)
        for j in range(out.size()[0]):
            if j == 0:
                out[j, :, :] = (0)
            else:
                out[j, :, :] = out[j, :, :] * j
        out = out.max(dim=0)[0].unsqueeze(dim=0).unsqueeze(dim=0)
        # img = out[0].cpu().numpy()
        # Image.fromarray(img).show()
        if i == 0:
            pred = out
        else:
            pred = torch.cat((pred, out), dim=0)

    return pred


def get_iou(pred, gt):
    total_iou = 0.0
    #max_i = pred.max(dim=1)[0].unsqueeze(dim=1)
    #max_i=torch.max(pred).item()
    #pred = (pred==max_i)
    for i in range(0, pred.size()[0]):
        n_classes = int(gt[i, :, :, :].max()) 
        pred_tmp = pred[i, :, :, :]
        pred_tmp=torch.max(pred_tmp,0)[1]
        gt_tmp = gt[i, :, :, :]
        #print(pred_tmp.size(),gt_tmp.size())

        intersect = [0] * (n_classes)
        union = [0] * (n_classes)
        for j in range(0,n_classes):
            match = (pred_tmp == j+1) + (gt_tmp == j+1)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(0,n_classes):
            if union[k] == 0:
                iou.append(1)
            else:
                iou.append(intersect[k] / union[k])

        if len(iou)==0:
            if pred_tmp.max()==0:
                img_iou=1
            else:
                img_iou=0
        else:
            img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou, len(iou) if len(iou) else 1


def get_mae(pred, gt, n_classes=2):
    # just for 二分类
    total_mae = []
    for i in range(len(pred)):
        pred_tmp = pred[i].detach().cpu().numpy()
        gt_tmp = gt[i].detach().cpu().numpy()
        _mae = 0.
        for j in range(n_classes - 1):
            match = (pred_tmp == j) + (gt_tmp == j)
            diff = np.abs(pred_tmp[1] - gt_tmp)
            _mae += np.mean(diff)
        total_mae.append(_mae)

    return np.mean(_mae)


def parameter():
    p = {}
    p['gtThreshold'] = 0.5
    p['beta'] = np.sqrt(0.3)
    p['thNum'] = 100
    p['thList'] = np.linspace(0, 1, p['thNum'])

    return p


eps = 2.2204e-16


def get_prcurve(_gtMask, _curSMap, p=parameter()):
    for i in range(len(_gtMask)):
        gtMask = _gtMask[i][0]
        curSMap = _curSMap[i][1]
        curSMap = curSMap.detach().cpu().numpy()
        gtMask = (gtMask >= p['gtThreshold']).detach().cpu().numpy()
        gtInd = np.where(gtMask > 0)
        gtCnt = np.sum(gtMask)

        if gtCnt == 0:
            prec = []
            recall = []
        else:
            hitCnt = np.zeros((p['thNum'], 1), np.float32)
            algCnt = np.zeros((p['thNum'], 1), np.float32)

            for k, curTh in enumerate(p['thList']):
                thSMap = (curSMap >= curTh)

                hitCnt[k] = np.sum(thSMap[gtInd])
                algCnt[k] = np.sum(thSMap)

            prec = hitCnt / (algCnt + eps)
            recall = hitCnt / gtCnt

    return prec, recall


def PR_Curve(prec, recall):
    p = parameter()
    beta = p['beta']

    prec = np.hstack(prec[:])
    recall = np.hstack(recall[:])
    prec = np.mean(prec, 1)
    recall = np.mean(recall, 1)

    # compute the max F-Score
    score = (1 + beta ** 2) * prec * recall / (beta ** 2 * prec + recall + eps)
    curTh = np.argmax(score)
    curScore = np.max(score)

    res = {}
    res['prec'] = prec
    res['recall'] = recall
    res['curScore'] = curScore
    res['curTh'] = curTh

    return res
