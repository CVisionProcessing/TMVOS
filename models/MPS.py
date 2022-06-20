import torch.nn as nn
import torch
#from lstm import  ConvLSTMCell,ConvLSTMCellMask
try:
    from models import resnet
except ImportError as e:
    import resnet


import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, inplanes ,planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,stride=stride, padding=1),nn.BatchNorm2d(planes),
                                    nn.ReLU(),nn.Conv2d(planes,planes,kernel_size=3,padding=1),nn.BatchNorm2d(planes))
        if stride!=1 or inplanes!=planes:
            self.downsample =nn.Sequential(nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride, bias=False),nn.BatchNorm2d(planes))
        else:
            self.downsample = None
        self.relu=nn.ReLU()
    def forward(self,x,y=None):
        if y is not None:
            x = torch.cat((x,y),dim=1)
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


class MPS(nn.Module):
    """docstring for MPS"""
    def __init__(self, inputchannal=3,pretrained=True,num_classes=1000):
        super(MPS, self).__init__()
        self.sie_resnet = resnet.resnet34(pretrained=pretrained,num_classes=1000)
        expansion = 1

        self.side_output1 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.side_output2 = nn.Sequential(nn.Conv2d(64*expansion,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.side_output3 = nn.Sequential(nn.Conv2d(128*expansion,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.side_output4 = nn.Sequential(nn.Conv2d(256*expansion,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.side_output5 = nn.Sequential(nn.Conv2d(512*expansion,64,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU())

        self.Sigmoid = nn.Sigmoid()

        # 并没有用空洞卷积，由于现有模型可以带入使用，暂未修改
        self.dilation2 = nn.Sequential(nn.Conv2d(64*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128)) #dilation(64*expansion+2,32)
        self.dilation3 = nn.Sequential(nn.Conv2d(128*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))
        self.dilation4 = nn.Sequential(nn.Conv2d(256*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))
        self.dilation5 = nn.Sequential(nn.Conv2d(512*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))


        self.dilation2_b = nn.Sequential(nn.Conv2d(64*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128)) #dilation(64*expansion+2,32)
        self.dilation3_b = nn.Sequential(nn.Conv2d(128*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))
        self.dilation4_b = nn.Sequential(nn.Conv2d(256*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))
        self.dilation5_b = nn.Sequential(nn.Conv2d(512*expansion+2,128,kernel_size=1,stride=1,padding=0),
                                                ResBlock(128,128))


        self.Resblock1 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))
        self.Resblock2 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))
        self.Resblock3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))

        self.last_conv = nn.Sequential(nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
                                        nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1))


        self.Resblock1_b = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))
        self.Resblock2_b = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))
        self.Resblock3_b = nn.Sequential(nn.Conv2d(256,128,kernel_size=1,stride=1,padding=0),
                                        ResBlock(128,128))

        self.last_conv_b = nn.Sequential(nn.Conv2d(128,32,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
                                        nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1))


        self.softmax = nn.Softmax(dim=1)

    def forward(self,input, input_1,first_mask, spa_prior,num_object,spa_img):
        # feature extractor
        x1,x2,x3,x4,x5 =self.sie_resnet(input)
        infer_x1,infer_x2,infer_x3,infer_x4,infer_x5 = self.sie_resnet(input_1)

        infer_x5 = F.interpolate(self.side_output5(infer_x5),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        infer_x4 = F.interpolate(self.side_output4(infer_x4),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        infer_x3 = self.side_output3(infer_x3)
        infer_x2 = F.interpolate(self.side_output2(infer_x2),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        infer_x1 = F.interpolate(self.side_output1(infer_x1),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        infer_x1 = infer_x1+infer_x2+infer_x3+infer_x4+infer_x5

        target_x5 = F.interpolate(self.side_output5(x5),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        target_x4 = F.interpolate(self.side_output4(x4),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        target_x3 = self.side_output3(x3)
        target_x2 = F.interpolate(self.side_output2(x2),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        target_x1 = F.interpolate(self.side_output1(x1),size = x3.size()[2:],mode="bilinear",align_corners=True) 
        target_x1  = target_x1+target_x2+target_x3+target_x4+target_x5

        #triplet matching 
        first_mask=F.interpolate(first_mask,size=infer_x3.size()[2:],mode="nearest",align_corners=None)
        
        embedding = self.softmax(self.embedding2(infer_x1,target_x1,first_mask,num_object))

        #background 
        conv2_dilation_b=torch.cat((x2,F.interpolate(embedding[:,0].unsqueeze(1),size=x2.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv3_dilation_b=torch.cat((x3,F.interpolate(embedding[:,0].unsqueeze(1),size=x3.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv4_dilation_b=torch.cat((x4,F.interpolate(embedding[:,0].unsqueeze(1),size=x4.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv5_dilation_b=torch.cat((x5,F.interpolate(embedding[:,0].unsqueeze(1),size=x5.size()[2:],mode="bilinear",align_corners=True)),dim=1)
   
        spa_prior_i =(spa_prior==0).float()
        conv2_dilation_b=torch.cat((conv2_dilation_b,F.interpolate(spa_prior_i,size=x2.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv3_dilation_b=torch.cat((conv3_dilation_b,F.interpolate(spa_prior_i,size=x3.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv4_dilation_b=torch.cat((conv4_dilation_b,F.interpolate(spa_prior_i,size=x4.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        conv5_dilation_b=torch.cat((conv5_dilation_b,F.interpolate(spa_prior_i,size=x5.size()[2:],mode="bilinear",align_corners=True)),dim=1)
        
        
        conv2_dilation_b = self.dilation2_b(conv2_dilation_b)
        conv3_dilation_b = self.dilation3_b(conv3_dilation_b)
        conv4_dilation_b = self.dilation4_b(conv4_dilation_b)
        conv5_dilation_b = self.dilation5_b(conv5_dilation_b)

        x_4_b = F.interpolate(conv5_dilation_b,size=conv4_dilation_b.size()[2:],mode="bilinear", align_corners = True)

        x_3_b = self.Resblock3_b(torch.cat((x_4_b,(conv4_dilation_b)),dim=1))
        x_3_b = F.interpolate(x_3_b,size=conv3_dilation_b.size()[2:],mode="bilinear", align_corners = True)

        x_2_b = self.Resblock2_b(torch.cat((x_3_b,(conv3_dilation_b )),dim=1))
        x_2_b = F.interpolate(x_2_b,size=conv2_dilation_b.size()[2:],mode="bilinear", align_corners = True)

        x_1_b = self.Resblock1_b(torch.cat((x_2_b,(conv2_dilation_b)),dim=1))

        prev1_b = F.interpolate(x_1_b, size=input.size()[2:], mode='bilinear', align_corners=True)
        prev1_b = self.last_conv_b(prev1_b)
            
        output = prev1_b

        #foreground
        for i in range(1,num_object+1):
            
            conv2_dilation=torch.cat((x2,F.interpolate(embedding[:,i].unsqueeze(1),size=x2.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv3_dilation=torch.cat((x3,F.interpolate(embedding[:,i].unsqueeze(1),size=x3.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv4_dilation=torch.cat((x4,F.interpolate(embedding[:,i].unsqueeze(1),size=x4.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv5_dilation=torch.cat((x5,F.interpolate(embedding[:,i].unsqueeze(1),size=x5.size()[2:],mode="bilinear",align_corners=True)),dim=1)

            spa_prior_i = (spa_prior==i).float()
            conv2_dilation=torch.cat((conv2_dilation,F.interpolate(spa_prior_i,size=x2.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv3_dilation=torch.cat((conv3_dilation,F.interpolate(spa_prior_i,size=x3.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv4_dilation=torch.cat((conv4_dilation,F.interpolate(spa_prior_i,size=x4.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            conv5_dilation=torch.cat((conv5_dilation,F.interpolate(spa_prior_i,size=x5.size()[2:],mode="bilinear",align_corners=True)),dim=1)
            
            conv2_dilation = self.dilation2(conv2_dilation)
            conv3_dilation = self.dilation3(conv3_dilation)
            conv4_dilation = self.dilation4(conv4_dilation)
            conv5_dilation = self.dilation5(conv5_dilation)

            x_4 = F.interpolate(conv5_dilation,size=conv4_dilation.size()[2:],mode="bilinear", align_corners = True)

            x_3 = self.Resblock3(torch.cat((x_4,(conv4_dilation)),dim=1))
            x_3 = F.interpolate(x_3,size=conv3_dilation.size()[2:],mode="bilinear", align_corners = True)

            x_2 = self.Resblock2(torch.cat((x_3,(conv3_dilation )),dim=1))
            x_2 = F.interpolate(x_2,size=conv2_dilation.size()[2:],mode="bilinear", align_corners = True)

            x_1 = self.Resblock1(torch.cat((x_2,(conv2_dilation)),dim=1))

            prev1 = F.interpolate(x_1, size=input.size()[2:], mode='bilinear', align_corners=True)
            prev1 = self.last_conv(prev1)
                
            output = torch.cat((output,prev1),dim=1)
        
        return output,infer_x1
    
    #global matching（仅前景）
    def embedding1(self,infer,target,first_mask,num_object):
        b,c,w,h = infer.size()
        infer = infer.permute(0,2,3,1).view(b,-1,c)
        target = target.view(b,c,-1)

        target_tmp = (target**2).sum(dim=1,keepdim=True)
        target_tmp = target_tmp.repeat(1,w*h,1)

        infer_tmp  = (infer**2).sum(dim=2,keepdim=True)
        infer_tmp = infer_tmp.repeat(1,1,w*h)

        corr = torch.sqrt(target_tmp+infer_tmp-2*torch.matmul(infer,target))
        first_mask = first_mask.view(b,-1)
        sigmoid = nn.Sigmoid()
        for i in range(0,num_object+1):
            if (first_mask==i).sum():
                pos = torch.min(corr[first_mask==i],dim=0)[0].view(b,-1,w,h)
            else:
                pos = torch.max(corr[first_mask!=i],dim=0)[0].view(b,-1,w,h)
            if i ==0:
                tmp = pos
            else:
                tmp = torch.cat((tmp,pos),dim=1)

        return tmp 

    #triplet matching
    def embedding2(self,infer,target,first_mask,num_object):
        b,c,w,h = infer.size()
        infer = infer.permute(0,2,3,1).view(b,-1,c)
        target = target.view(b,c,-1)

        target_tmp = (target**2).sum(dim=1,keepdim=True)
        target_tmp = target_tmp.repeat(1,w*h,1)

        infer_tmp  = (infer**2).sum(dim=2,keepdim=True)
        infer_tmp = infer_tmp.repeat(1,1,w*h)

        corr = torch.sqrt(target_tmp+infer_tmp-2*torch.matmul(infer,target))
        first_mask = first_mask.view(b,-1)
        sigmoid = nn.Sigmoid()
        for i in range(0,num_object+1):
            if (first_mask==i).sum() and (first_mask!=i).sum():
                pos = torch.min(corr[first_mask==i],dim=0)[0].view(b,-1,w,h)
                neg = torch.min(corr[first_mask!=i],dim=0)[0].view(b,-1,w,h)
            elif not (first_mask==i).sum():
                #pos = torch.autograd.Variable(torch.zeros(b,1,w,h)).cuda()
                pos = torch.max(corr[first_mask!=i],dim=0)[0].view(b,-1,w,h)
                neg = torch.min(corr[first_mask!=i],dim=0)[0].view(b,-1,w,h)
            else:
                pos = torch.min(corr[first_mask==i],dim=0)[0].view(b,-1,w,h)
                neg = torch.max(corr[first_mask==i],dim=0)[0].view(b,-1,w,h)
                #neg = torch.autograd.Variable(torch.zeros(b,1,w,h)).cuda()

            if i ==0:
                tmp = (neg-pos)
            else:
                tmp = torch.cat((tmp,(neg-pos)),dim=1)

        return tmp 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__=="__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    model=MPS().cuda()
    from PIL import Image
    import numpy as np
    a=torch.autograd.Variable(torch.ones(1,3,480,910)).cuda()
    b=torch.autograd.Variable(torch.ones(1,3,480,910)).cuda()
    c=torch.autograd.Variable(torch.from_numpy(np.array(Image.open("/home/ljj/data/davis-2017/data/DAVIS/Annotations/480p/bike-packing/00000.png"))).unsqueeze(0).unsqueeze(0)).float().cuda()
    d=torch.autograd.Variable(torch.from_numpy(np.array(Image.open("/home/ljj/data/davis-2017/data/DAVIS/Annotations/480p/bike-packing/00001.png"))).unsqueeze(0).unsqueeze(0)).float().cuda()
    
    print(c.shape,c.max())
    import time
    start = time.time()
    with torch.no_grad():
        e= model(a,b,c,d,2,a)
    #print(e.size())
    print(time.time()-start)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        #print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        #print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))
    count_ops = 0

    def measure_layer(layer, x, multi_add=1):
        type_name = str(layer)[:str(layer).find('(')].strip()
        #print(type_name)
        if type_name in ['Conv2d']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                        layer.stride[1] + 1)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                    layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

        ### ops_nonlinearity
        elif type_name in ['ReLU']:
            delta_ops = x.numel()

        ### ops_pooling
        elif type_name in ['AvgPool2d']:
            in_w = x.size()[2]
            kernel_ops = layer.kernel_size * layer.kernel_size
            out_w = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
            out_h = int((in_w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1)
            delta_ops = x.size()[1] * out_w * out_h * kernel_ops

        elif type_name in ['AdaptiveAvgPool2d']:
            delta_ops = x.numel()

        ### ops_linear
        elif type_name in ['Linear']:
            weight_ops = layer.weight.numel() * multi_add
            bias_ops = layer.bias.numel()
            delta_ops = weight_ops + bias_ops

        elif type_name in ['BatchNorm2d']:
            normalize_ops = x.numel()
            scale_shift = normalize_ops
            delta_ops = normalize_ops + scale_shift

        ### ops_nothing
        elif type_name in ['Dropout2d', 'DropChannel', 'Dropout','MaxPool2d','Sigmoid']:
            delta_ops = 0

        ### unknown layer type
        else:
            raise TypeError('unknown layer type: %s' % type_name)

        global count_ops
        count_ops += delta_ops
        return

    def is_leaf(module):
        return sum(1 for x in module.children()) == 0

    # 判断是否为需要计算flops的结点模块
    def should_measure(module):
        # 代码中的残差结构可能定义了空内容的Sequential
        if str(module).startswith('Sequential'):
            return False
        if is_leaf(module):
            return True
        return False

    def measure_model(model, shape=(1,3,480,854)):
        global count_ops
        a=torch.autograd.Variable(torch.ones(1,3,480,854)).cuda()
        b=torch.autograd.Variable(torch.ones(1,3,480,854)).cuda()
        c=torch.autograd.Variable(torch.ones(1,1,480,854)).cuda()
        d=torch.autograd.Variable(torch.ones(1,1,480,854)).cuda()

        # 将计算flops的操作集成到forward函数
        def new_forward(m):
            def lambda_forward(x):
                measure_layer(m, x)
                return m.old_forward(x)
            return lambda_forward

        def modify_forward(model):
            for child in model.children():
                if should_measure(child):
                    # 新增一个old_forward属性保存默认的forward函数
                    # 便于计算flops结束后forward函数的恢复
                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # 对修改后的forward函数进行恢复
                if is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

        modify_forward(model)
        # forward过程中对全局的变量count_ops进行更新
        model.forward(a,b,c,d)
        restore_forward(model)

        return count_ops
    print(measure_model(model))
    #print(model)