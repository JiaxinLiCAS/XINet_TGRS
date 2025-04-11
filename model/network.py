# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：使用到的具体网络模块以及损失函数
"""

import numpy as np
import torch
import torch.nn.functional as fun
import torch.nn as nn
#import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

'''       Loss           '''
class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, dim=1) #计算每个像素位置光谱向量的和 #1 x h x w 的1矩阵
        target_tensor = self.get_target_tensor(input) #1 x h x w 的1矩阵
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss
'''       Loss           '''
    
'''       模型初始化公用代码           '''
def init_weights(net, init_type, gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            #print('classname',classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(height*weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1/(channel))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                #print('classname',classname)
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type, init_gain, device):
    
    net.to(device)  
    init_weights(net, init_type, gain=init_gain)
    return net
'''    模型初始化公用代码           '''
##########################################################################################################


########################   HrMSI:Initial feature extraction module     #############################

#输出波段为endmember_num/4
def def_hr_msi_initial_feature(input_channel,output_channel, device, block_num, init_type='kaiming', init_gain=0.02): 
    #block_num:spatial_res_blockd的个数
    #input_channel：hr_msi的波段数
    #output_channel：输出的波段数 即端元数/4
  
    
    net = hr_msi_initial_feature(input_channel, output_channel,block_num)

    return init_net(net, init_type, init_gain,device)    
 
class hr_msi_initial_feature(nn.Module):
    def __init__(self,input_channel,output_channel,block_num): 
      
        #input_channel：hr_msi的波段数
        #output_channel：输出的波段数 即端元数/4
        
        
        super().__init__()
       
        self.begin=nn.Conv2d(in_channels=input_channel,out_channels=60,kernel_size=3,stride=1,padding=1)
        
        layer = []
        
        for i in range(block_num):
            layer.append(
                                spatial_res_block(60)
                              ) 
        self.middle=nn.Sequential(*layer)
        
        
        #self.middel=spatial_res_block(60) #不改变波段数
        
        ###最后输出的波段数量是int(endmember_num/4)
        self.end=nn.Conv2d(in_channels=60,out_channels=output_channel,kernel_size=3,stride=1,padding=1) 
                                                                  #padding=(x,y) x是在高度上增高,y是在宽度上变宽
    
    def forward(self,input):
        output1=self.begin(input)
        #print("output1~",output1.shape) torch.Size([1, 60, 240, 240])
        output2=self.middle(output1)
        #print("output2~",output2.shape) torch.Size([1, 60, 240, 240])
        output3=self.end(output2)
        #print("output3~",output3.shape)   torch.Size([1, 100, 240, 240])
        
        
        return output3 #1 int(endnum/4)  H  W


class spatial_res_block(nn.Module): #padding=(x,y) x是在高度上增高,y是在宽度上变宽
    def __init__(self,input_channel): 
        super().__init__()
        assert(input_channel % 3==0)
        self.three=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(1,3),stride=1,padding=(0,1)) ,
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(3,1),stride=1,padding=(1,0)) 
                                )
        #padding=(x,y) x是在高度上增高,y是在宽度上变宽
        self.five=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(1,5),stride=1,padding=(0,2)),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(5,1),stride=1,padding=(2,0)) 
                                )
        
        
        
        self.seven=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(1,7),stride=1,padding=(0,3)) ,
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=int(input_channel/3),kernel_size=(7,1),stride=1,padding=(3,0))
                                )
    
    def forward(self,input):
        identity_data = input
        output1=self.three(input)  
        #print("output1",output1.shape) #torch.Size([1, 20, 240, 240])
        
        output2=self.five(input)
        #print("output2",output2.shape) #torch.Size([1, 20, 240, 240])
        
        output3=self.seven(input) 
        #print("output3",output3.shape) #torch.Size([1, 20, 240, 240])
        
        output=torch.cat((output1,output2,output3),dim=1) # 60
        #print("output",output.shape)  #torch.Size([1, 60, 240, 240])
        
        output = torch.add(output, identity_data)
        #output = nn.ReLU(inplace=True)(output)
        
        return output
    
    

########################   HrMSI:Initial feature extraction module     #############################

########################   LrHSI:Initial feature extraction module     #############################
#输出波段为endmember_num/4
def def_lr_hsi_initial_feature(input_channel,output_channel,device,block_num,init_type='kaiming', init_gain=0.02):
    
    #input_channel：lr_hsi的波段数
    #output_channel：self.begin输出的波段数
    #endmember_num：端元个数
    net = lr_hsi_initial_feature(input_channel,  output_channel, block_num)

    return init_net(net, init_type, init_gain,device) 
    
class lr_hsi_initial_feature(nn.Module):
    def __init__(self,input_channel,output_channel,block_num): 

        #input_channel：lr_hsi的波段数
        #output_channel：输出的波段数
       
        super().__init__()
        
       
        self.begin=nn.Conv2d(in_channels=input_channel,out_channels=60,kernel_size=1,stride=1) 
        
        layer = []
        for i in range(block_num):
            layer.append(
                                spectral_res_block(60)
                              ) 
        self.middle=nn.Sequential(*layer)
        #self.middle=spectral_res_block(60) #不改变波段数
        
        ###最后输出的波段数量是int(endmember_num/4)
        self.end=nn.Conv2d(in_channels=60,out_channels=output_channel,kernel_size=1,stride=1)
    
    def forward(self,input):
        output1=self.begin(input)
        #print("output1~",output1.shape) torch.Size([1, 60, 240, 240])
        output2=self.middle(output1)
        #print("output2~",output2.shape) torch.Size([1, 60, 240, 240])
        output3=self.end(output2)
        #print("output3~",output3.shape)   torch.Size([1, 100, 240, 240])
        
        
        return output3 #1 endnum  h w
    

class spectral_res_block(nn.Module):
    def __init__(self,input_channel): #input_channel 60
        super().__init__()
        self.one=nn.Sequential(
        nn.Conv2d(in_channels=input_channel,out_channels=int(input_channel/3),kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=int(input_channel/3),out_channels=input_channel,kernel_size=1,stride=1) ,
        nn.ReLU(inplace=True)
                                )
        
    def forward(self,input):
        identity_data = input
        output = self.one(input) # 60
        output = torch.add(output, identity_data) # 60
        #output = nn.ReLU(inplace=True)(output) # 60
        return output 
    
########################   LrHSI:Initial feature extraction module     #############################


########################   Down_up_sample     #############################


class Down_up(nn.Module):
    """实现空间上采样以及下采样 """

    def __init__(self, wh):
        super().__init__()
        
        self.w=wh[0]
        self.h=wh[1]
        
        #self.down=nn.AdaptiveAvgPool2d(output_size=(self.ker_size, self.ker_size))
        
    def forward(self, x):
        
        return fun.interpolate(x,(self.w,self.h),mode='bilinear')
    

########################   Down_up_sample     #############################


########################   feature_extraction     #############################
def def_feature_extraction(input_channel,output_channel,device,init_type='kaiming', init_gain=0.02): 
   
    net = feature_extraction(input_channel, output_channel)

    return init_net(net, init_type, init_gain,device) 
    
class feature_extraction(nn.Module):
    def __init__(self,input_channel,output_channel): #中间层为30

        
        super().__init__()
       
        self.one=nn.Sequential(
            
        nn.Conv2d(in_channels=input_channel,out_channels=30,kernel_size=1,stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=30,out_channels=output_channel,kernel_size=1,stride=1) ,
        nn.ReLU(inplace=True)
        
                       )   
    def forward(self,input):
        
        return self.one(input)
########################   feature_extraction     #############################


########################   abundance_reconstruction     ############################# 
def define_abundance_reconstruction(input_channel,output_channel,device,activation='clamp', init_type='kaiming', init_gain=0.02): 
    
    net = abundance_reconstruction(input_channel,output_channel,activation)

    return init_net(net, init_type, init_gain,device) 

class abundance_reconstruction(nn.Module):
    def __init__(self,input_channel,output_channel,activation): # #中间层为30
        assert(activation in ['sigmoid','softmax','clamp','No'])
        super().__init__()
        
        self.activation=activation
        self.layer1=nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=30,kernel_size=3,stride=1,padding=1),
               
                nn.ReLU(inplace=True),
                
                nn.Conv2d(in_channels=30,out_channels=output_channel,kernel_size=3,stride=1,padding=1),
                
            )
    def forward(self,input):
        output=0.0005*self.layer1(input)
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output)
        if self.activation =='clamp':
            return output.clamp_(0,1)
        if self.activation =='softmax':
            return nn.Softmax(dim=1)(output)
        else:     #'No' 不需要激活函数
            return  output
########################   abundance_reconstruction     #############################



########################   Cross_Modality Information Interactive module     ############################# 



def define_CMII(input_ouput_channel,wh,device, use_CMII='No',init_type='kaiming', init_gain=0.02): 
    
    net = CMII(input_ouput_channel,wh,use_CMII)

    return init_net(net, init_type, init_gain,device) 


class SELayer(nn.Module):  #输入波段为channel，生成channel波段的attention map
    def __init__(self, channel, reduction=5):
        super(SELayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,stride=1,padding=0, bias=False), #in_channels=input_channel, out_channels=30,kernel_size=3,stride=1,padding=1
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,stride=1,padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        #return x * y.expand_as(x)
        return self.layer(x)          #返回1 × channel × 1 × 1
       

class CMII(nn.Module):
    def __init__(self,input_ouput_channel,wh,use_CMII): # wh为元组或者list
        
        super().__init__()
        self.wh=wh
        assert use_CMII in ['Yes','No']
        #print("self.wh:{}".format(self.wh))
        self.use_CMII=use_CMII
        if self.use_CMII == 'Yes' :            #判断是否使用CMII模块
            self.SE3=SELayer(30)
            self.SE7=SELayer(30)
            
            
            self.scale3=nn.Sequential(
                    nn.Conv2d(in_channels=input_ouput_channel, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True),     
                    nn.Conv2d(in_channels=30,out_channels=30,kernel_size=3,stride=1,padding=1),
                )
            
            self.scale5=nn.Sequential(
                    nn.Conv2d(in_channels=input_ouput_channel, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True),     
                    nn.Conv2d(in_channels=30,out_channels=30,kernel_size=5,stride=1,padding=2),
                )
            
            self.scale7=nn.Sequential(
                    nn.Conv2d(in_channels=input_ouput_channel, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True),     
                    nn.Conv2d(in_channels=30,out_channels=30,kernel_size=7,stride=1,padding=3),
                )
            
            self.net1=nn.Sequential(
                    nn.Conv2d(in_channels=60, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True)      
                )
            
            self.net2=nn.Sequential(
                    nn.Conv2d(in_channels=60, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True)      
                )
            
            self.net3=nn.Sequential(
                    nn.Conv2d(in_channels=30, out_channels=30,kernel_size=3,stride=1,padding=1),             
                    nn.ReLU(inplace=True)     ,
                    nn.Conv2d(in_channels=30, out_channels=input_ouput_channel,kernel_size=3,stride=1,padding=1), 
                )
            
            self.net4=nn.Sequential(
                    nn.Conv2d(in_channels=input_ouput_channel, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True)  ,
                    nn.Conv2d(in_channels=30, out_channels=input_ouput_channel,kernel_size=1,stride=1),  
                )
            
            self.down_up=Down_up(self.wh) # wh为元组或者list
        
        else:
            self.net4=nn.Sequential(
                    nn.Conv2d(in_channels=input_ouput_channel, out_channels=30,kernel_size=1,stride=1),             
                    nn.ReLU(inplace=True)  ,
                    nn.Conv2d(in_channels=30, out_channels=input_ouput_channel,kernel_size=1,stride=1),  
                )
            
            
            
    def forward(self,auxiliary,original):
        if self.use_CMII == 'Yes'  :        
            scale3_out=self.scale3(auxiliary) #30个波段
            scale5_out=self.scale5(auxiliary) #30个波段
            scale7_out=self.scale7(auxiliary) #30个波段
            assert(scale3_out.shape == scale5_out.shape)
            assert(scale5_out.shape == scale7_out.shape)
            #print('scale3_out_shape {},scale5_out_shape {},scale7_out_shape {}'.format(scale3_out.shape,scale5_out.shape,scale7_out.shape))
            
            
            concat35=torch.cat((scale3_out,scale5_out),dim=1) #60个波段
            concat57=torch.cat((scale5_out,scale7_out),dim=1) #60个波段
            assert(concat35.shape == concat57.shape)
            #print('concat35_shape {},concat57_shape {}'.format(concat35.shape,concat57.shape))
            
            
            
            SE3_attention=self.SE3(scale3_out) #1 × 30个波段 × 1 × 1
            SE7_attention=self.SE7(scale7_out) #1 × 30个波段 × 1 × 1
            #print('SE3_attention_shape {},SE7_attention_shape {}'.format(SE3_attention.shape,SE7_attention.shape))
            
            
            concat35_pro=self.net1(concat35) #30个波段
            concat57_pro=self.net2(concat57) #30个波段
            assert(concat35_pro.shape == concat57_pro.shape)
            #print('concat35_pro_shape {},concat57_pro_shape {}'.format(concat35_pro.shape,concat57_pro.shape))
            
            
            concat35_pro_attention = concat35_pro * SE3_attention.expand_as(concat35_pro)  #30个波段
            concat57_pro_attention = concat57_pro * SE7_attention.expand_as(concat57_pro)  #30个波段
            #print('concat35_pro_attention_shape {},concat57_pro_attention_shape {}'.format(concat35_pro_attention.shape,concat57_pro_attention.shape))
            
            
            assert(concat35_pro_attention.shape == concat57_pro_attention.shape)
            
            sum_out=torch.add(concat35_pro_attention,concat57_pro_attention)  #30个波段
            #print('sum_out_shape {}'.format(sum_out.shape))

            
            sum_out_pro=self.net3(sum_out)  #input_ouput_channel个波段
            #print('sum_out_pro_shape {}'.format(sum_out_pro.shape))
            
            sum_out_pro_interpolate =  self.down_up(sum_out_pro)  #让sum_out_pro的空间尺寸和original_pro一样
            #print('sum_out_pro_interpolate_shape {}'.format(sum_out_pro_interpolate.shape))
            
            ######
            original_pro=self.net4(original)
            #print('original_pro_shape {}'.format(original_pro.shape))
            
             
            assert(sum_out_pro_interpolate.shape == original_pro.shape)
            
            final_out=torch.add(original_pro,sum_out_pro_interpolate)
        
        else:
            final_out=self.net4(original)
            #print('final_out_shape {}'.format(final_out.shape))
            
        return final_out
########################   Cross_Modality Information Interactive module     #############################




########################  abundance2image   ###################################

def define_abundance2image(output_channel,device,endmember_num=100,activation='clamp', init_type='kaiming', init_gain=0.02): #A
    
   
    #output_channel：从丰度恢复出图像的波段数
    #endmember_num：端元个数
    net = abundance2image(endmember_num, output_channel,activation)

    return init_net(net, init_type, init_gain,device) 


class abundance2image(nn.Module): #参数充当端元
    def __init__(self, endmember_num, output_channel,activation):
        assert(activation in ['sigmoid','clamp','No'])
        super(abundance2image, self).__init__()
        self.activation=activation
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=endmember_num, out_channels=output_channel, kernel_size=1, stride=1,bias=False),
        )
        
    def forward(self, input):
        output=self.layer(input)
        
        if self.activation =='sigmoid':
            return torch.sigmoid(output)
        if self.activation =='clamp':
            return output.clamp_(0,1)
        else:     #'No' 不需要激活函数
            return  output
        
        return output #1 endnum  h w  
    
########################  abundance2image   ###################################
    
''' PSF and SRF '''    
class PSF_down():

    def __call__(self, input_tensor, psf, ratio): #PSF为#1 1 ratio ratio 大小的tensor
        _,C,_,_=input_tensor.shape[0],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3]
        if psf.shape[0] == 1:
            psf = psf.repeat(C, 1, 1, 1) #8X1X8X8
                                               #input_tensor: 1X8X400X400
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=C) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor

class SRF_down():
  
    def __call__(self, input_tensor, srf): # srf 为 ms_band hs_bands 1 1 的tensor      
        output_tensor = fun.conv2d(input_tensor, srf, None)
        return output_tensor
''' PSF and SRF '''  

''' 将参数裁剪到[0,1] '''
class ZeroOneClipper(object):

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1)
''' 将参数裁剪到[0,1] '''




''' TV '''
class TVLoss(nn.Module):
    def __init__(self, data,args) :
   

        super().__init__()
        
        self.args=args
        
        self.weight=self.cal_weight(data,self.args.flag) #两维度的 h-1 × w-1 cuda tensor
        
    
    def cal_weight(self,data,flag=0.8): #输入的data是三维度的numpy H W C 输出的是两维度的 H W cuda tensor
        
        tv_h = np.abs(data[:-1,:,:] - data[1:,:,:]  ) #  shape: h-1 × w × c
        tv_w = np.abs(data[:,:-1,:] - data[:,1:,:]  ) #  shape: h × w-1 × c
        
        tv_h_temp = tv_h[:,:-1,:] #空间尺寸比原始图像小1  shape:h-1 × w-1 × c
        tv_w_temp = tv_w[:-1,:,:] #空间尺寸比原始图像小1  shape:h-1 × w-1 × c
        
        tv = tv_h_temp + tv_w_temp   #空间尺寸比原始图像小1  shape:h-1 × w-1 × c
        
        TV=np.sum(tv,axis=2) #每个波段TV求和 shape: h-1 × w-1
        
        number_flag= int (  len( TV.flatten() )*flag )
        
        sort_result=sorted(TV.flatten(),reverse=True) #从大到小
        
        threshold=sort_result[-number_flag] #找到所有值的flag位置

        TV[TV>threshold]=threshold #大于该阈值的都设置为该阈值 shape: h-1 × w-1
        
        #print("TV max{}".format(TV.max()))
        #print("TV min{}".format(TV.min()))
        
        TV_normalization =  (TV-TV.min())/( TV.max()-TV.min()  ) #shape: h-1 × w-1 归一化到0-1
        
        inverse_V_normalization=  1-TV_normalization  #shape: h-1 × w-1  梯度大的地方 TV 约束小
        
        return torch.from_numpy(inverse_V_normalization).to(self.args.device) #shape: h-1 × w-1 cuda tensor
        
    def cal_tv(self,data):
        #
        batch_size, c, h, w = data.size()
        #tv_h = torch.abs(x[:,:,:-1,:] - x[:,:,1:,:]  ).sum()
        #tv_w = torch.abs(x[:,:,:,:-1] - x[:,:,:,1:]  ).sum()
        tv_h = torch.abs(data[:,:,:-1,:] - data[:,:,1:,:]  ) #  shape:1 × c × h-1 × w
        tv_w = torch.abs(data[:,:,:,:-1] - data[:,:,:,1:]  ) #  shape:1 × c × h × w-1
        
        tv_h_temp = tv_h[:,:,:,:-1] #空间尺寸比原始图像小1  shape:1 × c × h-1 × w-1
        tv_w_temp = tv_w[:,:,:-1,:] #空间尺寸比原始图像小1  shape:1 × c × h-1 × w-1
        
        tv = tv_h_temp + tv_w_temp   #空间尺寸比原始图像小1  shape:1 × c × h-1 × w-1
        
        TV=torch.sum(tv,dim=1,keepdim=True) #每个波段TV求和 shape:1 × 1 × h-1 × w-1
        
        return TV
    
    def forward(self, data):
        
        TV = self.cal_tv(data)
        
        if self.args.use_ATV == '1':
            
            TV_loss = self.weight * TV
        
        if self.args.use_ATV == '2':
            
            TV_loss = TV
            
        return torch.sum(TV_loss)






if __name__ == "__main__":
   pass