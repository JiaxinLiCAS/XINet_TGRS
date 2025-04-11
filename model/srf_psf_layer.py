# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：估计未知的PSF和SRF
"""

import numpy as np
import scipy.io as sio
import os
import torch
import torch.nn.functional as fun
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from .read_data import readdata
from .evaluation import compute_sam,compute_psnr,compute_ergas,compute_cc,compute_rmse
import random
import matplotlib.pyplot as plt




class BlurDown(object):
    def __init__(self):
        #self.shift_h = shift_h
        #self.shift_w = shift_w
        #self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1) #8X1X8X8
        
        
        output_tensor = fun.conv2d(input_tensor, psf, None, (ratio, ratio),  groups=groups) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor
    
class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size #8
        self.ratio = ratio #8
        
        #psf = torch.rand([1, 1, self.ker_size, self.ker_size]) #0-1均匀分布
        psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        self.psf = nn.Parameter(psf)
        
        #srf = torch.rand([self.ms_bands, self.hs_bands, 1, 1]) #0-1均匀分布
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands) 
        self.srf = nn.Parameter(srf)
        self.blur_down = BlurDown()

    def forward(self, lr_hsi, hr_msi):
        
        srf_div = torch.sum(self.srf, dim=1, keepdim=True) # 8 x 1x 1 x 1
        #print('srf_div',srf_div,srf_div.shape)  #
        
        srf_div = torch.div(1.0, srf_div)     #8 x 1x 1 x 1
        #print('srf_div',srf_div,srf_div.shape)
        
        srf_div = torch.transpose(srf_div, 0, 1)  # 1 x l x 1 x 1    1 x 8 x 1 x 1
        #print('srf_div',srf_div,srf_div.shape)
        
        lr_msi_fhsi = fun.conv2d(lr_hsi, self.srf, None) #(1,8,30, 30)
        lr_msi_fhsi = torch.mul(lr_msi_fhsi, srf_div) #element-wise broadcast Ylow:1X8X30X30
        lr_msi_fhsi = torch.clamp(lr_msi_fhsi, 0.0, 1.0)
        lr_msi_fmsi = self.blur_down(hr_msi, self.psf,  self.ms_bands, self.ratio)
        lr_msi_fmsi = torch.clamp(lr_msi_fmsi, 0.0, 1.0)
        return lr_msi_fhsi, lr_msi_fmsi


class Blind(readdata):
    def __init__(self, args):
        super().__init__(args)
        #self.strBR = 'BR.mat'
        # set
        self.S1_lr = args.S1_lr
        self.ker_size = self.args.scale_factor  #8
        self.ratio    = self.args.scale_factor 
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]
        # variable, graph and etc.
        #self.__hsi = torch.tensor(self.hsi)
        #self.__msi = torch.tensor(self.msi)
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ker_size, self.ratio).to(self.args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.S1_lr)
        

    def train(self, max_iter=5000, verb=True):
        
        #hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        lr_hsi, hr_msi = self.tensor_lr_hsi.to(self.args.device), self.tensor_hr_msi.to(self.args.device)
        for epoch in range(1, max_iter+1):
            lr_msi_fhsi_est, lr_msi_fmsi_est = self.model(lr_hsi, hr_msi)
            #Ylow, Zlow = self.model(lr_hsi, hr_msi)
            #print(lr_msi_fhsi_est.shape)
            loss = torch.sum(torch.abs(lr_msi_fhsi_est - lr_msi_fmsi_est))
            
                    
            self.optimizer.zero_grad()
            loss.backward()
            
            #print('更新之前',torch.sum(self.model.srf, dim=1, keepdim=True))
            
            self.optimizer.step()
            #print('更新之前',torch.sum(self.model.srf, dim=1, keepdim=True))
            
            self.model.apply(self.check_weight)
            #print('更新之前',torch.sum(self.model.srf, dim=1, keepdim=True))
            
            with torch.no_grad():
                if verb is True:
                    if (epoch ) % 100 == 0:
                        print("____________________________________________")
                        print('epoch: %s, lr: %s, loss: %s' % (epoch, self.S1_lr, loss))
                        
                        lr_msi_fhsi_est_numpy=lr_msi_fhsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        lr_msi_fmsi_est_numpy=lr_msi_fmsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                        self.lr_msi_fhsi_est_numpy=lr_msi_fhsi_est_numpy
                        self.lr_msi_fmsi_est_numpy=lr_msi_fmsi_est_numpy
                        train_message='生成的两个图像 train epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( lr_msi_fhsi_est_numpy- lr_msi_fmsi_est_numpy ) ) ,
                                         compute_sam(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy) ,
                                         compute_psnr(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy) ,
                                         compute_ergas(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy, self.ratio),
                                         compute_cc(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy),
                                         compute_rmse(lr_msi_fhsi_est_numpy, lr_msi_fmsi_est_numpy)
                                             )
                        print(train_message)
                        
                        print('************')
                        test_message_SRF='SRF lr_msi_fhsi_est与lr_msi_fhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_msi_fhsi- lr_msi_fhsi_est_numpy ) ) ,
                                         compute_sam(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy) ,
                                         compute_psnr(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy) ,
                                         compute_ergas(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy),
                                         compute_rmse(self.lr_msi_fhsi, lr_msi_fhsi_est_numpy)
                                         )
                        print(test_message_SRF)
                        
                        print('************')
                        test_message_PSF='PSF lr_msi_fmsi_est与lr_msi_fmsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_msi_fmsi- lr_msi_fmsi_est_numpy ) ) ,
                                         compute_sam(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) ,
                                         compute_psnr(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) ,
                                         compute_ergas(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy),
                                         compute_rmse(self.lr_msi_fmsi, lr_msi_fmsi_est_numpy) 
                                         )
                        print(test_message_PSF)
                        
                        
                        psf_info="estimated psf \n {} \n psf_gt \n{}".format(
                            np.squeeze(self.model.psf.data.cpu().detach().numpy()),
                            self.psf_gt 
                            )
                        #print(psf_info) 自行打印出来预测的PSF
                        
                       
                        srf_info="estimated srf \n {} \n srf_gt \n{}".format(
                            np.squeeze(self.model.srf.data.cpu().detach().numpy()).T,
                            self.srf_gt 
                            )
                        #print(srf_info)  自行打印出来预测的SRF
                       
                        
                        
                        #####从目标高光谱空间下采样####
                        psf = self.model.psf.repeat(self.hs_bands, 1, 1, 1)
                        lr_hsi_est = fun.conv2d(self.tensor_gt.to(self.args.device), 
                                                psf, None, (self.ker_size, self.ker_size),  
                                                groups=self.hs_bands)

                        lr_hsi_est_numpy=lr_hsi_est.data.cpu().detach().numpy()[0].transpose(1,2,0)
                
                        print('************')
                        from_hrhsi_PSF='PSF lr_hsi_est与lr_hsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.lr_hsi- lr_hsi_est_numpy ) ) ,
                                         compute_sam(self.lr_hsi, lr_hsi_est_numpy) ,
                                         compute_psnr(self.lr_hsi, lr_hsi_est_numpy) ,
                                         compute_ergas(self.lr_hsi, lr_hsi_est_numpy,self.ratio),
                                         compute_cc(self.lr_hsi, lr_hsi_est_numpy),
                                         compute_rmse(self.lr_hsi, lr_hsi_est_numpy) 
                                         )
                        print(from_hrhsi_PSF) 
                        #####从目标高光谱空间下采样####
                        
                        #####从目标高光谱光谱下采样####
                        if self.model.srf.data.cpu().detach().numpy().shape[0]!=1:
                            srf_est=np.squeeze(self.model.srf.data.cpu().detach().numpy()).T
                        else:
                            srf_est_tmp=np.squeeze(self.model.srf.data.cpu().detach().numpy()).T
                            srf_est=srf_est_tmp[:,np.newaxis]                   
                        w,h,c = self.gt.shape
                        if srf_est.shape[0] == c:
                            hr_msi_est_numpy = np.dot(self.gt.reshape(w*h,c), srf_est).reshape(w,h,srf_est.shape[1])
                        print('************')
                        from_hrhsi_SRF='SRF hr_msi_est与hr_msi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC:{}, rmse:{}\n'.\
                                  format(epoch,self.S1_lr,
                                         np.mean( np.abs( self.hr_msi- hr_msi_est_numpy ) ) ,
                                         compute_sam(self.hr_msi, hr_msi_est_numpy) ,
                                         compute_psnr(self.hr_msi, hr_msi_est_numpy) ,
                                         compute_ergas(self.hr_msi, hr_msi_est_numpy,self.ratio),
                                         compute_cc(self.hr_msi, hr_msi_est_numpy),
                                         compute_rmse(self.hr_msi, hr_msi_est_numpy) 
                                         )
                        print(from_hrhsi_SRF)        
                        #####从目标高光谱光谱下采样####
                        
                        
                        
                       
        
        PATH=os.path.join(self.args.expr_dir,self.model.__class__.__name__+'.pth')
        torch.save(self.model.state_dict(),PATH)
        #self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        #self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))

    def get_save_result(self, is_save=True):
        
        #self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        psf = self.model.psf.data.cpu().detach().numpy() ## 1 1 15 15
        srf = self.model.srf.data.cpu().detach().numpy() # 8 46 1 1
        psf = np.squeeze(psf)  #15 15
        srf = np.squeeze(srf).T  # b x B 8 X 46 变为 46X8 和srf_gt保持一致
        self.psf, self.srf = psf, srf
        if is_save is True:
            #PATH_psf=os.path.join(self.args.expr_dir,'psf_est')
            #PATH_srf=os.path.join(self.args.expr_dir,'srf_est')
            #sio.savemat(PATH_psf, {'psf_est': self.psf})
            #sio.savemat(PATH_srf, {'srf_est': self.srf})
            sio.savemat(os.path.join(self.args.expr_dir , 'estimated_psf_srf.mat'), {'psf_est': psf, 'srf_est': srf})
        return

    @staticmethod
    def check_weight(model):
        
        if hasattr(model, 'psf'):
            #print(model,'psf')
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)             
            psf_div = torch.div(1.0, psf_div)                                                                 
            w.mul_(psf_div)
        
        if hasattr(model, 'srf'):
            #print(model,'srf')
            w = model.srf.data # torch.Size([8, 46, 1, 1])        
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True) #torch.Size([8, 1, 1, 1])
            srf_div = torch.div(1.0, srf_div) #torch.Size([8, 1, 1, 1])
            w.mul_(srf_div)
            
if __name__ == "__main__":
    pass