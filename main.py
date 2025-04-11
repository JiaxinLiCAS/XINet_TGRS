# -*- coding: utf-8 -*-

"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""
"""
❗❗❗❗❗❗#此py作用：main文件，点击运行
"""


import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os

import random
import scipy.io as sio
import time


from model.config import args

from utils.visualizer import Visualizer

from model.srf_psf_layer import Blind       #将退化函数看作一层的参数
from model.evaluation import MetricsCal


#设定随机种子 方便调参
def setup_seed(seed):
             torch.manual_seed(seed)
             torch.cuda.manual_seed_all(seed)
             np.random.seed(seed)
             random.seed(seed)
             torch.backends.cudnn.deterministic = True
             print(1)
        
setup_seed(30) #30
   
start = time.perf_counter() # 记录开始时间    

from model.fusion import Fusion
if args.blind == 'Yes': #如果是盲融合，就学习退化函数PSF SRF
            
    blind = Blind(args)
    
    blind.train()                    #学习退化函数PSF SRF
    blind.get_save_result()  #保存预测的PSF和SRF
            
    Model=Fusion(args,blind.psf,blind.srf)    #初始化网络 使用学习到的退化函数
    
if args.blind == 'No':
    
    Model=Fusion(args)    #初始化网络 使用给出的退化函数
           

visualizer = Visualizer(args, Model.srf_gt) #初始化visdom
      
for epoch in range(args.epoch_count, args.niter + args.niter_decay + 1):
        
        Model.optimize_joint_parameters()
        
        if epoch % args.print_freq == 0:  #满足一定的训练轮次 就通过Visdom可视化以及控制台输出结果
            with torch.no_grad():
                hues.info("[{}/{}]".format(epoch,args.niter + args.niter_decay + 1))
                losses = Model.get_current_losses()
                visualizer.print_current_losses(epoch, losses) #将loss保存到loss_log.txt里面 并 在控制台里输出print

                if args.display_id > 0:
                    visualizer.plot_current_losses(epoch,  losses) #将losses可视化在Visdom
                    visualizer.display_current_results(Model.get_current_visuals(),win_id=[1]) #将中间的6个结果可视化在Visdom，分别是真值LrHSI HrMSI GT 以及 对应的自编码器重建出来的LrHSI HrMSI 和 GT'

                    visualizer.plot_spectral_lines(Model.get_current_visuals(), #光谱曲线可视化在Visdom 每轮都随机选点  #1.GT以及重建GT的曲线，2.lrhsi以及自编码器重建出lrhsi的曲线
                                                    visual_corresponding_name=Model.visual_corresponding_name,
                                                    win_id=[2,3])
                    
                    #将PSNR 和 SAM 可视化 ，保存到precision.txt，保存到psnr_and_sam.pickle
                    visualizer.plot_psnr_sam(Model.get_current_visuals(), 
                                              epoch, Model.visual_corresponding_name)

                    #将学习率可视化
                    visualizer.plot_lr( Model.get_LR(), epoch)
                
                sam,psnr,ergas,cc,rmse,Ssim,Uqi=MetricsCal(Model.gt,Model.gt_est.data.cpu().float().numpy()[0].transpose(1,2,0), args.scale_factor)
                print("sam {},psnr {},ergas {},cc {},rmse {},Ssim {},Uqi {}".format(sam,psnr,ergas,cc,rmse,Ssim,Uqi))
                
        Model.update_learning_rate()
        
end = time.perf_counter()   # 记录结束时间
elapsed = end - start        # 计算经过的时间（单位为秒）        
print(elapsed)

#保存最终重建结果
gt_est=Model.gt_est.data.cpu().float().numpy()[0]
file_name = os.path.join(args.expr_dir, 'Out.mat')
sio.savemat(file_name,{'Out': gt_est.transpose(1,2,0)})


print('李嘉鑫 作者微信 BatAug 欢迎加微信交流 空天信息创新研究院20-25直博生，导师高连如')

"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如
"""

