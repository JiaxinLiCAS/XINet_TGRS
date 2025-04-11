# -*- coding: utf-8 -*-
"""
❗❗❗❗❗❗李嘉鑫 作者微信 BatAug 欢迎加微信交流
空天信息创新研究院20-25直博生，导师高连如

"""
"""
❗❗❗❗❗❗#此py作用：超分融合网络主体
"""



import torch
import torch.nn
import itertools
import numpy as np
from . import network
from collections import OrderedDict
from .read_data import readdata


class Fusion(readdata):
    
    def __init__(self,args,psf_est=None,srf_est=None):
        #self.args = args在父类readdata定义了
        
        super().__init__(args)
        self.hs_bands = self.srf_gt.shape[0]
        self.ms_bands = self.srf_gt.shape[1]
        self.lrhsi_scale=[
                          (  self.lr_hsi.shape[0],self.lr_hsi.shape[1]  ),
                          (  int(self.lr_hsi.shape[0]/2),int(self.lr_hsi.shape[1]/2)  ),
                          (  int(self.lr_hsi.shape[0]/4), int(self.lr_hsi.shape[1]/4) )
                          ]
        
        self.hrmsi_scale=[
                          (  self.hr_msi.shape[0],self.hr_msi.shape[1]  ),
                          (  int(self.hr_msi.shape[0]/2),int(self.hr_msi.shape[1]/2)  ),
                          (  int(self.hr_msi.shape[0]/4), int(self.hr_msi.shape[1]/4) )
                          ]        
        
 
        #获取SRF and PSF
        if psf_est is not None :
            self.psf=psf_est
        else:#在父类readdata定义了
            self.psf=self.psf_gt
        self.psf = np.reshape(self.psf, newshape=(1, 1, self.args.scale_factor, self.args.scale_factor)) #1 1 ratio ratio 大小的tensor
        self.psf = torch.tensor(self.psf).to(self.args.device).float()
       
        if srf_est is not None:
            self.srf=srf_est
        else:
            self.srf=self.srf_gt
        self.srf = np.reshape(self.srf.T, newshape=(self.ms_bands, self.hs_bands, 1, 1)) #self.srf.T 有一个T转置
        self.srf = torch.tensor(self.srf).to(self.args.device).float()             # ms_band hs_bands 1 1 的tensor
        #获取SRF and PSF
        
        #初始化XINet所需要的网络
        self.initialize_network()
        print("initialize_network over")
        
        #初始化loss
        self.initialize_loss()
        print("initialize_loss over")
        
        #初始化optimizer 和 schedulers
        self.initialize_optimizer_scheduler()
        print("initialize_optimizer_scheduler over")
        
        #输出模型参数个数
        self.get_information()
        self.print_parameters()
        print("get_information over")
        print("print_parameters over")
        
    def get_information(self):
        #self.model_names = ['net_hr_msi_stream', 'net_lr_hsi_stream', 'net_shared_stream', 'net_abun2hrmsi', 'net_abun2lrhsi']
        
        self.model_names=[            
            'net_lrhsi_initial','net_feature_extraction_1','net_hrmsi_initial','net_feature_extraction_3',
            'net_feature_extraction_share',   
            'net_CMII1','net_CMII2','net_CMII3','net_CMII4',
            'net_feature_extraction_2','net_lrhsi_abundance_reconstruction','net_feature_extraction_4','net_hrmsi_abundance_reconstruction',
            'net_abun2lrhsi',  'net_abun2hrmsi'       
                          ]
        
        if self.args.use_ATV != '3':
            self.loss_names = ['loss_hr_msi_rec'             ,  'loss_lr_hsi_rec', 
                           'loss_hr_msi_from_hrhsi'       , 'loss_lr_hsi_from_hrhsi',
                           'loss_abundance_sum2one_hrmsi' , 'loss_abundance_sum2one_lrhsi',
                           'tv_loss_hr_msi','tv_loss_lr_hsi'
                           ]
        else:
            self.loss_names = ['loss_hr_msi_rec'             ,  'loss_lr_hsi_rec', 
                           'loss_hr_msi_from_hrhsi'       , 'loss_lr_hsi_from_hrhsi',
                           'loss_abundance_sum2one_hrmsi' , 'loss_abundance_sum2one_lrhsi',      
                           ]
            

        self.visual_names = ['tensor_lr_hsi','lr_hsi_rec', 
                             'tensor_hr_msi','hr_msi_rec',
                             'tensor_gt','gt_est']
        
        self.visual_corresponding_name={}
        self.visual_corresponding_name['tensor_lr_hsi'] = 'lr_hsi_rec'
        self.visual_corresponding_name['tensor_hr_msi'] = 'hr_msi_rec'
        self.visual_corresponding_name['tensor_gt']     = 'gt_est'
        
    def initialize_network(self):
        #初始化XINet所需要的网络
        '''
        self.lrhsi_scale=[
                          (  self.lr_hsi.shape[0],self.lr_hsi.shape[1]  ),
                          (  int(self.lr_hsi.shape[0]/2),int(self.lr_hsi.shape[1]/2)  ),
                          (  int(self.lr_hsi.shape[0]/4), int(self.lr_hsi.shape[1]/4) )
                          ]
        
        self.hrmsi_scale=[
                          (  self.hr_msi.shape[0],self.hr_msi.shape[1]  ),
                          (  int(self.hr_msi.shape[0]/2),int(self.hr_msi.shape[1]/2)  ),
                          (  int(self.hr_msi.shape[0]/4), int(self.hr_msi.shape[1]/4) )
                          ]
        '''
        
        '''Encoder'''
        
          # LrHSI
        self.net_lrhsi_initial = network.def_lr_hsi_initial_feature(input_channel=self.hs_bands , output_channel=int(self.args.endmember_num/4),
                                                                     device=self.args.device,block_num=self.args.block_num) #输出波段endmember_num/4
          
        self.down_up1=network.Down_up(self.lrhsi_scale[1]) #降低到原本的1/2
        
        self.net_feature_extraction_1 = network.def_feature_extraction(input_channel=int(self.args.endmember_num/4),
                                                                       output_channel=int(self.args.endmember_num/2),
                                                                       device=self.args.device) #输出波段endmember_num/2
        
        self.down_up2=network.Down_up(self.lrhsi_scale[2]) #降低到原本的1/4
        
          #HrMSI
        
        self.net_hrmsi_initial = network.def_hr_msi_initial_feature(input_channel=self.ms_bands , output_channel=int(self.args.endmember_num/4),
                                                                     device=self.args.device,block_num=self.args.block_num) #输出波段endmember_num/4
          
        self.down_up3=network.Down_up(self.hrmsi_scale[1]) #降低到原本的1/2
        
        self.net_feature_extraction_3 = network.def_feature_extraction(input_channel=int(self.args.endmember_num/4),
                                                                       output_channel=int(self.args.endmember_num/2),
                                                                       device=self.args.device) #输出波段endmember_num/2
        
        self.down_up4=network.Down_up(self.hrmsi_scale[2]) #降低到原本的1/4
        
        #define_CMII(input_ouput_channel,wh,device, use_CMMI=None,init_type='kaiming', init_gain=0.02)
          #CMMI
        
        # net_CMII1 提取Y1信息 注入到Z1 生成ZY1
        self.net_CMII1=network.define_CMII(input_ouput_channel = int(self.args.endmember_num/4),
                                       wh = self.hrmsi_scale[0] ,device = self.args.device, use_CMII=self.args.use_CMII)
        
        # net_CMII2 提取Y3信息 注入到Z3 生成ZY3
        self.net_CMII2=network.define_CMII(input_ouput_channel = int(self.args.endmember_num/2),
                                       wh = self.hrmsi_scale[1] ,device = self.args.device, use_CMII=self.args.use_CMII)
        
        # net_CMII3 提取Z1信息 注入到Y1 生成YZ1
        self.net_CMII3=network.define_CMII(input_ouput_channel = int(self.args.endmember_num/4),
                                       wh = self.lrhsi_scale[0] ,device = self.args.device, use_CMII=self.args.use_CMII)
        
        # net_CMII4 提取Z3信息 注入到Y3 生成YZ3
        self.net_CMII4=network.define_CMII(input_ouput_channel = int(self.args.endmember_num/2),
                                       wh = self.lrhsi_scale[1] ,device = self.args.device, use_CMII=self.args.use_CMII)
        
        
        '''share'''
        self.net_feature_extraction_share = network.def_feature_extraction(input_channel=int(self.args.endmember_num/2),
                                                                       output_channel=int(self.args.endmember_num/2),
                                                                       device=self.args.device) #输出波段endmember_num/2
        
        '''Decoder'''
        
          # LrHSI
        self.down_up5=network.Down_up(self.lrhsi_scale[1]) #恢复到原本的1/2
        
        #输入的波段数乘2 因为concat了
        self.net_feature_extraction_2 = network.def_feature_extraction(input_channel=int(self.args.endmember_num/2)*2,
                                                                       output_channel=int(self.args.endmember_num/4),
                                                                       device=self.args.device) #输出波段endmember_num/4
        
        self.down_up6=network.Down_up(self.lrhsi_scale[0]) #恢复到原本的1/1
        
         #输入的波段数乘2 因为concat了
        self.net_lrhsi_abundance_reconstruction = network.define_abundance_reconstruction(input_channel=int(self.args.endmember_num/4) * 2,
                                                                                    output_channel = self.args.endmember_num,
                                                                                    device= self.args.device,
                                                                                    activation=self.args.abundance_activation)
        
        
          # HrMSI
        self.down_up7=network.Down_up(self.hrmsi_scale[1]) #恢复到原本的1/2
        
        #输入的波段数乘2 因为concat了
        self.net_feature_extraction_4 = network.def_feature_extraction(input_channel=int(self.args.endmember_num/2)*2,
                                                                       output_channel=int(self.args.endmember_num/4),
                                                                       device=self.args.device) #输出波段endmember_num/4
        
        self.down_up8=network.Down_up(self.hrmsi_scale[0]) #恢复到原本的1/1
        
        #输入的波段数乘2 因为concat了
        self.net_hrmsi_abundance_reconstruction = network.define_abundance_reconstruction(input_channel=int(self.args.endmember_num/4) * 2,
                                                                                    output_channel = self.args.endmember_num,
                                                                                    device= self.args.device,
                                                                                    activation=self.args.abundance_activation)
        
        
        
        
        #丰度重建回图像
        self.net_abun2lrhsi = network.define_abundance2image(output_channel=self.hs_bands,device=self.args.device,
                                                         endmember_num=self.args.endmember_num,activation=self.args.abun2img_activation)
        
                                                    
        self.net_abun2hrmsi = network.define_abundance2image(output_channel=self.ms_bands,device=self.args.device,
                                                         endmember_num=self.args.endmember_num,activation=self.args.abun2img_activation)
       
      
        
        
        self.psf_down=network.PSF_down() #__call__(self, input_tensor, psf, ratio):
        self.srf_down=network.SRF_down() #__call__(self, input_tensor, srf):
        self.cliper_zeroone = network.ZeroOneClipper()    
        
    def initialize_loss(self):
        if self.args.Pixelwise_avg_crite == "No":
            self.criterionL1Loss = torch.nn.L1Loss(reduction='sum').to(self.args.device)  #reduction=mean sum
        else:
            self.criterionL1Loss = torch.nn.L1Loss(reduction='mean').to(self.args.device)
            
        self.criterionPixelwise = self.criterionL1Loss
        self.criterionSumToOne = network.SumToOneLoss().to(self.args.device)
        
        if self.args.use_ATV != '3':

            from .network import TVLoss
            self.Tv_loss_hr_msi=TVLoss(self.hr_msi,self.args)
            self.Tv_loss_lr_hsi=TVLoss(self.lr_hsi,self.args)
            
    def initialize_optimizer_scheduler(self):
        #optimizer
        '''
        self.model_names=[            
            'net_lrhsi_initial','net_feature_extraction_1','net_hrmsi_initial','net_feature_extraction_3',
            'net_feature_extraction_share', 
            'net_CMII1','net_CMII2','net_CMII3','net_CMII4',
            'net_feature_extraction_2','net_lrhsi_abundance_reconstruction','net_feature_extraction_4','net_hrmsi_abundance_reconstruction',
            'net_abun2lrhsi',  'net_abun2hrmsi'       
                          ]
        '''
        lr=self.args.S2_lr
        self.optimizers = []
        
        '''Encoder'''
        
          # LrHSI
        self.optimizer_lrhsi_initial = torch.optim.Adam(itertools.chain(self.net_lrhsi_initial.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_lrhsi_initial)
       
        self.optimizer_feature_extraction_1 = torch.optim.Adam(itertools.chain(self.net_feature_extraction_1.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_feature_extraction_1)
       
          # HrMSI
        self.optimizer_hrmsi_initial = torch.optim.Adam(itertools.chain(self.net_hrmsi_initial.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_hrmsi_initial)
       
        self.optimizer_feature_extraction_3 = torch.optim.Adam(itertools.chain(self.net_feature_extraction_3.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_feature_extraction_3)
        
          #share
        self.optimizer_feature_extraction_share = torch.optim.Adam(itertools.chain(self.net_feature_extraction_share.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_feature_extraction_share)
        
          #CMII
        self.optimizer_CMII1 = torch.optim.Adam(itertools.chain(self.net_CMII1.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_CMII1)
          
        self.optimizer_CMII2 = torch.optim.Adam(itertools.chain(self.net_CMII2.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_CMII2)
        
        self.optimizer_CMII3 = torch.optim.Adam(itertools.chain(self.net_CMII3.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_CMII3)
        
        self.optimizer_CMII4 = torch.optim.Adam(itertools.chain(self.net_CMII4.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_CMII4)
        
        '''Decoder'''
        
          # LrHSI
        self.optimizer_feature_extraction_2 = torch.optim.Adam(itertools.chain(self.net_feature_extraction_2.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_feature_extraction_2)
        
        self.optimizer_lrhsi_abundance_reconstruction = torch.optim.Adam(itertools.chain(self.net_lrhsi_abundance_reconstruction.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_lrhsi_abundance_reconstruction)
     
          #HrMSI
        self.optimizer_feature_extraction_4 = torch.optim.Adam(itertools.chain(self.net_feature_extraction_4.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_feature_extraction_4)
      
        self.optimizer_hrmsi_abundance_reconstruction = torch.optim.Adam(itertools.chain(self.net_hrmsi_abundance_reconstruction.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_hrmsi_abundance_reconstruction)
        
        
          #丰度重建回图像
        self.optimizer_abun2lrhsi = torch.optim.Adam(itertools.chain(self.net_abun2lrhsi.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_abun2lrhsi)
       
          #丰度重建回图像
        self.optimizer_abun2hrmsi = torch.optim.Adam(itertools.chain(self.net_abun2hrmsi.parameters()),
                                            lr=lr,betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer_abun2hrmsi)
        
        
        #scheduler
        self.schedulers = [network.get_scheduler(optimizer, self.args) for optimizer in self.optimizers]
        
        
    
        
    def optimize_joint_parameters(self):
        
        #前向传播
        self.forward()
        
        #梯度清零
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        #反向传播，求梯度
        self.backward_g_joint()
        
        for optimizer in self.optimizers:
            optimizer.step()
            
        #对端元裁剪到[0,1]
        self.net_abun2hrmsi.apply(self.cliper_zeroone)
        self.net_abun2lrhsi.apply(self.cliper_zeroone)
        
        
        
    def forward(self):
        
        ''' lrhsi Encoder  '''
        self.Y1=self.net_lrhsi_initial(self.tensor_lr_hsi)
        #print("self.Y1 shape:{}".format(self.Y1.shape))
        
        self.Y2=self.down_up1(self.Y1)
        #print("self.Y2 shape:{}".format(self.Y2.shape))
        
        self.Y3=self.net_feature_extraction_1(self.Y2)
        #print("self.Y3 shape:{}".format(self.Y3.shape))
        
        self.Y4=self.down_up2(self.Y3)
        #print("self.Y4 shape:{}".format(self.Y4.shape))
        
        ''' hrmsi Encoder  '''
        
        self.Z1=self.net_hrmsi_initial(self.tensor_hr_msi)
        #print("self.Z1 shape:{}".format(self.Z1.shape))
        
        self.Z2=self.down_up3(self.Z1)
        #print("self.Z2 shape:{}".format(self.Z2.shape))
        
        self.Z3=self.net_feature_extraction_3(self.Z2)
        #print("self.Z3 shape:{}".format(self.Z3.shape))
        
        self.Z4=self.down_up4(self.Z3)
        #print("self.Z4 shape:{}".format(self.Z4.shape))
        
        
        ''' 根据Encoder的Y1 Y3 和 Z1 Z3 生成decoder需要使用的 YZ1 YZ3 以及ZY1 ZY3'''
        # net_CMII1 提取Y1信息 注入到Z1 生成ZY1
        # net_CMII2 提取Y3信息 注入到Z3 生成ZY3
        # net_CMII3 提取Z1信息 注入到Y1 生成YZ1
        # net_CMII4 提取Z3信息 注入到Y3 生成YZ3
        
        #define_CMII(input_ouput_channel,wh,device, use_CMII=None,init_type='kaiming', init_gain=0.02)
        #forward(self,auxiliary,original)
        self.ZY1=self.net_CMII1(self.Y1, self.Z1)
        self.ZY3=self.net_CMII2(self.Y3, self.Z3)
        #print("self.ZY1 shape:{}".format(self.ZY1.shape))
        #print("self.ZY3 shape:{}".format(self.ZY3.shape))
        
        self.YZ1=self.net_CMII3(self.Z1, self.Y1)
        self.YZ3=self.net_CMII4(self.Z3, self.Y3)
        #print("self.YZ1 shape:{}".format(self.YZ1.shape))
        #print("self.YZ3 shape:{}".format(self.YZ3.shape))
        
        ''' share '''
        
        self.Y5=self.net_feature_extraction_share(self.Y4)
        self.Z5=self.net_feature_extraction_share(self.Z4)
        #print("self.Y5 shape:{}".format(self.Y5.shape))
        #print("self.Z5 shape:{}".format(self.Z5.shape))
        
        
        ''' lrhsi Decoder  '''
        
        self.Y6=self.down_up5(self.Y5)
        #print("self.Y6 shape:{}".format(self.Y6.shape))
        
        self.Y7 = self.net_feature_extraction_2 (  torch.cat((self.YZ3,self.Y6),dim=1)  )
        #print("self.Y7 shape:{}".format(self.Y7.shape))
        
        self.Y8=self.down_up6(self.Y7)
        #print("self.Y8 shape:{}".format(self.Y8.shape))
        
        self.lr_hsi_abundance= self.net_lrhsi_abundance_reconstruction ( torch.cat((self.YZ1,self.Y8),dim=1)  )
        #print("self.lr_hsi_abundance shape:{}".format(self.lr_hsi_abundance.shape))
        
        ''' hrmsi Decoder  '''
        self.Z6=self.down_up7(self.Z5)
        #print("self.Z6 shape:{}".format(self.Z6.shape))
        
        self.Z7 = self.net_feature_extraction_4 (  torch.cat((self.ZY3,self.Z6),dim=1)  )
        #print("self.Z7 shape:{}".format(self.Z7.shape))

        self.Z8=self.down_up8(self.Z7)
        #print("self.Z8 shape:{}".format(self.Z8.shape))
        
        self.hr_msi_abundance= self.net_hrmsi_abundance_reconstruction ( torch.cat((self.ZY1,self.Z8),dim=1)  )
        #print("self.hr_msi_abundance shape:{}".format(self.hr_msi_abundance.shape))
    
        
        
        '''从丰度重建回图像'''
        
        ''' hr_msi_abundance 2 hr_msi '''
        self.hr_msi_rec = self.net_abun2hrmsi(self.hr_msi_abundance)
        #print("self.hr_msi_rec shape:{}".format(self.hr_msi_rec.shape))
        
        ''' lr_hsi_abundance 2 lr_hsi '''
        self.lr_hsi_rec =  self.net_abun2lrhsi(self.lr_hsi_abundance)
        #print("self.lr_hsi_rec shape:{}".format(self.lr_hsi_rec.shape))
        
        ''' generate hrhsi_est '''
        self.gt_est= self.net_abun2lrhsi(self.hr_msi_abundance)
        #print("self.gt_est shape:{}".format(self.gt_est.shape))
        
        ''' generate hr_msi_est '''
        self.hr_msi_from_hrhsi = self.srf_down(self.gt_est,self.srf)
        #print("self.hr_msi_from_hrhsi shape:{}".format(self.hr_msi_from_hrhsi.shape))

        ''' generate lr_hsi_est '''
        self.lr_hsi_from_hrhsi = self.psf_down(self.gt_est, self.psf, self.args.scale_factor)
        #print("self.lr_hsi_from_hrhsi shape:{}".format(self.lr_hsi_from_hrhsi.shape))
        
        
       
        
        #self.psf_down=network.PSF_down() #__call__(self, input_tensor, psf, ratio):
        #self.srf_down=network.SRF_down() #__call__(self, input_tensor, srf):

        

    def backward_g_joint(self):
        '''
        self.loss_names = ['loss_hr_msi_rec'             ,  'loss_lr_hsi_rec', 
                           'loss_hr_msi_from_hrhsi'       , 'loss_lr_hsi_from_hrhsi',
                           'loss_abundance_sum2one_hrmsi' , 'loss_abundance_sum2one_lrhsi',
                           'loss_abundance_rec'
                           ]
        '''
        #lambda_A hr_msi 重建误差
        self.loss_hr_msi_rec=self.criterionPixelwise(self.tensor_hr_msi,self.hr_msi_rec)
        self.loss_hr_msi_rec_ceo=self.loss_hr_msi_rec*self.args.lambda_A
        
        #lambda_B lr_hsi 重建误差
        self.loss_lr_hsi_rec=self.criterionPixelwise(self.tensor_lr_hsi,self.lr_hsi_rec)
        self.loss_lr_hsi_rec_ceo=self.loss_lr_hsi_rec*self.args.lambda_B
        
        #lambda_C hr_msi_from_hrhsi \ lr_hsi_from_hrhsi 从预测的hr_hsi恢复到hr_msi\lr_hsi的误差
        self.loss_hr_msi_from_hrhsi=self.criterionPixelwise(self.tensor_hr_msi,self.hr_msi_from_hrhsi)
        self.loss_lr_hsi_from_hrhsi=self.criterionPixelwise(self.tensor_lr_hsi,self.lr_hsi_from_hrhsi)

        self.loss_degradation_ceo=(self.loss_hr_msi_from_hrhsi + self.loss_lr_hsi_from_hrhsi)*self.args.lambda_C
        
        
        if self.args.use_ATV != '3': #判断是否使用TV约束

            
            
            self.tv_loss_hr_msi=self.Tv_loss_hr_msi(self.hr_msi_abundance)
            self.tv_loss_lr_hsi=self.Tv_loss_lr_hsi(self.lr_hsi_abundance)
            
        
        #lambda_E abundance_sum2one
        self.loss_abundance_sum2one_hrmsi = self.criterionSumToOne(self.hr_msi_abundance)
        self.loss_abundance_sum2one_lrhsi = self.criterionSumToOne(self.lr_hsi_abundance)
        
        
        self.loss_abundance_sum2one_ceo = (self.loss_abundance_sum2one_hrmsi + self.loss_abundance_sum2one_lrhsi)*self.args.lambda_E
        
        if self.args.use_ATV != '3':  
            
            self.loss_all = self.loss_hr_msi_rec_ceo  + self.loss_lr_hsi_rec_ceo  + self.loss_degradation_ceo + self.loss_abundance_sum2one_ceo  + \
                          self.tv_loss_hr_msi +  self.tv_loss_lr_hsi
        else:
            self.loss_all = self.loss_hr_msi_rec_ceo  + self.loss_lr_hsi_rec_ceo  + self.loss_degradation_ceo + self.loss_abundance_sum2one_ceo
                
        #self.loss_degradation.backward(retain_graph=True)
        self.loss_all.backward(retain_graph=True)

    
    def print_parameters(self):
         all_parameters=[]
         for name in self.model_names:  #model_name 定义在 def initialize_network(self)里面
             if isinstance(name, str):
                 net = getattr(self,name)
                 num_params = 0
                 for param in net.parameters():
                     num_params += param.numel()
                 print('[Network %s] Total number of parameters : %.0f ' % (name, num_params))
                 all_parameters.append(num_params)
         print('Total number of all networks :{}'.format(sum(all_parameters)))
         print('-----------------------------------------------')    
         
    def update_learning_rate(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        #print('learning rate = %.7f' % lr)
        #print('-----------------------------------------------')    
        for scheduler in self.schedulers:
             if self.args.lr_policy == 'plateau':
                 scheduler.step()
             else:
                 scheduler.step()
         
        
        
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names: #在get_information里面定义了self.visual_names
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret


    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:  #在get_information里面定义了self.loss_names
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret
    
    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]['lr'] 
        return lr

    
    
if __name__ == "__main__":
    
   pass
