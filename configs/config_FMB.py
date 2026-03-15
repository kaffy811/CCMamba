import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# ==================== 数据集配置 ====================
C.dataset_name = 'FMB'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'FMB')

# 指向根目录，让 Dataloader 自动拼凑 train/test 子目录
C.rgb_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.x_root_folder = C.dataset_path

C.rgb_format = '.png'
C.x_format = '.png'
C.gt_format = '.png'

# 标签背景值为 0，做减 1 变换使其变成 255 (ignore_index)
C.gt_transform = True 
C.x_is_single_channel = True 

# 核心改变：直接把 source 设为真实的物理文件夹
C.train_source = osp.join(C.dataset_path, "train", "Visible")
C.eval_source = osp.join(C.dataset_path, "test", "Visible")
C.is_test = False

# 自动扫描文件夹并获取真实的图片数量
if os.path.exists(C.train_source):
    C.num_train_imgs = len([f for f in os.listdir(C.train_source) if f.endswith(C.rgb_format)])
else:
    C.num_train_imgs = 1220

if os.path.exists(C.eval_source):
    C.num_eval_imgs = len([f for f in os.listdir(C.eval_source) if f.endswith(C.rgb_format)])
else:
    C.num_eval_imgs = 280

# 评估前景的 14 个类别
C.num_classes = 14
C.class_names = ['road', 'sidewalk', 'building', 'curb', 'fence', 'pole', 'vegetation', 'terrain', 'sky', 'person', 'car', 'bike', 'motorcycle', 'other']

# ==================== 图像与训练配置 ====================
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

C.backbone = 'sigma_tiny' 
C.pretrained_model = None 
C.decoder = 'MambaDecoder' 
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 2
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 4
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

# ==================== CCMamba 掩码蒸馏配置 (冲 SOTA 核心) ====================
C.mask_prob = 0.5          
C.mask_patch_size = 32     
C.mask_complementary = True  
C.distill_temperature = 2.0  
C.distill_alpha = 1.0      
C.distill_start_epoch = 20 # 必须从 20 开始热身
C.mask_apply_prob = 1.0    

# ==================== 评估与存储配置 (冲 SOTA 核心) ====================
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.75, 1, 1.25] # TTA 多尺度测试
C.eval_flip = True                   # TTA 镜像测试
C.eval_crop_size = [480, 640] 

C.checkpoint_start_epoch = 50
C.checkpoint_step = 5

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_fmb/' + 'log_' + C.dataset_name + '_' + C.backbone + '_' + 'cromb_conmb_cvssdecoder')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'
