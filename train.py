import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F  

from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from eval import SegEvaluator
import shutil

from tensorboardX import SummaryWriter

# ==========================================
# 新增：将主逻辑封装到 main 函数中
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    logger = get_logger()

    os.environ['MASTER_PORT'] = '16005'

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        print(args)
        
        dataset_name = args.dataset_name
        if dataset_name == 'mfnet':
            from configs.config_MFNet import config
        elif dataset_name == 'pst':
            from configs.config_pst900 import config
        elif dataset_name == 'nyu':
            from configs.config_nyu import config
        elif dataset_name == 'sun':
            from configs.config_sunrgbd import config
        else:
            raise ValueError('Not a valid dataset name')

        print("=======================================")
        print(config.tb_dir)
        print("=======================================")

        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # data loader
        train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        # config network and criterion
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

        if engine.distributed:
            BatchNorm2d = nn.SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d
        
        model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
        
        # group weight and config optimizer
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr
        
        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        if engine.distributed:
            logger.info('.............distributed training.............')
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                                output_device=engine.local_rank, find_unused_parameters=False)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model,
                              optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')
        
        # Initialize the evaluation dataset and evaluator
        val_setting = {'rgb_root': config.rgb_root_folder,
                        'rgb_format': config.rgb_format,
                        'gt_root': config.gt_root_folder,
                        'gt_format': config.gt_format,
                        'transform_gt': config.gt_transform,
                        'x_root':config.x_root_folder,
                        'x_format': config.x_format,
                        'x_single_channel': config.x_is_single_channel,
                        'class_names': config.class_names,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'class_names': config.class_names}
        val_pre = ValPre()
        val_dataset = RGBXDataset(val_setting, 'val', val_pre)

        best_mean_iou = 0.0  # Track the best mean IoU for model saving
        best_epoch = 100000  # Track the epoch with the best mean IoU for model saving
        
        for epoch in range(engine.state.epoch, config.nepochs+1):
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)

            sum_loss = 0

            for idx in pbar:
                engine.update_iteration(epoch, idx)

                minibatch = next(dataloader)
                imgs = minibatch['data']
                gts = minibatch['label']
                modal_xs = minibatch['modal_x']

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                modal_xs = modal_xs.cuda(non_blocking=True)

                # ========================================================
                # 步骤 A: 教师模型（Teacher）生成完整图像的软标签 
                # ========================================================
                model.eval()
                with torch.no_grad():
                    clean_logits = model(imgs, modal_xs)
                model.train()

                # ========================================================
                # 步骤 B: 引入互补随机掩码 (CRM)
                # ========================================================
                batch_size, _, height, width = imgs.shape
                patch_size = 32  
                mask_prob = 0.5  
                
                rand_tensor = torch.rand((batch_size, 1, height // patch_size, width // patch_size), device=imgs.device)
                small_mask = (rand_tensor > mask_prob).float()
                mask = F.interpolate(small_mask, size=(height, width), mode='nearest')
                
                imgs_masked = imgs * mask
                modal_xs_masked = modal_xs * (1 - mask)

                # ========================================================
                # 步骤 C: 学生模型（Student）提取掩码图像的预测 (有梯度)
                # ========================================================
                masked_logits = model(imgs_masked, modal_xs_masked)

                # ========================================================
                # 步骤 D: 计算联合损失 (交叉熵损失 + 自蒸馏损失)
                # ========================================================
                loss_ce = criterion(masked_logits, gts.long())
                
                T = 2.0  # 蒸馏温度参数
                prob_clean = F.softmax(clean_logits / T, dim=1)
                log_prob_masked = F.log_softmax(masked_logits / T, dim=1)
                
                valid_mask = (gts != config.background).unsqueeze(1).float()
                
                loss_distill_raw = F.kl_div(log_prob_masked, prob_clean, reduction='none') * (T * T)
                loss_distill = (loss_distill_raw * valid_mask).sum() / (valid_mask.sum() * config.num_classes + 1e-8)
                
                alpha = 1.0  
                loss = loss_ce + alpha * loss_distill
                # ========================================================

                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_idx = (epoch- 1) * config.niters_per_epoch + idx 
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                if engine.distributed:
                    if dist.get_rank() == 0:
                        sum_loss += reduce_loss.item()
                        print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                        pbar.set_description(print_str, refresh=False)
                else:
                    sum_loss += loss
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                    pbar.set_description(print_str, refresh=False)
                del loss
                
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
            
            torch.cuda.empty_cache()
            if engine.distributed:
                if dist.get_rank() == 0:
                    if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                        model.eval() 
                        with torch.no_grad():
                            all_dev = parse_devices(args.devices)
                            segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                    norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                    network=model, multi_scales=config.eval_scale_array,
                                                    is_flip=config.eval_flip, devices=[model.device],
                                                    verbose=False, config=config,
                                                    )
                            _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                        config.link_val_log_file)
                            print('mean_IoU:', mean_IoU)
                            
                            if mean_IoU > best_mean_iou:
                                checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                                if os.path.exists(checkpoint_path):
                                    os.remove(checkpoint_path)
                                best_epoch = epoch
                                best_mean_iou = mean_IoU
                            else:
                                checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                                if os.path.exists(checkpoint_path):
                                    os.remove(checkpoint_path)
                            
                        model.train()
            else:
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        devices_val = [engine.local_rank] if engine.distributed else [0]
                        segmentor = SegEvaluator(dataset=val_dataset, class_num=config.num_classes,
                                                norm_mean=config.norm_mean, norm_std=config.norm_std,
                                                network=model, multi_scales=config.eval_scale_array,
                                                # ==========================================
                                                # 修正：将 devices=[1,2,3] 修改为 [0]
                                                # ==========================================
                                                is_flip=config.eval_flip, devices=[0], 
                                                verbose=False, config=config,
                                                )
                        _, mean_IoU = segmentor.run(config.checkpoint_dir, str(epoch), config.val_log_file,
                                    config.link_val_log_file)
                        print('mean_IoU:', mean_IoU)
                        
                        if mean_IoU > best_mean_iou:
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            best_epoch = epoch
                            best_mean_iou = mean_IoU
                        else:
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                    model.train()

# ==========================================
# 新增：保护主进程，防止多进程递归导入
# ==========================================
if __name__ == '__main__':
    main()
