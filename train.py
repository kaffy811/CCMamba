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
                # Adaptive Mask Distillation Strategy
                # Parameters are read from config to adapt to dataset characteristics:
                #   - MFNet: aggressive masking (mask_prob=0.5, complementary=True) from epoch 0
                #   - PST900: gentle masking (mask_prob=0.15, complementary=False) from epoch 200
                # ========================================================
                mask_apply_prob = getattr(config, 'mask_apply_prob', 1.0)
                distill_start_epoch = getattr(config, 'distill_start_epoch', 0)

                use_mask_distill = (
                    epoch >= distill_start_epoch and
                    torch.rand(1).item() < mask_apply_prob
                )

                if use_mask_distill:
                    # Step A: Teacher generates soft labels from the complete (unmasked) image
                    model.eval()
                    with torch.no_grad():
                        clean_logits = model(imgs, modal_xs)
                    model.train()

                    # Step B: Generate block mask using config parameters
                    mask_prob = getattr(config, 'mask_prob', 0.5)
                    patch_size = getattr(config, 'mask_patch_size', 32)
                    mask_complementary = getattr(config, 'mask_complementary', True)

                    batch_size, _, height, width = imgs.shape
                    rand_tensor = torch.rand(
                        (batch_size, 1, height // patch_size, width // patch_size),
                        device=imgs.device
                    )
                    small_mask = (rand_tensor > mask_prob).float()
                    mask = F.interpolate(small_mask, size=(height, width), mode='nearest')

                    imgs_masked = imgs * mask
                    if mask_complementary:
                        # Complementary masking: RGB sees mask, thermal sees (1-mask)
                        # Forces the model to learn cross-modal compensation (effective for MFNet)
                        modal_xs_masked = modal_xs * (1 - mask)
                    else:
                        # Same-location masking: both modalities mask the same region
                        # Simulates regional occlusion while preserving fusion ability (for PST900)
                        modal_xs_masked = modal_xs * mask

                    # Step C: Student forward pass on masked inputs
                    masked_logits = model(imgs_masked, modal_xs_masked)

                    # Step D: Compute combined loss (CE + distillation)
                    loss_ce = criterion(masked_logits, gts.long())

                    T = getattr(config, 'distill_temperature', 2.0)
                    prob_clean = F.softmax(clean_logits / T, dim=1)
                    log_prob_masked = F.log_softmax(masked_logits / T, dim=1)
                    # KL divergence distillation loss scaled by T^2.
                    # Use reduction='none' then .mean() to normalise over all dimensions
                    # (B×C×H×W), keeping the loss magnitude comparable to CE loss.
                    # 'batchmean' only divides by B, making it ~H*W times larger and
                    # causing gradient explosion for dense-prediction tasks.
                    loss_distill = F.kl_div(log_prob_masked, prob_clean, reduction='none').mean() * (T * T)

                    # Progressively increase distillation weight from 0 to distill_alpha.
                    # ramp_epochs covers the period from distill_start_epoch to nepochs.
                    # The factor of 2 means alpha reaches its full value at the midpoint of the
                    # ramp period, ensuring the model trains with full distillation strength for
                    # the second half of the distillation phase.
                    distill_alpha = getattr(config, 'distill_alpha', 1.0)
                    epochs_since_start = epoch - distill_start_epoch
                    ramp_epochs = max(1, config.nepochs - distill_start_epoch)
                    alpha = distill_alpha * min(1.0, (epochs_since_start / ramp_epochs) * 2)

                    # Anomaly guard: if distillation loss is unexpectedly large
                    # (more than distill_anomaly_threshold× the CE loss), skip it and
                    # fall back to CE-only training to prevent a single bad step from
                    # destroying weights.
                    anomaly_threshold = getattr(config, 'distill_anomaly_threshold', 10.0)
                    if loss_distill.item() > anomaly_threshold * loss_ce.item():
                        loss = loss_ce
                    else:
                        loss = loss_ce + alpha * loss_distill
                else:
                    # Standard training without mask distillation
                    out = model(imgs, modal_xs)
                    loss = criterion(out, gts.long())

                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                
                optimizer.zero_grad()
                loss.backward()
                grad_clip_norm = getattr(config, 'grad_clip_max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
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
