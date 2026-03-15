import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
            
        # ==================== 新增：动态路径匹配逻辑 ====================
        if 'FMB' in self._rgb_path:
            # 针对 FMB 数据集，自动拼接到 train 或 test 的特定子文件夹中
            split_dir = 'train' if self._split_name == 'train' else 'test'
            rgb_path = os.path.join(self._rgb_path, split_dir, 'Visible', item_name + self._rgb_format)
            x_path = os.path.join(self._x_path, split_dir, 'Infrared', item_name + self._x_format)
            gt_path = os.path.join(self._gt_path, split_dir, 'Label', item_name + self._gt_format)
        else:
            # 兼容其他数据集（如 PST900），保持原样直接在根目录寻找
            rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
            x_path = os.path.join(self._x_path, item_name + self._x_format)
            gt_path = os.path.join(self._gt_path, item_name + self._gt_format)
        # ================================================================

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt) 

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x =  self._open_image(x_path, cv2.COLOR_BGR2RGB)
        
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        # 增加 'test' 以防止 eval.py 传递 'test' 参数时报错
        assert split_name in ['train', 'val', 'test']
        source = self._train_source
        if split_name in ["val", "test"]:
            source = self._eval_source

        file_names = []
        
        # ==================== 新增：自动扫描文件夹逻辑 ====================
        if os.path.isdir(source):
            # 如果传入的是个文件夹（比如 FMB 传的 Visible 目录），直接扫描里面所有的图片
            files = os.listdir(source)
            for item in files:
                if item.endswith(self._rgb_format):
                    # 剥离后缀，提取纯文件名 (如 2007_000032)
                    file_names.append(os.path.splitext(item)[0])
            file_names.sort()  # 排序以保证 RGB 和 Thermal 是严格对应的
            
        elif source.endswith('.txt'):
            # 兼容旧逻辑：如果传进来的是 .txt 文件，还是按行读取
            with open(source) as f:
                files = f.readlines()
            for item in files:
                file_name = item.strip()
                file_names.append(file_name)
        else:
            raise ValueError(f"Invalid source (not a directory or .txt file): {source}")
        # ================================================================

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
