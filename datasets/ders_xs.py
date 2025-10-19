import os
import random
import numpy as np
import cv2
import glob
from PIL import Image
import torch
import logging
from torch.utils.data import Dataset
from pathlib import Path

from . import label_convert


class DERS_XS(Dataset):
    def __init__(self, global_config, config, transform):
        self.data_path = global_config['data_path']
        self.width = global_config['width']
        self.height = global_config['height']
        self.keys = config['data_keys']
        self.inputs = config['data_inputs']
        self.partition = config['partition']

        rgb_key, dvs_key, seg_key = self.inputs
        self.rgb_files = sorted(glob.glob(os.path.join(*[self.data_path, self.partition, rgb_key, '*.png'])))
        self.dvs_files = [file.replace(rgb_key, dvs_key).replace('.png', '.npy') for file in self.rgb_files]
        self.seg_files = [file.replace(rgb_key, seg_key) for file in self.rgb_files]
        self.ids = [Path(file).stem for file in self.rgb_files]
        
        self.data = [{'RGB': self.rgb_files[i], 'DVS': self.dvs_files[i], 'SEG': self.seg_files[i], 'ID': self.ids[i]} for i in range(len(self.rgb_files))]

        self.dummy_rgb, self.dummy_dvs, self.dummy_seg = config['dummy']

        self.label_convert = global_config['label_convert']
        self.transform = transform
        self.mask_value = config['mask_value'] if 'mask_value' in config else None
        self.e_mask_value = config['e_mask_value'] if 'e_mask_value' in config else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        sample = {}
        for key in self.keys:
            sample[key] = self.read_data(data[key], key)
        assert len(sample['SEG'].shape) == 2, 'Masks must be encoded without colourmap'
        sample['SEG'] = np.expand_dims(sample['SEG'], axis=0)
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])
        sample = self.transform(sample)
        sample = self.expand_sample(sample)
        sample['ID'] = data['ID']
        return sample
    
    def expand_sample(self, sample):
        seg_uint8 = sample['SEG'][0].numpy()
        seg_float32 = seg_uint8.astype(np.float32)
        boundary = cv2.absdiff(seg_float32, cv2.blur(seg_float32, (3, 3)))
        boundary = (boundary > 0).astype(np.uint8)
        boundary = np.expand_dims(boundary, axis=0)
        sample['SEG_BY'] = torch.tensor(boundary, dtype=torch.uint8)
        sample['SEG'] = getattr(label_convert, self.label_convert)()(sample['SEG'])
        return sample

    def read_data(self, path, key):
        if key == 'RGB':
            if self.dummy_rgb:
                return np.zeros((3, self.height, self.width), dtype=np.uint8)
            rgb = np.array(Image.open(path).convert('RGB')).transpose(2, 0, 1)
            if self.mask_value is not None:
                if isinstance(self.mask_value, list):
                    i, j, h, w = self.mask_value
                    rgb[:, j: j + w, i: i + h] = 0
                elif isinstance(self.mask_value, int):
                    rgb[rgb < self.mask_value] = 0
            return rgb
        elif key == 'DVS':
            if self.dummy_dvs:
                return np.zeros((5, self.height, self.width), dtype=np.float32)
            dvs = np.load(path)
            if self.e_mask_value is not None:
                if isinstance(self.e_mask_value, list):
                    i, j, h, w = self.e_mask_value
                    dvs[:, j: j + w, i: i + h] = 0
            return dvs
        elif key == 'SEG':
            if self.dummy_seg:
                return np.zeros((self.height, self.width), dtype=np.uint8)
            try:
                seg = np.array(Image.open(path))
                if len(seg.shape) == 3:
                    seg = seg[:, :, 0]
            except:
                seg = np.zeros((self.height, self.width), dtype=np.uint8)
            return seg