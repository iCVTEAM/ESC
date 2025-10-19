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


class DSEC(Dataset):
    def __init__(self, global_config, config, transform):
        self.data_path = global_config['data_path']
        self.width = global_config['width']
        self.height = global_config['height']
        self.keys = config['data_keys']
        self.inputs = config['data_inputs']
        self.partition = config['partition']

        assert self.partition in ['Train', 'Valid', 'Test']
        if self.partition == 'Train':
            self.seqs = ['zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 
                         'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a', 
                         'zurich_city_07_a', 'zurich_city_08_a']
        else:
            self.seqs = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']

        rgb_key, dvs_key, seg_key = self.inputs
        self.rgb_files = []
        for seq in self.seqs:
            self.rgb_files.extend(sorted(glob.glob(os.path.join(*[self.data_path, seq, rgb_key, '*.png']))))
        self.rgb_files = list(filter(lambda path: not path.endswith(
            ('00000000.png', '00000002.png', '00000004.png', '1.png', '3.png', '5.png', '7.png', '9.png')
            ), self.rgb_files))
        self.dvs_files = [file.replace(rgb_key, dvs_key).replace('.png', '.npy') for file in self.rgb_files]
        self.seg_files = [file.replace(rgb_key, seg_key) for file in self.rgb_files]
        self.ids = [Path(file).stem for file in self.rgb_files]
        
        self.data = [{'RGB': self.rgb_files[i], 'DVS': self.dvs_files[i], 'SEG': self.seg_files[i], 'ID': self.ids[i]} for i in range(len(self.rgb_files))]

        self.dummy_rgb, self.dummy_dvs, self.dummy_seg = config['dummy']

        self.label_convert = global_config['label_convert']
        self.transform = transform

        logging.info(f'Found {len(self.data)} Frames for {self.partition}.')

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
            return rgb
        elif key == 'DVS':
            if self.dummy_dvs:
                return np.zeros((5, self.height, self.width), dtype=np.float32)
            dvs = np.load(path)
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