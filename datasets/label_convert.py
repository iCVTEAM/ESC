import numpy as np
import torch

class LabelConvert_Carla11:
    def __init__(self):
        self.label_list = [255, 1, 2, 255, 3, 4, 5, 5, 6, 7, 8, 9, 10, 0, 255, 255, 255, 255, 10, 255, 255, 255, 7]
        # self.label_dict = dict(zip(range(len(self.label_list)), self.label_list))
        self.label_dict = dict(enumerate(self.label_list))
        self.label_dict[255] = 255

    def __call__(self, seg):
        return torch.tensor(np.vectorize(self.label_dict.get)(seg), dtype=torch.uint8)
    

class NoConvert:
    def __init__(self):
        pass

    def __call__(self, seg):
        return seg