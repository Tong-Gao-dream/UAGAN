import os
import cv2 as cv
import torch
import numpy as np

class SegmentationDataset(object):
    def __init__(self, image_dir, mask_dir):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        sfiles = os.listdir(mask_dir)
        for i in range(len(sfiles)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, sfiles[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.masks.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        # 输入图像
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0)

        # 目标标签0 ~ 1， 对于
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)
        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), }
        return sample
