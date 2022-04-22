from re import L
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from augmentation import *


class CityScapeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir) # ['val', 'train']
        print(len(self.list_files))
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))

        x_image = image[:, :256, :] # 절반만 가져오기 (512, 256)
        y_image = image[:, 256:, :]

        augs = transform_both(image=x_image, image0=y_image)
        x_image, y_image = augs["image"], augs["image0"]

        x_image = transform_input(image=x_image)["image"]
        y_image = transform_target(image=y_image)["image"]

        return x_image, y_image

    def __len__(self):
        return len(self.list_files)


def test():
    train_dataset = CityScapeDataset("./cityscapes/train")

if __name__ == '__main__':
    test()