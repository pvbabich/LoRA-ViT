from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class KernelDataset(Dataset):
    def __init__(self, folder, mode='train'):
        self.mode = mode
        self.Trans = T.Compose([
            T.Resize(size=(384, 384)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        self.image_paths = glob(f"{folder}/images/*")
        self.mask_paths = glob(f"{folder}/mask_paths/*")
        self.num_image = len(self.image_paths)

    def __len__(self):
        return self.num_image

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.Trans(img)
        masks_image = Image.open(self.mask_paths[idx]).convert('RGB')
        masks_array = np.array(masks_image)

        colors = np.unique(masks_array.reshape(-1, img.shape[2]), axis=0)
        masks = [masks_array == color for color in colors]

        return img, masks
