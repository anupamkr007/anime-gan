import torch
from utils import *

class AnimeFaceDataset(Dataset):

    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.images = filesInDir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images[idx])

        image = io.imread(img_name)

        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


