import os
import cv2
from keras.utils import to_categorical
from torch.utils.data import DataLoader, Dataset

class ImageNetDataset(Dataset):

    def __init__(self, root_dir, folder_to_label, transform=None):

        self.samples = self.find_all_images(root_dir, folder_to_label)
        self.transform = transform

    def find_all_images(self, root_dir, folder_to_label):

        samples = []

        for class_dir in os.listdir(root_dir):

            label = folder_to_label[class_dir]
            dir_path = os.path.join(root_dir, class_dir)

            for f in os.listdir(dir_path):
                image_path = (os.path.join(dir_path, f))
                samples.append((image_path, label))
        return samples

    def imread(self, path):
        return cv2.imread(path)[..., ::-1]

    def __getitem__(self, i):

        sample = {
            'image': self.imread(self.samples[i][0]),
            'label': to_categorical(int(self.samples[i][1]), 1000),
        }

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self.samples)