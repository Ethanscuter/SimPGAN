from torchvision import datasets
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from skimage import io
from scipy import misc
from PIL import Image


class PersonReidDataset(data.Dataset):
    """Person Reid dataset

    Args:
        train_list(string): Directory of dataset label .list file
        train_dir(string): Directory of dataset where consists of person images
        transform(callable, optional): A function/transform that takes in an PIL image and returns a transform version.
        target_transform(callable,optional): A function/transform that takes in the target and transform it.

    """

    def __init__(self, train_list, train_dir, transform=None):
        # Initialize file path or list of file name

        self.list_path = train_list
        self.root_dir = train_dir
        self.transform = transform

        # now load the picked numpy arrarys
        self.images = []
        self.labels = []
        with open(self.list_path, 'r') as f:
            last_label = -1
            label_cnt = -1
            for line in f:
                line = line.strip()
                img = line
                lbl = line.split('_')[0]
                if last_label != lbl:
                    label_cnt += 1
                last_label = lbl
                img_name = os.path.join(self.root_dir, img)
                image = io.imread(img_name)
                image = misc.imresize(image, size=(224, 224))
                image = np.expand_dims(image, axis=0)

                self.images.append(image[0])
                self.labels.append(label_cnt)

        self.images = np.array(self.images, dtype=float)
        self.labels = np.array(self.labels, dtype=int)

        """Pytorch do not need one-hot encode"""
        # num = np.max(self.labels) + 1
        # n = self.labels.shape[0]
        # categorical = np.zeros((n, num))
        # categorical[np.arange(n), self.labels] = 1
        # categorical = categorical.astype('int64')

        # self.labels = categorical

    def __getitem__(self, index):
        # Read one data from file
        # Preprocess the data (torchvision.Transform)
        # Return a data pair (image and label)
        """
        Args:
            param index(int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[index], self.labels[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        # Before make samples, image and labels should be numpy data structure
        return img, target

    def __len__(self):
        return len(self.images)


def get_loader(image_size, market_path, grid_path, market_list, grid_list, batch_size, num_workers):
    """
    Build and returns Dataloader for MNIST and SVHN dataset.
    """
    data_transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        transforms.Scale([286, 286], Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    market_dataset = PersonReidDataset(train_list=market_list,
                                       train_dir=market_path,
                                       transform=data_transform)

    grid_dataset = PersonReidDataset(train_list=grid_list,
                                     train_dir=grid_path,
                                     transform=data_transform)

    market_loader = torch.utils.data.DataLoader(dataset=market_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

    grid_loader = torch.utils.data.DataLoader(dataset=grid_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    return market_loader, grid_loader


class TestDataset(data.Dataset):
    def __init__(self, test_dir, transform=None):
        self.images = []
        self.infos = []
        self.test_dir = test_dir
        self.transform = transform

        for image_name in sorted(os.listdir(self.test_dir)):
            if '.txt' in image_name or '.db' in image_name:
                continue
            if 's' not in image_name:
                # grid
                arr = image_name.split('_')
                person = int(arr[0])
                camera = int(arr[1])
            elif 's' in image_name:
                # market
                arr = image_name.split('_')
                person = int(arr[0])
                camera = int(arr[1][1])
            else:
                continue
            img_name = os.path.join(self.test_dir, image_name)
            image = io.imread(img_name)
            image = misc.imresize(image, size=(224, 224))
            image = np.expand_dims(image, axis=0)

            self.images.append(image[0])
            self.infos.append((person, camera))

        self.images = np.array(self.images, dtype=float)

    def __getitem__(self, index):

        img, info = self.images[index], self.infos[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        return img, info

    def __len__(self):
        return len(self.images)


def get_test_loader(dir_path):
    data_transform = transforms.Compose([
        transforms.Scale([256, 256], Image.BICUBIC),
        # transforms.RandomCrop(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(test_dir=dir_path,
                               transform=data_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)
    return test_loader


class SampleDataset(data.Dataset):
    def __init__(self, train_dir, transform=None):
        self.root_dir = train_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for image_name in sorted(os.listdir(self.root_dir)):
            self.labels.append(image_name)
            img_name = os.path.join(self.root_dir, image_name)
            image = io.imread(img_name)
            image = misc.imresize(image, size=(224, 224))
            image = np.expand_dims(image, axis=0)

            self.images.append(image[0])

        self.images = np.array(self.images, dtype=float)

    def __getitem__(self, index):

        img, target = self.images[index], self.labels[index]

        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        # Before make samples, image and labels should be numpy data structure
        return img, target

    def __len__(self):
        return len(self.images)


def get_sample_loader(dir_path):
    data_transform = transforms.Compose([
        transforms.Scale([286, 286], Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    sample_dataset = SampleDataset(train_dir=dir_path,
                                   transform=data_transform)

    sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0)
    return sample_loader

