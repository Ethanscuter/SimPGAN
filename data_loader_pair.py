import os
import torch.utils.data as data
import numpy as np
from skimage import io
from scipy import misc
from numpy.random import randint, shuffle, choice
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn


# Hyper Parameters
input_size = 224
num_classes = 751
num_epochs = 300
steps_per_epoch = 16530  # 33000
batch_size = 10
learning_rate = 0.001


class ReidDataPrepare():
    def __init__(self, data_list_path, train_dir_path):
        self.list = data_list_path
        self.dir = train_dir_path

    def return_class_img_labels(self):
        class_img_labels = dict()
        class_cnt = -1
        last_label = -2
        with open(self.list, 'r') as f:
            for line in f:
                line = line.strip()
                img = line
                lbl = int(line.split('_')[0])
                if lbl != last_label:
                    class_cnt = class_cnt + 1
                    cur_list = list()
                    class_img_labels[str(class_cnt)] = cur_list
                last_label = lbl

                img_name = os.path.join(self.dir, img)
                image = io.imread(img_name)
                image = misc.imresize(image, size=(224, 224))
                image = np.expand_dims(image, axis=0)

                class_img_labels[str(class_cnt)].append(image[0])

        return class_img_labels


class SiameseNetworkDataset(data.Dataset):
    def __init__(self, train_list, train_dir, transforms=None):
        # Initialize file path or list of file names.
        self.list_path = train_list
        self.root_dir = train_dir
        self.transform = transforms
        self.class_img_labels = ReidDataPrepare(self.list_path, self.root_dir).return_class_img_labels()

        self.left_images = list()
        self.right_images = list()

        self.label1 = list()
        self.label2 = list()

        self.cur_epoch = 0
        self.pos_prop = 4

        for m in range(int(steps_per_epoch / batch_size)):
            self.left_label = randint(len(self.class_img_labels), size=batch_size)
            if self.cur_epoch % self.pos_prop == 0:
                self.right_label = self.left_label
            else:
                self.right_label = np.copy(self.left_label)
                shuffle(self.right_label)
            slice_start = 0
            for i in range(batch_size):
                len_left_label_i = len(self.class_img_labels[str(self.left_label[i])])
                self.left_images.append(self.class_img_labels[str(self.left_label[i])]
                                        [int(slice_start * len_left_label_i):]
                                        [choice(len_left_label_i - int(len_left_label_i * slice_start))])
                len_right_label_i = len(self.class_img_labels[str(self.right_label[i])])
                self.right_images.append(self.class_img_labels[str(self.right_label[i])]
                                        [int(slice_start * len_right_label_i):]
                                        [choice(len_right_label_i - int(len_right_label_i * slice_start))]
                                        )

            # self.label1.append(self.left_label.tolist())
            # self.label2.append(self.right_label.tolist())
            self.label1 += self.left_label.tolist()
            self.label2 += self.right_label.tolist()

            self.cur_epoch += 1

        self.label1 = np.array(self.label1, dtype=int)
        self.label2 = np.array(self.label2, dtype=int)
        # transform before the the construct dataset
        self.left_images = np.array(self.left_images, dtype=float)
        self.right_images = np.array(self.right_images, dtype=float)
        self.binary_label = (self.label1 == self.label2).astype(int)

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

        img_left, img_right, target_left, target_right, target_binary = self.left_images[index], \
                                                                        self.right_images[index], \
                                                                        self.label1[index], \
                                                                        self.label2[index], \
                                                                        self.binary_label[index]

        img_left = Image.fromarray(img_left.astype('uint8'))
        img_right = Image.fromarray(img_right.astype('uint8'))

        if self.transform is not None:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        # Before make samples, image and labels should be numpy data structure
        return img_left, img_right, target_left, target_right, target_binary

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.left_images)


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
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    market_dataset = SiameseNetworkDataset(train_list=market_list,
                                           train_dir=market_path,
                                           transforms=data_transform)

    grid_dataset = SiameseNetworkDataset(train_list=grid_list,
                                         train_dir=grid_path,
                                         transforms=data_transform)

    market_loader = torch.utils.data.DataLoader(dataset=market_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)

    grid_loader = torch.utils.data.DataLoader(dataset=grid_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    return market_loader, grid_loader


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
